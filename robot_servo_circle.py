  #!/usr/bin/env python3
"""
识别画面中最像圆的物体并追踪到画面中心
检测策略：边缘检测 → 轮廓提取 → 圆形度评分（形状为主，颜色为辅）
- Canny边缘检测提取所有轮廓
- 圆形度 + 凸度双重验证
- 帧间连续性追踪防跳变
"""
import cv2
import time
import numpy as np
import Adafruit_PCA9685

# ==================== 配置参数 ====================
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_CX = FRAME_WIDTH // 2   # 160
FRAME_CY = FRAME_HEIGHT // 2  # 120

# 舵机PWM范围
SERVO_MIN = 150
SERVO_MAX = 650
SERVO_CENTER = 300   # 下方舵机(5) 初始角度90° (PWM=650-300=350)
SERVO_CENTER_Y = 527 # 上方舵机(4) 初始角度180° (PWM=650-100=550)

# PID参数 (X轴 - 水平, Y轴 - 垂直)
PID_X_KP = 0.040
PID_X_KD = 0.015
PID_X_KI = 0.002

PID_Y_KP = 0.040
PID_Y_KD = 0.015
PID_Y_KI = 0.002

# 死区（像素）：误差在此范围内不调整，防止微抖
DEAD_ZONE_X = 5
DEAD_ZONE_Y = 5

# 每帧最大变化量（PWM单位），越小越慢
MAX_DELTA_X = 5
MAX_DELTA_Y = 5

# 帧率限制（每秒处理帧数，设为0则不限制）
FPS_LIMIT = 30

# ==================== 圆形检测参数（Canny + fitEllipse，空心圆） ====================
CANNY_LOW = 50
CANNY_HIGH = 150

# 椭圆长短轴比（1.0=正圆，排除太扁的）
MIN_AXIS_RATIO = 0.5

# 面积
MIN_AREA = 20
MAX_AREA = 20000

# 最小半径
MIN_RADIUS = 15

# 帧间追踪：目标在连续帧间最大移动像素
MAX_TRACK_DIST = 80
# 连续多少帧未检测到目标才算丢失
LOST_FRAME_LIMIT = 10
# EMA 平滑系数
EMA_ALPHA = 0.3

# ==================== 初始化舵机 ====================
servo_pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
servo_pwm.set_pwm_freq(60)
servo_pwm.set_pwm(5, 0, 650 - SERVO_CENTER)    # 下方舵机 90° (PWM=350)
servo_pwm.set_pwm(4, 0, 650 - SERVO_CENTER_Y)  # 上方舵机 180° (PWM=550)
time.sleep(1)

# ==================== 工具函数 ====================
def find_best_circle(frame):
    """Canny + fitEllipse：边缘检测后拟合椭圆，空心圆也能抓"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_r = -1
    min_r = float('inf')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA or len(cnt) < 5:
            continue
        (cx, cy), (w, h), _ = cv2.fitEllipse(cnt)
        ratio = min(w, h) / max(w, h)
        r = min(w, h) / 2
        if ratio < MIN_AXIS_RATIO or r < MIN_RADIUS:
            continue
        #识别最大圆
        # if r > best_r:
        #     best_r = r
        #     best = (int(cx), int(cy), int(r))
        #识别最小圆
        if r < min_r:
            min_r = r
            best = (int(cx), int(cy), int(r))
    if best:
        return True, best[0], best[1], best[2], edges
    return False, 0, 0, 0, edges


# ==================== 帧间追踪器 ====================
class BallTracker:
    def __init__(self):
        self.sx = self.sy = self.sr = None
        self.lost = 0
        self.ok = False
    def update(self, found, cx, cy, r):
        if found:
            if self.sx is not None:
                if np.hypot(cx - self.sx, cy - self.sy) > MAX_TRACK_DIST:
                    self.lost += 1
                    if self.lost > LOST_FRAME_LIMIT: self.ok = False; return False
                    return self.ok
                self.lost = 0; self.ok = True
            else:
                self.lost = 0; self.ok = True
            self.sx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * (self.sx if self.sx is not None else cx)
            self.sy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * (self.sy if self.sy is not None else cy)
            self.sr = EMA_ALPHA * r  + (1 - EMA_ALPHA) * (self.sr if self.sr is not None else r)
            return True
        self.lost += 1
        if self.lost > LOST_FRAME_LIMIT: self.ok = False
        return self.ok
    def pos(self):
        return (int(self.sx) if self.sx is not None else 0,
                int(self.sy) if self.sy is not None else 0,
                int(self.sr) if self.sr is not None else 0)


# ==================== PID控制器类 ====================
class PIDController:
    def __init__(self, kp, ki, kd, output_min=-SERVO_CENTER, output_max=SERVO_CENTER):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.last_error = 0
        self.integral = 0

    def update(self, error, dt=1.0):
        """更新PID，返回控制量"""
        # 比例项
        p_term = self.kp * error

        # 积分项（带限幅防饱和）
        self.integral += error * dt
        self.integral = max(-self.output_max, min(self.output_max, self.integral))
        i_term = self.ki * self.integral

        # 微分项
        d_term = self.kd * (error - self.last_error) / max(dt, 0.001)
        self.last_error = error

        # 总输出
        output = p_term + i_term + d_term
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        self.last_error = 0
        self.integral = 0


# ==================== 主循环 ====================
usb_cap = cv2.VideoCapture(0)
usb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
usb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

pid_x = PIDController(PID_X_KP, PID_X_KI, PID_X_KD)
pid_y = PIDController(PID_Y_KP, PID_Y_KI, PID_Y_KD)
tracker = BallTracker()

servo_x_pos = SERVO_CENTER   # 当前X轴舵机PWM
servo_y_pos = SERVO_CENTER_Y  # 当前Y轴舵机PWM

prev_time = time.time()
frame_interval = 1.0 / FPS_LIMIT if FPS_LIMIT > 0 else 0

while True:
    # 帧率控制
    now = time.time()
    if frame_interval > 0 and (now - prev_time) < frame_interval:
        time.sleep(0.001)
        continue
    dt = now - prev_time if FPS_LIMIT > 0 else 0.03
    prev_time = now

    ret, frame = usb_cap.read()
    if not ret:
        break

    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    found_raw, sx, sy, sr, _ = find_best_circle(frame)
    found = tracker.update(found_raw, sx, sy, sr)
    sx, sy, sr = tracker.pos()

    if found:
        cv2.circle(frame, (sx, sy), sr, (0, 255, 0), 2)
        cv2.circle(frame, (sx, sy), 5, (0, 255, 0), -1)
        error_x = sx - FRAME_CX
        error_y = sy - FRAME_CY

        control_x = 0 if abs(error_x) < DEAD_ZONE_X else pid_x.update(error_x, dt)
        control_y = 0 if abs(error_y) < DEAD_ZONE_Y else pid_y.update(error_y, dt)
        if abs(error_x) < DEAD_ZONE_X: pid_x.reset()
        if abs(error_y) < DEAD_ZONE_Y: pid_y.reset()

        # 方向修正：+
        new_servo_x = servo_x_pos + int(control_x)
        new_servo_y = servo_y_pos + int(control_y)

        # 限制每帧最大变化量
        servo_x_pos += max(-MAX_DELTA_X, min(MAX_DELTA_X, new_servo_x - servo_x_pos))
        servo_y_pos += max(-MAX_DELTA_Y, min(MAX_DELTA_Y, new_servo_y - servo_y_pos))

        # 限幅
        servo_x_pos = max(SERVO_MIN, min(SERVO_MAX, servo_x_pos))
        servo_y_pos = max(SERVO_MIN, min(SERVO_MAX, servo_y_pos))

        # 执行舵机控制
        servo_pwm.set_pwm(5, 0, 650 - servo_x_pos)
        servo_pwm.set_pwm(4, 0, 650 - servo_y_pos)

    # ===== 可视化 =====
    # 画面中心红色十字
    cv2.line(frame, (FRAME_CX - 20, FRAME_CY), (FRAME_CX + 20, FRAME_CY), (0, 0, 255), 2)
    cv2.line(frame, (FRAME_CX, FRAME_CY - 20), (FRAME_CX, FRAME_CY + 20), (0, 0, 255), 2)
    cv2.circle(frame, (FRAME_CX, FRAME_CY), 3, (0, 0, 255), -1)

    if found:
        cv2.line(frame, (FRAME_CX, FRAME_CY), (sx, sy), (255, 0, 255), 1)
        cv2.putText(frame, f"Offset: ({sx-FRAME_CX:+d}, {sy-FRAME_CY:+d})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Servo: ({servo_x_pos}, {servo_y_pos})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame, f"Lost: {tracker.lost}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)
    else:
        cv2.putText(frame, "No target", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, "Red + = frame center | Green O = black circle center", (10, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Black Circle Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q 或 ESC 退出
        break

usb_cap.release()
cv2.destroyAllWindows()  
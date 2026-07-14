#!/usr/bin/env python3
"""
* @par Copyright (C): 2010-2020, hunan CLB Tech
* @file         laser_align_circle
* @version      V1.0
* @details      蓝紫色激光对准圆心：检测画面中的圆形，通过PID控制舵机使激光对准圆心
* @par History
*
* @author:
"""

import cv2
import time
import numpy as np
import RPi.GPIO as GPIO
import Adafruit_PCA9685

# ================= 激光 GPIO 配置 =================

LASER_PIN = 17          # 激光控制引脚 (BCM编号)
LASER_ON = GPIO.LOW    # 低电平点亮（根据实际电路调整）
LASER_OFF = GPIO.HIGH

# ================= 通用参数 =================

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CENTER_X = FRAME_WIDTH // 2   # 160 (画面中心 = 激光理想瞄准点)
CENTER_Y = FRAME_HEIGHT // 2  # 120

# ================= 舵机初始化 =================

servo_pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
servo_pwm.set_pwm_freq(60)

# 底座舵机(水平) 和 倾斜舵机(垂直) 初始角度
SERVO_X_INIT = 350
SERVO_Y_INIT = 370

servo_pwm.set_pwm(5, 0, SERVO_X_INIT)  # 底座舵机 (水平旋转)
servo_pwm.set_pwm(4, 0, SERVO_Y_INIT)  # 倾斜舵机 (俯仰)
time.sleep(1)

# ================= 舵机 PID 变量 =================

pid_thisError_x = 0
pid_lastError_x = 0
pid_thisError_y = 0
pid_lastError_y = 0

pid_X_P = 300  # 当前水平角度
pid_Y_P = 280  # 当前俯仰角度

# PID 系数
KP = 3         # 比例系数
KD = 1         # 微分系数
PID_DIVISOR = 100  # 缩放除数

# 死区阈值（误差小于此值不调整，防止抖动）
DEAD_ZONE = 3

# ================= 圆形检测参数（轮廓法）=================

CANNY_LOW = 50
CANNY_HIGH = 150
MIN_AREA = 20
MAX_AREA = 30000
MIN_RADIUS_CONTOUR = 10
MIN_AXIS_RATIO = 0.75
CENTER_DISTANCE = 5

# ================= 圆形检测参数（霍夫圆）=================

HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 80
HOUGH_MIN_R = 10
HOUGH_MAX_R = 100

# ================= 激光光斑检测参数 =================
# 蓝紫色激光在 HSV 空间的范围（需根据实际激光颜色标定）

LASER_H_LOW = 100     # 蓝色色调下限
LASER_H_HIGH = 150    # 紫色色调上限
LASER_S_LOW = 100      # 饱和度下限
LASER_S_HIGH = 255
LASER_V_LOW = 200     # 明度下限（激光很亮）
LASER_V_HIGH = 255

LASER_BLOB_MIN_AREA = 3    # 激光光斑最小面积
LASER_BLOB_MAX_AREA = 500  # 激光光斑最大面积

# =================================================

# 摄像头初始化
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# =================================================


def laser_on():
    """打开激光"""
    GPIO.output(LASER_PIN, LASER_ON)
    print("[Laser] 激光已打开")


def laser_off():
    """关闭激光"""
    GPIO.output(LASER_PIN, LASER_OFF)
    print("[Laser] 激光已关闭")


def setup():
    """初始化 GPIO 和舵机"""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LASER_PIN, GPIO.OUT)
    laser_off()  # 初始关闭激光

    # 舵机归中
    servo_pwm.set_pwm(5, 0, SERVO_X_INIT)
    servo_pwm.set_pwm(4, 0, SERVO_Y_INIT)
    time.sleep(1)


def cleanup():
    """清理资源"""
    laser_off()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("[System] 资源已清理")


def Robot_servo(X_P, Y_P):
    """驱动舵机到指定角度"""
    servo_pwm.set_pwm(5, 0, 650 - X_P)
    servo_pwm.set_pwm(4, 0, 650 - Y_P)


def pid_control(circle_cx, circle_cy, laser_cx, laser_cy):
    """
    基于激光光斑与圆心的偏差计算 PID，使激光对准圆心
    核心思路：激光在舵机上可动，摄像头固定不动
             误差 = 圆心坐标 - 激光光斑坐标
             控制目标是让这个误差趋近于零（激光落在圆心上）
    """
    global pid_thisError_x, pid_lastError_x
    global pid_thisError_y, pid_lastError_y
    global pid_X_P, pid_Y_P

    # 误差 = 圆心 - 激光光斑（我们要让激光移动到圆心）
    pid_thisError_x = circle_cx - laser_cx
    pid_thisError_y = circle_cy - laser_cy

    # 死区：误差太小则不调整，防止舵机抖动
    if abs(pid_thisError_x) < DEAD_ZONE and abs(pid_thisError_y) < DEAD_ZONE:
        return  # 已对准，无需调整

    # PD 控制
    pwm_x = pid_thisError_x * KP + KD * (pid_thisError_x - pid_lastError_x)
    pwm_y = pid_thisError_y * KP + KD * (pid_thisError_y - pid_lastError_y)

    # 迭代误差
    pid_lastError_x = pid_thisError_x
    pid_lastError_y = pid_thisError_y

    # 更新角度（方向符号需根据实际舵机安装方向微调）
    pid_X_P = pid_X_P - int(pwm_x / PID_DIVISOR)
    pid_Y_P = pid_Y_P - int(pwm_y / PID_DIVISOR)

    # 限幅
    pid_X_P = max(0, min(650, pid_X_P))
    pid_Y_P = max(0, min(650, pid_Y_P))


def detect_laser_dot(hsv_frame):
    """
    检测蓝紫色激光光斑位置
    返回 (cx, cy, radius) 或 None
    """
    # 蓝紫色范围掩膜
    lower = np.array([LASER_H_LOW, LASER_S_LOW, LASER_V_LOW])
    upper = np.array([LASER_H_HIGH, LASER_S_HIGH, LASER_V_HIGH])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # 形态学去噪
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < LASER_BLOB_MIN_AREA or area > LASER_BLOB_MAX_AREA:
            continue

        # 最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)

        if area > best_area:
            best_area = area
            best = (cx, cy, radius)

    return best, mask


def detect_circles_contour(edges):
    """
    轮廓法检测圆形
    返回 circles_contour 列表 [(cx, cy, radius), ...] 和 groups 聚类列表
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    circles_contour = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if len(cnt) < 5:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.75:
            continue

        (cx, cy), (w, h), _ = cv2.fitEllipse(cnt)
        ratio = min(w, h) / max(w, h)
        if ratio < MIN_AXIS_RATIO:
            continue

        radius = (w + h) / 4
        if radius < MIN_RADIUS_CONTOUR:
            continue

        circles_contour.append((cx, cy, radius))

    # 聚类
    groups = []
    for cx, cy, r in circles_contour:
        found = False
        for group in groups:
            gx, gy = group["center"]
            d = np.hypot(cx - gx, cy - gy)
            if d < CENTER_DISTANCE:
                group["points"].append((cx, cy, r))
                n = len(group["points"])
                group["center"] = ((gx * (n - 1) + cx) / n, (gy * (n - 1) + cy) / n)
                found = True
                break
        if not found:
            groups.append({"center": (cx, cy), "points": [(cx, cy, r)]})

    return circles_contour, groups


def select_target(circles_contour, groups, hough_circles):
    """
    选择跟踪目标：轮廓聚类优先 > 单轮廓 > 霍夫圆
    返回 (cx, cy, radius, source) 或 (None, None, None, None)
    """
    # 优先：取聚类组中圈数最多的轮廓圆心
    best_group = None
    for g in groups:
        if len(g["points"]) >= 2:
            if best_group is None or len(g["points"]) > len(best_group["points"]):
                best_group = g

    if best_group is not None:
        cx = int(best_group["center"][0])
        cy = int(best_group["center"][1])
        r = int(np.mean([p[2] for p in best_group["points"]]))
        return cx, cy, r, "contour_group"

    if len(circles_contour) > 0:
        cx, cy, r = int(circles_contour[0][0]), int(circles_contour[0][1]), int(circles_contour[0][2])
        return cx, cy, r, "contour"

    if hough_circles is not None:
        hough_circles = np.round(hough_circles[0]).astype(int)
        cx, cy, r = int(hough_circles[0][0]), int(hough_circles[0][1]), int(hough_circles[0][2])
        return cx, cy, r, "hough"

    return None, None, None, None


def draw_overlay(frame, target_cx, target_cy, target_r, target_source,
                 laser_cx, laser_cy, laser_r,
                 circles_contour, groups, hough_circles):
    """绘制可视化叠加层"""
    display = frame.copy()

    # --- 画画面中心十字（摄像头光轴参考点）---
    cv2.line(display, (CENTER_X - 15, CENTER_Y), (CENTER_X + 15, CENTER_Y), (100, 100, 100), 1)
    cv2.line(display, (CENTER_X, CENTER_Y - 15), (CENTER_X, CENTER_Y + 15), (100, 100, 100), 1)

    # --- 画所有检测到的轮廓圆 ---
    for cx, cy, r in circles_contour:
        cx_i, cy_i, r_i = int(cx), int(cy), int(r)
        cv2.circle(display, (cx_i, cy_i), r_i, (255, 200, 0), 1)

    # --- 画聚类组 ---
    for group in groups:
        gx, gy = int(group["center"][0]), int(group["center"][1])
        count = len(group["points"])
        if count >= 2:
            cv2.circle(display, (gx, gy), 5, (255, 0, 0), -1)

    # --- 画霍夫圆 ---
    if hough_circles is not None:
        hcs = np.round(hough_circles[0]).astype(int)
        for cx, cy, r in hcs:
            cv2.circle(display, (cx, cy), r, (0, 180, 0), 1)

    # --- 画目标圆心（激光要打到的位置）---
    if target_cx is not None:
        color = (0, 255, 0)  # 绿色 = 目标
        cv2.circle(display, (target_cx, target_cy), target_r, color, 3)
        cv2.circle(display, (target_cx, target_cy), 8, color, 2)
        cv2.circle(display, (target_cx, target_cy), 3, (0, 0, 255), -1)

        label = f"TARGET[{target_source}]"
        cv2.putText(display, label, (target_cx + target_r + 8, target_cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # --- 画激光光斑（当前激光落点）---
    if laser_cx is not None:
        cv2.circle(display, (laser_cx, laser_cy), laser_r + 2, (0, 140, 255), 2)  # 橙色圈
        cv2.circle(display, (laser_cx, laser_cy), 5, (0, 165, 255), -1)  # 橙色实心
        cv2.putText(display, "LASER",
                    (laser_cx + laser_r + 5, laser_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

        # 激光 → 圆心偏移向量（核心：这就是PID要消除的误差）
        if target_cx is not None:
            offset = np.hypot(laser_cx - target_cx, laser_cy - target_cy)
            # 粗箭头：激光 → 圆心
            cv2.arrowedLine(display, (laser_cx, laser_cy),
                            (target_cx, target_cy), (0, 255, 255), 2, tipLength=0.15)
            # 偏移量文字
            mid_x = (laser_cx + target_cx) // 2
            mid_y = (laser_cy + target_cy) // 2
            cv2.putText(display, f"{offset:.1f}px",
                        (mid_x + 10, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- 信息栏 ---
    cv2.putText(display, f"Servo X:{pid_X_P} Y:{pid_Y_P}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

    if target_cx is not None and laser_cx is not None:
        err_x = target_cx - laser_cx
        err_y = target_cy - laser_cy
        aligned = abs(err_x) < DEAD_ZONE and abs(err_y) < DEAD_ZONE
        status = "ALIGNED!" if aligned else "MOVING->"
        status_color = (0, 255, 0) if aligned else (0, 200, 255)
        cv2.putText(display, f"Status: {status}  dXY({err_x:+d},{err_y:+d})",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    elif target_cx is None:
        cv2.putText(display, "Status: NO TARGET",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    else:
        cv2.putText(display, "Status: NO LASER",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return display


# ================= 主循环 =================

if __name__ == '__main__':
    setup()
    laser_on()
    time.sleep(0.5)

    print("[System] 激光对准圆心程序启动")
    print("[System] 按 ESC 退出, 按 'l' 切换激光, 按 'r' 复位舵机")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] 无法读取画面")
                break

            # 预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 1. 检测激光光斑
            laser_result, laser_mask = detect_laser_dot(hsv)
            laser_cx = laser_cy = laser_r = None
            if laser_result is not None:
                laser_cx, laser_cy, laser_r = laser_result

            # 2. 轮廓法检测圆形
            edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            circles_contour, groups = detect_circles_contour(edges)

            # 3. 霍夫圆检测
            hough_circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT,
                dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R
            )

            # 4. 选择跟踪目标
            target_cx, target_cy, target_r, target_source = select_target(
                circles_contour, groups, hough_circles
            )

            # 5. PID 控制：驱动舵机使激光光斑移动到圆心
            if target_cx is not None and laser_cx is not None:
                pid_control(target_cx, target_cy, laser_cx, laser_cy)

            # 6. 驱动舵机
            Robot_servo(pid_X_P, pid_Y_P)

            # 7. 可视化
            display = draw_overlay(
                frame, target_cx, target_cy, target_r, target_source,
                laser_cx, laser_cy, laser_r,
                circles_contour, groups, hough_circles
            )

            cv2.imshow("Laser Align Circle", display)
            cv2.imshow("Edges (Contour)", edges)       # 与 circle_see.py 一致的边缘图
            cv2.imshow("Laser Mask", laser_mask)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 退出
                break
            elif key == ord('l'):  # 切换激光
                if GPIO.input(LASER_PIN) == LASER_ON:
                    laser_off()
                else:
                    laser_on()
            elif key == ord('r'):  # 复位舵机
                pid_X_P = 300
                pid_Y_P = 280
                Robot_servo(pid_X_P, pid_Y_P)
                print("[Servo] 舵机已复位")

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        print("[System] 程序结束")

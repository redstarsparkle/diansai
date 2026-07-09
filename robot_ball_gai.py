#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    * @par Copyright (C): 2010-2020, Hunan CLB Tech
    * @file        robot_sevo_ball
    * @version      V2.0
    * @details      双阶段圆心追踪：
    *                 阶段1 — 霍夫圆粗定位，稳定后移动舵机
    *                 阶段2 — 轮廓拟合精定位，接管舵机控制
    * @par History
    @author: zhulin
"""
import cv2
import time
import numpy as np
import Adafruit_PCA9685

# ==================== 舵机初始化 ====================

servo_pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
servo_pwm.set_pwm_freq(60)
servo_pwm.set_pwm(5, 0, 350)
servo_pwm.set_pwm(4, 0, 370)
time.sleep(1)

# ==================== 摄像头初始化 ====================

usb_cap = cv2.VideoCapture(0)
FRAME_W = 320
FRAME_H = 240
usb_cap.set(3, FRAME_W)
usb_cap.set(4, FRAME_H)

CENTER_X = FRAME_W // 2   # 画面中心 X (160)
CENTER_Y = FRAME_H // 2   # 画面中心 Y (120)

# ==================== 霍夫圆参数（阶段1：粗定位） ====================

HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 80        # 调大到 80，减少误检
HOUGH_MIN_R = 10
HOUGH_MAX_R = 100

# 霍夫圆稳定判断
HOUGH_STABILITY_PX = 8    # 相邻帧圆心偏移 ≤ 8 像素视为稳定
HOUGH_STABLE_FRAMES = 5   # 连续稳定帧数达到后，认为锁定

# ==================== 轮廓法参数（阶段2：精定位） ====================

CANNY_LOW = 30
CANNY_HIGH = 120

MIN_CIRCULARITY = 0.35
MIN_AXIS_RATIO = 0.5
MIN_RADIUS = 6
MAX_AREA = 20000
V_MAX = 90

# 轮廓丢失容忍度：连续多少帧没检测到才退回阶段1
CONTOUR_LOST_MAX = 10

# ==================== PID 变量 ====================

pid_thisError_x = 0
pid_lastError_x = 0
pid_thisError_y = 0
pid_lastError_y = 0

pid_X_P = 300
pid_Y_P = 280

# ==================== 状态机 ====================

PHASE_SEARCH = "search"           # 阶段0：搜索中，霍夫圆尚未稳定
PHASE_HOUGH = "hough_track"       # 阶段1：霍夫圆锁定，粗跟踪
PHASE_CONTOUR = "contour_track"   # 阶段2：轮廓圆锁定，精跟踪

phase = PHASE_SEARCH
hough_stable_count = 0
hough_last_cx = hough_last_cy = None
contour_lost_count = 0


def Robot_servo(X_P, Y_P):
    servo_pwm.set_pwm(5, 0, 650 - X_P)
    servo_pwm.set_pwm(4, 0, 650 - Y_P)


def pid_control(cx, cy):
    """根据目标 (cx,cy) 计算并更新舵机角度"""
    global pid_thisError_x, pid_lastError_x
    global pid_thisError_y, pid_lastError_y
    global pid_X_P, pid_Y_P

    pid_thisError_x = cx - CENTER_X
    pid_thisError_y = cy - CENTER_Y

    pwm_x = pid_thisError_x * 3 + 1 * (pid_thisError_x - pid_lastError_x)
    pwm_y = pid_thisError_y * 3 + 1 * (pid_thisError_y - pid_lastError_y)

    pid_lastError_x = pid_thisError_x
    pid_lastError_y = pid_thisError_y

    pid_XP = pwm_x / 100
    pid_YP = pwm_y / 100

    pid_X_P = pid_X_P - int(pid_XP)
    pid_Y_P = pid_Y_P - int(pid_YP)

    if pid_X_P > 670:
        pid_X_P = 650
    if pid_X_P < 0:
        pid_X_P = 0
    if pid_Y_P > 650:
        pid_Y_P = 650
    if pid_Y_P < 0:
        pid_Y_P = 0


# ==================== 主循环 ====================

while True:

    ret, frame = usb_cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------
    # 霍夫圆检测（始终运行，用于阶段1）
    # ---------------------------------------------------
    hough_cx = hough_cy = hough_r = None
    hough_circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R
    )
    if hough_circles is not None:
        # 取投票最高的圆（OpenCV 默认已排序）
        hc = np.round(hough_circles[0, 0]).astype(int)
        hough_cx, hough_cy, hough_r = int(hc[0]), int(hc[1]), int(hc[2])

    # ---------------------------------------------------
    # 轮廓拟合椭圆检测（始终运行，用于阶段2）
    # ---------------------------------------------------
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    best_score = 0
    best_cx = best_cy = best_r = None
    best_v = 0

    for cnt in cnts:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area > MAX_AREA:
            continue
        try:
            (cx, cy), (w, h), _ = cv2.fitEllipse(cnt)
        except:
            continue
        r = min(w, h) / 2
        if r < MIN_RADIUS:
            continue
        axis_ratio = min(w, h) / max(w, h)
        if axis_ratio < MIN_AXIS_RATIO:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        # 颜色验证
        ellipse_pts = cv2.ellipse2Poly((int(cx), int(cy)), (int(w/2), int(h/2)), 0, 0, 360, 5)
        v_values = []
        for pt in ellipse_pts:
            x, y = pt[0], pt[1]
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                v_values.append(hsv[y, x, 2])
        if not v_values:
            continue
        v_mean = np.mean(v_values)
        if v_mean > V_MAX:
            continue
        cv2.ellipse(debug_frame, ((int(cx), int(cy)), (int(w), int(h)), 0), (0, 255, 0), 1)
        score = circularity * 0.7 + axis_ratio * 0.3
        if score > best_score:
            best_score = score
            best_cx, best_cy, best_r = int(cx), int(cy), int(r)
            best_v = v_mean

    contour_ok = (best_cx is not None)

    # ================================================
    # 状态机逻辑
    # ================================================

    if phase == PHASE_SEARCH:
        # --- 等待霍夫圆稳定 ---
        if hough_cx is not None:
            if hough_last_cx is not None and hough_last_cy is not None:
                dist = np.hypot(hough_cx - hough_last_cx, hough_cy - hough_last_cy)
                if dist <= HOUGH_STABILITY_PX:
                    hough_stable_count += 1
                else:
                    hough_stable_count = 0
            else:
                hough_stable_count = 1

            hough_last_cx = hough_cx
            hough_last_cy = hough_cy

        if hough_stable_count >= HOUGH_STABLE_FRAMES:
            phase = PHASE_HOUGH
            hough_stable_count = 0
            print("[STATE] Hough stable, switching to HOUGH_TRACK phase")

    elif phase == PHASE_HOUGH:
        # --- 霍夫圆粗跟踪 ---
        if contour_ok:
            # 轮廓也检测到了，切换到精跟踪
            phase = PHASE_CONTOUR
            contour_lost_count = 0
            print("[STATE] Contour detected, switching to CONTOUR_TRACK phase")
        elif hough_cx is not None:
            # 还在用霍夫圆跟踪
            pid_control(hough_cx, hough_cy)
        else:
            # 霍夫圆丢失，回退到搜索
            phase = PHASE_SEARCH
            hough_stable_count = 0
            hough_last_cx = hough_last_cy = None
            print("[STATE] Hough lost, back to SEARCH phase")

    elif phase == PHASE_CONTOUR:
        # --- 轮廓精跟踪 ---
        if contour_ok:
            contour_lost_count = 0
            pid_control(best_cx, best_cy)
        else:
            contour_lost_count += 1
            if contour_lost_count >= CONTOUR_LOST_MAX:
                # 轮廓丢失太久，退回阶段1（霍夫）
                phase = PHASE_HOUGH
                contour_lost_count = 0
                print("[STATE] Contour lost too long, fallback to HOUGH_TRACK phase")

    # ================================================
    # 画面绘制
    # ================================================

    # 画霍夫圆（黄色虚线圆）
    if hough_cx is not None:
        cv2.circle(frame, (hough_cx, hough_cy), hough_r, (0, 255, 255), 1)
        cv2.circle(frame, (hough_cx, hough_cy), 3, (0, 200, 255), -1)
        cv2.putText(frame, "Hough", (hough_cx + hough_r + 3, hough_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # 画轮廓拟合圆（品红色实线圆），只有阶段2或者轮廓OK才画
    if contour_ok:
        cv2.circle(frame, (best_cx, best_cy), best_r, (255, 0, 255), 2)
        cv2.circle(frame, (best_cx, best_cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, "Contour", (best_cx + best_r + 3, best_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

    # 画画面中心十字
    cv2.line(frame, (CENTER_X - 10, CENTER_Y), (CENTER_X + 10, CENTER_Y), (0, 255, 0), 1)
    cv2.line(frame, (CENTER_X, CENTER_Y - 10), (CENTER_X, CENTER_Y + 10), (0, 255, 0), 1)

    # 状态栏
    phase_colors = {PHASE_SEARCH: (0, 0, 255), PHASE_HOUGH: (0, 255, 255), PHASE_CONTOUR: (0, 255, 0)}
    cv2.putText(frame, f"Phase: {phase}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_colors.get(phase, (255, 255, 255)), 1)

    if phase == PHASE_SEARCH:
        cv2.putText(frame, f"Hough stable: {hough_stable_count}/{HOUGH_STABLE_FRAMES}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    elif phase == PHASE_CONTOUR:
        cv2.putText(frame, f"Lost: {contour_lost_count}/{CONTOUR_LOST_MAX}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 舵机位置显示
    cv2.putText(frame, f"Servo X:{pid_X_P} Y:{pid_Y_P}", (5, FRAME_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 调用舵机
    Robot_servo(pid_X_P, pid_Y_P)

    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Debug Candidates", debug_frame)
    cv2.imshow("MAKEROBO Robot", frame)

    if cv2.waitKey(1) == 119:  # 按 'w' 退出
        break

usb_cap.release()
cv2.destroyAllWindows()

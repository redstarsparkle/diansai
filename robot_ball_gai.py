#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    * @par Copyright (C): 2010-2020, Hunan CLB Tech
    * @file        robot_sevo_ball
    * @version      V1.0
    * @details
    * @par History
    @author: zhulin
"""
import cv2
import time
import numpy as np
import Adafruit_PCA9685

#初始化PCA9685和舵机
servo_pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)  # 实例话舵机云台

# 设置舵机初始值，可以根据自己的要求调试
servo_pwm.set_pwm_freq(60)  # 设置频率为60HZ
servo_pwm.set_pwm(5,0,350)  # 底座舵机
servo_pwm.set_pwm(4,0,370)  # 倾斜舵机
time.sleep(1)

#初始化摄像头并设置阙值
usb_cap = cv2.VideoCapture(0)

# ===== Canny 边缘检测参数（适配空心黑圆） =====
CANNY_LOW = 30
CANNY_HIGH = 120

# 圆形度过滤参数
MIN_CIRCULARITY = 0.35   # 圆形度阈值 (1.0=正圆)
MIN_AXIS_RATIO = 0.5     # 椭圆长短轴比下限
MIN_RADIUS = 6           # 最小半径 (像素)
MAX_AREA = 20000         # 最大轮廓面积
V_MAX = 90               # 边缘像素最大亮度 (黑色验证)

# 设置显示的分辨率，设置为320×240 px
usb_cap.set(3, 320)
usb_cap.set(4, 240)

#舵机云台的每个自由度需要4个变量
pid_thisError_x=500       #当前误差值
pid_lastError_x=100       #上一次误差值
pid_thisError_y=500
pid_lastError_y=100

pid_x=0
pid_y=0

# 舵机的转动角度
pid_Y_P = 280
pid_X_P = 300           #转动角度
pid_flag=0


# 机器人舵机旋转
def Robot_servo(X_P, Y_P):
    servo_pwm.set_pwm(5, 0, 650 - X_P)
    servo_pwm.set_pwm(4, 0, 650 - Y_P)

# 循环函数
while True:    
    ret,frame = usb_cap.read()

    #高斯模糊处理
    frame=cv2.GaussianBlur(frame,(5,5),0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Canny 边缘检测 → 空心圆也能抓（只靠边缘，不靠填充）
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选最圆的黑色轮廓
    debug_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    total = 0
    pass_fit = 0
    pass_circ = 0
    best_score = 0
    best_cx = best_cy = best_r = None
    best_v = 0
    for cnt in cnts:
        total += 1
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area > MAX_AREA:
            continue
        try:
            (cx, cy), (w, h), _ = cv2.fitEllipse(cnt)
        except:
            continue
        pass_fit += 1
        r = min(w, h) / 2
        if r < MIN_RADIUS:
            cv2.ellipse(debug_frame, ((int(cx), int(cy)), (int(w), int(h)), 0), (0, 0, 255), 1)
            continue
        axis_ratio = min(w, h) / max(w, h)
        if axis_ratio < MIN_AXIS_RATIO:
            cv2.ellipse(debug_frame, ((int(cx), int(cy)), (int(w), int(h)), 0), (0, 0, 255), 1)
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            cv2.ellipse(debug_frame, ((int(cx), int(cy)), (int(w), int(h)), 0), (255, 0, 0), 1)
            cv2.putText(debug_frame, f"c={circularity:.2f}", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            continue
        pass_circ += 1
        # 验证颜色
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
            cv2.ellipse(debug_frame, ((int(cx), int(cy)), (int(w), int(h)), 0), (255, 255, 0), 1)
            cv2.putText(debug_frame, f"V={v_mean:.0f}", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            continue
        # 通过所有检查：绿色
        cv2.ellipse(debug_frame, ((int(cx), int(cy)), (int(w), int(h)), 0), (0, 255, 0), 2)
        score = circularity * 0.7 + axis_ratio * 0.3
        if score > best_score:
            best_score = score
            best_cx, best_cy, best_r = int(cx), int(cy), int(r)
            best_v = v_mean

    # 调试信息
    cv2.putText(debug_frame, f"cnt:{total} fit:{pass_fit} circ:{pass_circ} best:{best_score:.2f}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(debug_frame, "RED=small/flat BLUE=lowC YEL=bright GRN=PASS", (5, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    if best_cx is not None:
        pid_x, pid_y, radius = best_cx, best_cy, best_r
        cv2.circle(frame, (pid_x, pid_y), radius, (255, 0, 255), 2)
        cv2.circle(frame, (pid_x, pid_y), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"S:{best_score:.2f} V:{best_v:.0f}", (pid_x - 30, pid_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 误差值处理
        pid_thisError_x = pid_x - 160
        pid_thisError_y = pid_y - 120

        #PID控制参数
        pwm_x = pid_thisError_x * 3 + 1 * (pid_thisError_x - pid_lastError_x)
        pwm_y = pid_thisError_y * 3 + 1 * (pid_thisError_y - pid_lastError_y)

        #迭代误差值操作
        pid_lastError_x = pid_thisError_x
        pid_lastError_y = pid_thisError_y

        pid_XP = pwm_x / 100
        pid_YP = pwm_y / 100

        # pid_X_P pid_Y_P 为最终PID值
        pid_X_P = pid_X_P - int(pid_XP)
        pid_Y_P = pid_Y_P - int(pid_YP)

        #限值舵机在一定的范围之内
        if pid_X_P > 670:
            pid_X_P = 650
        if pid_X_P < 0:
            pid_X_P = 0
        if pid_Y_P > 650:
            pid_Y_P = 650
        if pid_Y_P < 0:
            pid_Y_P = 0

    Robot_servo(pid_X_P, pid_Y_P)  # 直接控制舵机

    # 显示边缘图（调试用，确认 Canny 是否能抓到空心圆边缘）
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Debug Candidates", debug_frame)
    cv2.imshow("MAKEROBO Robot", frame)  # 显示图像
    if cv2.waitKey(1)==119:
        break
    
usb_cap.release()
cv2.destroyAllWindows()

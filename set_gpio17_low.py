#!/usr/bin/env python3
"""
* @par Copyright (C): 2010-2020, hunan CLB Tech
* @file         set_gpio19_low
* @version      V1.0
* @details      控制树莓派4B GPIO19 (BCM编号) 输出低电平
* @par History
*
* @author:
"""

import RPi.GPIO as GPIO
import time

GPIO17 = 17  # BCM 编号 17


def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)           # 使用 BCM 编号方式
    GPIO.setup(GPIO17, GPIO.OUT)     # 设置 GPIO17 为输出模式
    GPIO.output(GPIO17, GPIO.LOW)    # 设置 GPIO17 输出低电平
    print("GPIO17 已设置为低电平 (LOW)")


def cleanup():
    GPIO.cleanup()                   # 清理 GPIO 资源
    print("GPIO 资源已清理")


if __name__ == '__main__':
    setup()
    try:
        # 保持低电平输出，直到按下 Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

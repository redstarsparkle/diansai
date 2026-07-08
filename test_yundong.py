#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

PWMA = 18
AIN1 = 22
AIN2 = 27

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

L_Motor = GPIO.PWM(PWMA,100)
L_Motor.start(50)

GPIO.output(AIN1,1)
GPIO.output(AIN2,0)

try:
    while 1:
        time.sleep(1)
except KeyboardInterrupt:
    L_Motor.stop()
    GPIO.cleanup()
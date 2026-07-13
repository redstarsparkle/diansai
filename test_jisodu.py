import Adafruit_PCA9685
import time

pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
pwm.set_pwm_freq(60)

# 从 0° 到 180°，每隔 30° 测一次
for angle in [0, 30, 60, 90, 120, 150, 180]:
    # 反过来算：你想知道某个角度需要多少PWM，从这里读出
    # 这里先用 150~600 线性估算
    pwm_val = int(150 + angle * 2.5)  # 150→0°, 600→180°（估算）
    pwm.set_pwm(5, 0, pwm_val)
    print(f"角度约 {angle}° ← PWM={pwm_val}")
    time.sleep(2)
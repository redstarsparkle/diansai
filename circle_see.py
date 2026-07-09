import cv2
import numpy as np

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

MIN_AREA = 50
MAX_AREA = 30000

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 白纸黑圆
    _, binary = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        if len(cnt) < 5:
            continue

        # 最小外接圆
        (x,y), r = cv2.minEnclosingCircle(cnt)

        # 圆度
        peri = cv2.arcLength(cnt, True)

        if peri == 0:
            continue

        circularity = 4*np.pi*area/(peri*peri)

        if circularity < 0.75:
            continue

        x = int(x)
        y = int(y)
        r = int(r)

        cv2.circle(frame,(x,y),r,(0,255,0),2)
        cv2.circle(frame,(x,y),3,(0,0,255),-1)

        cv2.putText(
            frame,
            f"({x},{y})",
            (x+10,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            1
        )

    cv2.imshow("binary",binary)
    cv2.imshow("result",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
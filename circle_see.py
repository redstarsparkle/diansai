import cv2
import numpy as np

# ================= 参数 =================

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

CANNY_LOW = 50
CANNY_HIGH = 150

MIN_AREA = 20
MAX_AREA = 30000

MIN_RADIUS = 10
MIN_AXIS_RATIO = 0.75

# 圆心聚类距离（像素）
CENTER_DISTANCE = 5

# =======================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    circles = []

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

        (cx, cy), (w, h), angle = cv2.fitEllipse(cnt)

        ratio = min(w, h) / max(w, h)

        if ratio < MIN_AXIS_RATIO:
            continue

        radius = (w + h) / 4

        if radius < MIN_RADIUS:
            continue

        circles.append((cx, cy, radius))

    # =============================
    # 圆心聚类
    # =============================

    groups = []

    for cx, cy, r in circles:

        found = False

        for group in groups:

            gx, gy = group["center"]

            d = np.hypot(cx - gx, cy - gy)

            if d < CENTER_DISTANCE:

                group["points"].append((cx, cy, r))

                n = len(group["points"])

                group["center"] = (
                    (gx * (n - 1) + cx) / n,
                    (gy * (n - 1) + cy) / n
                )

                found = True

                break

        if not found:

            groups.append({
                "center": (cx, cy),
                "points": [(cx, cy, r)]
            })

    # =============================
    # 显示结果
    # =============================

    # 先画每个独立检测到的圆
    for i, (cx, cy, radius) in enumerate(circles):
        cx_i = int(cx)
        cy_i = int(cy)
        r_i = int(radius)

        # 画圆轮廓（青色）
        cv2.circle(frame, (cx_i, cy_i), r_i, (255, 255, 0), 2)
        # 画圆心（红点）
        cv2.circle(frame, (cx_i, cy_i), 3, (0, 0, 255), -1)
        # 标序号
        cv2.putText(
            frame,
            f"#{i+1}",
            (cx_i + r_i + 5, cy_i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1
        )

    # 再画聚类结果
    for group in groups:
        cx, cy = group["center"]
        cx = int(cx)
        cy = int(cy)
        count = len(group["points"])

        # 只有两个以上圆才认为是目标
        if count < 2:
            continue

        cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)

        cv2.putText(
            frame,
            f"Center ({cx},{cy})",
            (cx + 15, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Group: {count} circles",
            (cx + 15, cy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # 左上角显示统计
    cv2.putText(
        frame,
        f"Detected: {len(circles)} | Groups: {len(groups)}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )

    cv2.imshow("Edges", edges)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
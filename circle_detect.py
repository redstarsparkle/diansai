import cv2
import numpy as np

# ================= 通用参数 =================

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CENTER_X = FRAME_WIDTH // 2   # 160
CENTER_Y = FRAME_HEIGHT // 2  # 120

# ================= 方法1：轮廓法参数 =================

CANNY_LOW = 50
CANNY_HIGH = 150

MIN_AREA = 20
MAX_AREA = 30000

MIN_RADIUS_CONTOUR = 10
MIN_AXIS_RATIO = 0.75

# 圆心聚类距离（像素）
CENTER_DISTANCE = 5

# ================= 方法2：霍夫圆参数 =================
#  调参口诀：
#  - 误检太多 → 调大 param2（比如 40→60→80）
#  - 漏检太多 → 调小 param2（比如 50→30→20）
#  - 边缘噪声多 → 调大 param1（比如 50→80→120）
#  - 重复检出 → 调大 minDist（比如 20→40）

HOUGH_DP = 1.2          # 累加器分辨率（越小越敏感）
HOUGH_MIN_DIST = 30     # 圆心最小间距
HOUGH_PARAM1 = 80       # Canny 高阈值（低阈值自动=param1/2）
HOUGH_PARAM2 = 80       # 累加器阈值：越大越严格，杂圆越少
HOUGH_MIN_R = 10        # 最小半径
HOUGH_MAX_R = 100       # 最大半径

# =================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# =================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ============================
    # 方法1：Canny + 轮廓椭圆拟合（左图）
    # ============================

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

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

    # 圆心聚类
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

    # --- 左图绘制 ---
    left = frame.copy()

    for i, (cx, cy, radius) in enumerate(circles_contour):
        cx_i, cy_i, r_i = int(cx), int(cy), int(radius)
        cv2.circle(left, (cx_i, cy_i), r_i, (255, 255, 0), 2)
        cv2.circle(left, (cx_i, cy_i), 3, (0, 0, 255), -1)
        cv2.putText(left, f"#{i+1}", (cx_i + r_i + 5, cy_i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    for group in groups:
        cx, cy = group["center"]
        cx, cy = int(cx), int(cy)
        count = len(group["points"])
        if count < 2:
            continue
        cv2.circle(left, (cx, cy), 7, (255, 0, 0), -1)
        cv2.putText(left, f"({cx},{cy})", (cx + 15, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.putText(left, f"G:{count}", (cx + 15, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    cv2.putText(left, f"[Contour] Detected:{len(circles_contour)} Groups:{len(groups)}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # --- 选定跟踪目标（轮廓优先）---
    track_cx = track_cy = None
    best_group = None
    for g in groups:
        if len(g["points"]) >= 2:
            if best_group is None or len(g["points"]) > len(best_group["points"]):
                best_group = g
    if best_group is not None:
        track_cx, track_cy = int(best_group["center"][0]), int(best_group["center"][1])
    elif len(circles_contour) > 0:
        track_cx, track_cy = int(circles_contour[0][0]), int(circles_contour[0][1])

    # ============================
    # 方法2：霍夫圆检测（右图）
    # ============================

    right = frame.copy()

    hough_circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R
    )

    if hough_circles is not None:
        hough_circles = np.round(hough_circles[0]).astype(int)

        for i, (cx, cy, r) in enumerate(hough_circles):
            # 绿色圆 + 红色圆心
            cv2.circle(right, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(right, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(right, f"#{i+1}({cx},{cy})", (cx + r + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # 轮廓没有目标时，用霍夫圆兜底
        if track_cx is None and len(hough_circles) > 0:
            track_cx, track_cy = int(hough_circles[0][0]), int(hough_circles[0][1])

        cv2.putText(right, f"[Hough] Detected: {len(hough_circles)}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        cv2.putText(right, "[Hough] No circles found",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # ============================
    # 误差显示
    # ============================

    if track_cx is not None:
        error_x = track_cx - CENTER_X
        error_y = track_cy - CENTER_Y
        # 左图显示误差
        cv2.putText(left, f"Err X:{error_x:+d} Y:{error_y:+d}",
                    (5, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        # 右图也显示误差
        cv2.putText(right, f"Err X:{error_x:+d} Y:{error_y:+d}",
                    (FRAME_WIDTH - 130, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        # 画瞄准线（目标到中心）
        cv2.line(left, (track_cx, track_cy), (CENTER_X, CENTER_Y), (0, 0, 255), 1)
        cv2.line(right, (track_cx, track_cy), (CENTER_X, CENTER_Y), (0, 0, 255), 1)

    # ============================
    # 画中心十字（两图都画）
    # ============================

    for img in (left, right):
        cv2.line(img, (CENTER_X - 10, CENTER_Y), (CENTER_X + 10, CENTER_Y), (0, 255, 0), 1)
        cv2.line(img, (CENTER_X, CENTER_Y - 10), (CENTER_X, CENTER_Y + 10), (0, 255, 0), 1)

    # ============================
    # 并排显示
    # ============================

    comparison = np.hstack((left, right))

    # 中间分隔线
    h, w = comparison.shape[:2]
    cv2.line(comparison, (FRAME_WIDTH, 0), (FRAME_WIDTH, h), (255, 255, 255), 1)

    # 标题
    cv2.putText(comparison, "Contour", (5, FRAME_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(comparison, "HoughCircles", (FRAME_WIDTH + 5, FRAME_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Edges (Contour)", edges)
    cv2.imshow("Contour  vs  HoughCircles", comparison)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

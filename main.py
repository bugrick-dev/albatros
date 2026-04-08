import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

H_MIN, H_MAX = 100, 140
S_MIN, S_MAX = 90, 255
V_MIN, V_MAX = 50, 255
kernel = np.ones((5, 5), np.uint8)

def detect_square(cnt):
    area = cv2.contourArea(cnt)
    if area < 500:
        return None

    rect = cv2.minAreaRect(cnt)
    (center), (width, height), angle = rect

    if max(width, height) == 0:
        return None

    aspect_ratio = min(width, height) / max(width, height)
    if not (0.8 < aspect_ratio < 1.2):
        return None

    if area > 10000:
        return "blue_target", area, center
    elif area > 2500:
        return "red_target", area, center

    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, np.array([H_MIN, S_MIN, V_MIN]), np.array([H_MAX, S_MAX, V_MAX]))
    mask_red1 = cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([10, S_MAX, V_MAX]))
    mask_red2 = cv2.inRange(hsv, np.array([170, S_MIN, V_MIN]), np.array([180, S_MAX, V_MAX]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_grey = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 40, 200]))

    mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    mask_red = cv2.erode(mask_red, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask_grey)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=120, maxLineGap=15)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 15 or angle > 165:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, "pole", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

    for color, mask in [("blue_target", mask_blue), ("red_target", mask_red)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            result = detect_square(cnt)
            if not result:
                continue

            shape_type, area, center = result
            cx, cy = int(center[0]), int(center[1])
            color_rgb = (255, 0, 0) if "blue" in shape_type else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 5, color_rgb, -1)
            cv2.drawContours(frame, [cnt], -1, color_rgb, 2)
            cv2.putText(frame, shape_type, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)

    cv2.imshow("kamera", frame)
    cv2.imshow("maskeblue", mask_blue)
    cv2.imshow("maskered", mask_red)
    cv2.imshow("maskepole", edges)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
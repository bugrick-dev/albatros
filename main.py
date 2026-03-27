from picamera2 import Picamera2
import cv2
import numpy as np
import subprocess
import time
import os

# --- Ayarlar ---
DEST_IP = "192.168.50.23"
DEST_PORT = 5600
WIDTH = 640
HEIGHT = 480
FPS = 30
SNAPSHOT_DIR = "/home/albatros/albatros/snapshots"
SNAPSHOT_INTERVAL = 1  # saniye

# HSV Değerleri
H_MIN, H_MAX = 100, 140
S_MIN, S_MAX = 90, 255
V_MIN, V_MAX = 50, 255

# --- Klasör oluştur ---
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --- FFmpeg pipeline ---
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{WIDTH}x{HEIGHT}',
    '-r', str(FPS),
    '-i', '-',
    '-c:v', 'h264_v4l2m2m',
    '-b:v', '2000k',
    '-pkt_size', '1316',
    '-f', 'mpegts',
    f'udp://{DEST_IP}:{DEST_PORT}?pkt_size=1316'
]

process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# --- Kamera ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

kernel = np.ones((5, 5), np.uint8)
last_snapshot = time.time()

while True:
    frame = picam2.capture_array()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi maske
    mask_blue = cv2.inRange(hsv, np.array([H_MIN, S_MIN, V_MIN]), np.array([H_MAX, S_MAX, V_MAX]))
    mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    # Kırmızı maske
    mask_red1 = cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([10, S_MAX, V_MAX]))
    mask_red1 = cv2.erode(mask_red1, kernel, iterations=2)
    mask_red2 = cv2.inRange(hsv, np.array([170, S_MIN, V_MIN]), np.array([180, S_MAX, V_MAX]))
    mask_red2 = cv2.erode(mask_red2, kernel, iterations=2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    # Toplam maske
    mask = cv2.bitwise_or(mask_blue, mask_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Kontur tespiti
    for color, mask in [("mavi", mask_blue), ("kirmizi", mask_red)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
                corners = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"X: {cx} Y: {cy}", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
                if corners == 3:
                    isim = "ucgen"
                elif corners == 4:
                    isim = "kare" if 0.90 < aspectRatio < 1.10 else "dikdortgen"
                elif corners == 6:
                    isim = "hexagon"
                else:
                    isim = "circle"
                cv2.putText(frame, isim, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    # UDP stream
    try:
        process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        break

    # Saniye başı snapshot
    now = time.time()
    if now - last_snapshot >= SNAPSHOT_INTERVAL:
        small = cv2.resize(frame, (320, 240))
        filename = os.path.join(SNAPSHOT_DIR, f"{int(now)}.jpg")
        ret = cv2.imwrite(filename, small)
        print(f"Snapshot: {filename} - {'OK' if ret else 'FAILED'}")
        last_snapshot = now

        files = sorted(os.listdir(SNAPSHOT_DIR))
        while len(files) > 100:
            os.remove(os.path.join(SNAPSHOT_DIR, files.pop(0)))

picam2.stop()
process.stdin.close()

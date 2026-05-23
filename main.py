#!/usr/bin/env python3
"""
TEKNOFEST 2026 - Sabit Kanat - Görev 2
rpicam-vid → OpenCV (process) → GStreamer (encode) → WFB-ng

Pipeline:
1. rpicam-vid: Kamera H264 stream (TCP 8888)
2. FFmpeg decode: H264 → raw BGR frames
3. OpenCV: Detection + overlay (boxes, text)
4. FFmpeg encode: BGR → H264
5. GStreamer: H264 → RTP → UDP 5600
6. WFB-ng: UDP 5600 → RF broadcast
"""

import math
import asyncio
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
import subprocess
import sys

# ==================== KONFİGÜRASYON ====================

WIDTH = 640  # Düşük çözünürlük = hızlı processing
HEIGHT = 480
FPS = 30

MIN_AREA = int(0.001 * WIDTH * HEIGHT)
MAX_AREA = int(0.3 * WIDTH * HEIGHT)

# HSV
BLUE_H_MIN, BLUE_H_MAX = 100, 130
BLUE_S_MIN, BLUE_S_MAX = 80, 255
BLUE_V_MIN, BLUE_V_MAX = 50, 255

RED1_H_MIN, RED1_H_MAX = 0, 10
RED2_H_MIN, RED2_H_MAX = 170, 180
RED_S_MIN, RED_S_MAX = 80, 255
RED_V_MIN, RED_V_MAX = 50, 255

SQUARE_CORNER_TOLERANCE = 4
ASPECT_RATIO_MIN = 0.7
ASPECT_RATIO_MAX = 1.3

# WFB-ng
WFB_MAC = "6c:4c:bc:0a:62:a0"
WFB_KEY_PATH = "/home/albatros/gs.key"
WFB_LINK_ID = "7669206"
WFB_CHANNEL = 36

RPICAM_TCP_PORT = 8888
FFMPEG_UDP_PORT = 9000  # FFmpeg encode → bu port
WFB_UDP_PORT = 5600     # GStreamer → bu port → wfb_tx

FC_PORT = "/dev/ttyAMA3"
FC_BAUDRATE = 115200

CAMERA_FOV_H = 62.2
CAMERA_FOV_V = 48.8
CAMERA_PITCH = 60.0

# ==================== GLOBAL ====================

kernel = np.ones((5, 5), np.uint8)
target_queue = Queue()
telemetry_lock = threading.Lock()
current_telemetry = {"lat": None, "lon": None, "alt": None, "yaw": None}
detected_targets = {"mavi": None, "kirmizi": None}

wfb_process = None
rpicam_process = None
gst_process = None
ffmpeg_decode_process = None
ffmpeg_encode_process = None

# ==================== MONITOR MODE ====================

def setup_monitor_mode():
    """WiFi monitor mode"""
    print("[WFB-NG] WiFi arayüzü aranıyor...")

    try:
        result = subprocess.run(["ip", "link"], capture_output=True, text=True)
        iface = None
        lines = result.stdout.splitlines()
        
        for i, line in enumerate(lines):
            if WFB_MAC.lower() in line.lower():
                prev = lines[i - 1] if i > 0 else ""
                parts = prev.split(": ")
                if len(parts) >= 2:
                    iface = parts[1].split("@")[0].strip()
                break

        if not iface:
            print(f"[ERROR] MAC {WFB_MAC} bulunamadı!")
            return None

        print(f"[WFB-NG] ✓ Arayüz: {iface}")

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

    subprocess.run(["sudo", "rfkill", "unblock", "all"], check=False)
    subprocess.run(["sudo", "systemctl", "stop", "NetworkManager"], check=False)
    time.sleep(1)

    cmds = [
        ["sudo", "ip", "link", "set", iface, "down"],
        ["sudo", "iw", "dev", iface, "set", "monitor", "none"],
        ["sudo", "ip", "link", "set", iface, "up"],
        ["sudo", "iw", "dev", iface, "set", "channel", str(WFB_CHANNEL)],
        ["sudo", "iw", "dev", iface, "set", "txpower", "fixed", "2000"],
    ]

    for cmd in cmds:
        subprocess.run(cmd, check=False)

    print(f"[WFB-NG] ✓ Monitor mode: {iface}")
    return iface


# ==================== VIDEO PIPELINE ====================

def start_pipeline(iface):
    """
    rpicam-vid → (TCP) → opencv_thread → (UDP) → GStreamer → WFB-ng
    """
    global wfb_process, rpicam_process, gst_process

    print("\n[PIPELINE] Başlatılıyor...")

    # 1) wfb_tx
    wfb_cmd = f"sudo wfb_tx -K {WFB_KEY_PATH} -i {WFB_LINK_ID} -p 0 -u {WFB_UDP_PORT} {iface}"
    wfb_process = subprocess.Popen(wfb_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[WFB-TX] ✓ PID {wfb_process.pid}")
    time.sleep(2)

    # 2) rpicam-vid
    rpicam_cmd = f"""
    rpicam-vid -t 0 --inline --codec h264 \
      --width {WIDTH} --height {HEIGHT} --framerate {FPS} \
      --bitrate 3000000 --intra 30 \
      --listen -o tcp://127.0.0.1:{RPICAM_TCP_PORT}
    """
    rpicam_process = subprocess.Popen(rpicam_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[RPICAM] ✓ PID {rpicam_process.pid}, TCP {RPICAM_TCP_PORT}")
    time.sleep(5)

    # 3) GStreamer (FFmpeg'den UDP alır, wfb_tx'e gönderir)
    gst_cmd = f"""
    gst-launch-1.0 udpsrc port={FFMPEG_UDP_PORT} ! \
      application/x-rtp,encoding-name=H264,payload=96 ! \
      rtph264depay ! h264parse ! rtph264pay config-interval=1 pt=96 ! \
      udpsink host=127.0.0.1 port={WFB_UDP_PORT}
    """
    gst_process = subprocess.Popen(gst_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[GSTREAMER] ✓ PID {gst_process.pid}")

    print("[PIPELINE] ✓ Hazır\n")
    return True


def stop_pipeline():
    global wfb_process, rpicam_process, gst_process, ffmpeg_decode_process, ffmpeg_encode_process

    for proc, name in [(gst_process, "GStreamer"), (ffmpeg_encode_process, "FFmpeg-enc"),
                       (ffmpeg_decode_process, "FFmpeg-dec"), 
                       (rpicam_process, "rpicam"), (wfb_process, "wfb_tx")]:
        if proc:
            proc.terminate()
            print(f"[STOP] {name}")


# ==================== DETECTION + OVERLAY ====================

def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, target_cx, target_cy):
    camera_pitch_rad = math.radians(CAMERA_PITCH)
    fov_x_rad = math.radians(CAMERA_FOV_H)
    fov_y_rad = math.radians(CAMERA_FOV_V)

    center_x = WIDTH / 2
    center_y = HEIGHT / 2

    delta_x_pixel = target_cx - center_x
    delta_y_pixel = center_y - target_cy

    offset_angle_x = delta_x_pixel * (fov_x_rad / WIDTH)
    offset_angle_y = delta_y_pixel * (fov_y_rad / HEIGHT)

    total_pitch = camera_pitch_rad + offset_angle_y

    dist_y = alt * math.tan(total_pitch)
    dist_x = (alt / math.cos(total_pitch)) * math.tan(offset_angle_x)

    yaw_rad = math.radians(yaw_deg)
    delta_north = (dist_y * math.cos(yaw_rad)) - (dist_x * math.sin(yaw_rad))
    delta_east  = (dist_y * math.sin(yaw_rad)) + (dist_x * math.cos(yaw_rad))

    target_lat = drone_lat + (delta_north / 111320)
    target_lon = drone_lon + (delta_east / (111320 * math.cos(math.radians(drone_lat))))

    return target_lat, target_lon


def is_square(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) != SQUARE_CORNER_TOLERANCE:
        return False

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width <= 0 or height <= 0:
        return False

    aspect_ratio = max(width, height) / min(width, height)
    return ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX


def opencv_processing_thread(queue):
    """
    rpicam-vid TCP → decode → OpenCV process → encode → UDP → GStreamer
    """
    global ffmpeg_decode_process, ffmpeg_encode_process, detected_targets

    print("[OPENCV] Thread başlatıldı...")
    time.sleep(7)  # rpicam-vid'in TCP açması için bekle

    # FFmpeg DECODE (TCP → raw BGR)
    ffmpeg_decode_cmd = [
        "ffmpeg",
        "-i", f"tcp://127.0.0.1:{RPICAM_TCP_PORT}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-"
    ]

    ffmpeg_decode_process = subprocess.Popen(
        ffmpeg_decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    print("[OPENCV] FFmpeg decode başladı")

    # FFmpeg ENCODE (raw BGR → H264 → UDP) - HARDWARE ENCODER
    ffmpeg_encode_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",  # stdin
        "-c:v", "h264_v4l2m2m",  # Pi hardware encoder (HIZLI!)
        "-b:v", "2000k",
        "-g", "15",  # Keyframe her 0.5 saniye
        "-f", "h264",
        "-",  # stdout
    ]

    ffmpeg_encode_process = subprocess.Popen(
        ffmpeg_encode_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    print("[OPENCV] FFmpeg encode başladı (hardware)")

    # GStreamer'a pipe et (ayrı thread gerekli)
    def ffmpeg_to_gstreamer():
        gst_cmd = [
            "gst-launch-1.0",
            "fdsrc", "!",
            "h264parse", "!",
            "rtph264pay", "config-interval=1", "pt=96", "!",
            "udpsink", f"host=127.0.0.1", f"port={WFB_UDP_PORT}"
        ]
        gst_proc = subprocess.Popen(gst_cmd, stdin=ffmpeg_encode_process.stdout, stderr=subprocess.DEVNULL)
        gst_proc.wait()
    
    threading.Thread(target=ffmpeg_to_gstreamer, daemon=True).start()

    frame_size = WIDTH * HEIGHT * 3
    gps_data = {}  # {color: (lat, lon)}
    frame_count = 0
    fps_time = time.time()
    current_fps = 0

    while True:
        try:
            # Frame oku
            raw_frame = ffmpeg_decode_process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                time.sleep(0.01)
                continue

            # WRITABLE array oluştur (.copy() ile)
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
            
            # FPS hesapla
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # === DETECTION (SÜREKLI) ===
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # MAVİ KARE (Sürekli ara)
            mask_blue = cv2.inRange(hsv,
                np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
                np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX]))
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

            blue_found = False
            for cnt in cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA < area < MAX_AREA):
                    continue
                if not is_square(cnt):
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Her frame'de güncelle
                detected_targets["mavi"] = {"cx": cx, "cy": cy, "contour": cnt}
                blue_found = True
                
                # İlk tespit ise queue'ya ekle (mission için)
                if "mavi" not in gps_data:
                    queue.put({"color": "mavi", "cx": cx, "cy": cy})
                    print(f"✓ MAVİ KARE @ ({cx}, {cy})")
                break

            if not blue_found:
                detected_targets["mavi"] = None  # Kayboldu

            # KIRMIZI KARE (Sürekli ara)
            mask_red1 = cv2.inRange(hsv,
                np.array([RED1_H_MIN, RED_S_MIN, RED_V_MIN]),
                np.array([RED1_H_MAX, RED_S_MAX, RED_V_MAX]))
            mask_red2 = cv2.inRange(hsv,
                np.array([RED2_H_MIN, RED_S_MIN, RED_V_MIN]),
                np.array([RED2_H_MAX, RED_S_MAX, RED_V_MAX]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

            red_found = False
            for cnt in cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA < area < MAX_AREA):
                    continue
                if not is_square(cnt):
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                detected_targets["kirmizi"] = {"cx": cx, "cy": cy, "contour": cnt}
                red_found = True
                
                if "kirmizi" not in gps_data:
                    queue.put({"color": "kirmizi", "cx": cx, "cy": cy})
                    print(f"✓ KIRMIZI KARE @ ({cx}, {cy})")
                break

            if not red_found:
                detected_targets["kirmizi"] = None

            # === OVERLAY (Kutular + GPS) ===
            for color, data in detected_targets.items():
                if data:
                    # Kutu çiz
                    cv2.drawContours(frame, [data["contour"]], -1, 
                                   (255, 0, 0) if color == "mavi" else (0, 0, 255), 3)
                    
                    # GPS varsa yaz
                    if color in gps_data:
                        lat, lon = gps_data[color]
                        text = f"{color.upper()}: {lat:.6f}, {lon:.6f}"
                        cv2.putText(frame, text, (data["cx"] - 100, data["cy"] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Telemetri göster
            with telemetry_lock:
                tel = current_telemetry.copy()
            
            # FPS göster (sol üst)
            cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Altitude (sol üst, 2. satır)
            if tel["alt"]:
                cv2.putText(frame, f"ALT: {tel['alt']:.1f}m", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Detection status (sağ üst)
            status_y = 30
            if detected_targets["mavi"]:
                cv2.putText(frame, "MAVI: OK", (WIDTH - 150, status_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "MAVI: --", (WIDTH - 150, status_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            status_y += 30
            if detected_targets["kirmizi"]:
                cv2.putText(frame, "KIRMIZI: OK", (WIDTH - 150, status_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "KIRMIZI: --", (WIDTH - 150, status_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            # Frame gönder (encode)
            ffmpeg_encode_process.stdin.write(frame.tobytes())

            # GPS hesapla (hedef bulunduysa)
            if None not in tel.values():
                for color, data in detected_targets.items():
                    if data and color not in gps_data:
                        lat, lon = pixel_to_gps(tel["lat"], tel["lon"], tel["alt"], 
                                              tel["yaw"], data["cx"], data["cy"])
                        gps_data[color] = (lat, lon)

        except Exception as e:
            print(f"[OPENCV ERROR] {e}")
            time.sleep(0.1)


# ==================== MAVSDK ====================

async def telemetry_task(drone):
    async for position in drone.telemetry.position():
        with telemetry_lock:
            current_telemetry["lat"] = position.latitude_deg
            current_telemetry["lon"] = position.longitude_deg
            current_telemetry["alt"] = position.relative_altitude_m


async def attitude_task(drone):
    async for attitude in drone.telemetry.attitude_euler():
        with telemetry_lock:
            current_telemetry["yaw"] = attitude.yaw_deg


async def mission_task(drone, queue):
    waypoints = {}

    while True:
        try:
            target = queue.get(timeout=0.5)
        except Empty:
            await asyncio.sleep(0.1)
            continue

        with telemetry_lock:
            tel = current_telemetry.copy()

        if None in tel.values():
            await asyncio.sleep(1)
            continue

        color = target["color"]
        if color in waypoints:
            continue

        target_lat, target_lon = pixel_to_gps(
            tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
            target["cx"], target["cy"]
        )

        print(f"[MISSION] {color.upper()} → {target_lat:.7f}, {target_lon:.7f}")

        wp = MissionItem(
            latitude_deg=target_lat,
            longitude_deg=target_lon,
            relative_altitude_m=tel["alt"],
            speed_m_s=15.0,
            is_fly_through=True,
            gimbal_pitch_deg=float('nan'),
            gimbal_yaw_deg=float('nan'),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=float('nan'),
            camera_photo_interval_s=float('nan'),
            acceptance_radius_m=10.0,
            yaw_deg=float('nan'),
            camera_photo_distance_m=float('nan'),
            vehicle_action=MissionItem.VehicleAction.NONE
        )

        waypoints[color] = wp

        try:
            await drone.mission.upload_mission(MissionPlan(list(waypoints.values())))
            print(f"✓ Mission: {len(waypoints)} WP")
        except Exception as e:
            print(f"✗ Mission FAILED: {e}")
            del waypoints[color]

        await asyncio.sleep(1)


# ==================== MAIN ====================

async def run():
    print("=" * 60)
    print("TEKNOFEST 2026 - Sabit Kanat - Görev 2")
    print("rpicam → OpenCV (overlay) → GStreamer → WFB-ng")
    print("=" * 60)

    # 1) Monitor mode
    iface = setup_monitor_mode()
    if not iface:
        sys.exit(1)

    # 2) Pipeline
    if not start_pipeline(iface):
        sys.exit(1)

    # 3) OpenCV processing thread
    opencv_t = threading.Thread(target=opencv_processing_thread, args=(target_queue,), daemon=True)
    opencv_t.start()

    # 4) MAVSDK
    drone = System()
    print(f"\n[MAVSDK] {FC_PORT} bağlanıyor...")
    await drone.connect(system_address=f"serial://{FC_PORT}:{FC_BAUDRATE}")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAVSDK] ✓")
            break

    print("\n" + "=" * 60)
    print("✓ SİSTEM HAZIR")
    print("Yayında hedefler kutulu + GPS koordinatlı görünecek!")
    print("=" * 60 + "\n")

    try:
        await asyncio.gather(
            telemetry_task(drone),
            attitude_task(drone),
            mission_task(drone, target_queue),
        )
    except asyncio.CancelledError:
        pass
    finally:
        stop_pipeline()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n✓ Durduruldu")
        stop_pipeline()
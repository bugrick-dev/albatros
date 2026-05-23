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
import RPi.GPIO as GPIO

# ==================== KONFİGÜRASYON ====================

WIDTH = 640
HEIGHT = 480
FPS = 30

MIN_AREA = int(0.001 * WIDTH * HEIGHT)
MAX_AREA = int(0.3 * WIDTH * HEIGHT)

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
FFMPEG_UDP_PORT = 9000
WFB_UDP_PORT = 5600

FC_PORT = "/dev/ttyAMA3"
FC_BAUDRATE = 115200

CAMERA_FOV_H = 62.2
CAMERA_FOV_V = 48.8
CAMERA_PITCH = 60.0

# Şartname: MAVİ hedef (4x4m) → KIRMIZI boyalı yük, KIRMIZI hedef (2x2m) → MAVİ boyalı yük
SERVO_KIRMIZI_YUK_PIN = 18   # Mavi hedefe bırakılacak kırmızı boyalı yük
SERVO_MAVI_YUK_PIN    = 23   # Kırmızı hedefe bırakılacak mavi boyalı yük

DRONE_SPEED_MS = 15.0
DROP_TRIGGER_RADIUS_M = 20
SCAN_EXIT_DELAY_SEC = 15
DETECTION_ACTIVE_DELAY_SEC = 30   # Direk 2 geçildikten sonra detection başlamalı (saniye)
SINGLE_TARGET_TIMEOUT_SEC = 30    # Tek hedef bulununca ikincisi için max bekleme

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

servo_kirmizi_pwm = None   # kırmızı boyalı yük → mavi hedefe
servo_mavi_pwm    = None   # mavi boyalı yük    → kırmızı hedefe
detection_active  = threading.Event()  # Direk 2 geçildikten sonra set edilir

# ==================== SERVO ====================

def init_servo():
    global servo_kirmizi_pwm, servo_mavi_pwm
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(SERVO_KIRMIZI_YUK_PIN, GPIO.OUT)
    servo_kirmizi_pwm = GPIO.PWM(SERVO_KIRMIZI_YUK_PIN, 50)
    servo_kirmizi_pwm.start(7.5)

    GPIO.setup(SERVO_MAVI_YUK_PIN, GPIO.OUT)
    servo_mavi_pwm = GPIO.PWM(SERVO_MAVI_YUK_PIN, 50)
    servo_mavi_pwm.start(7.5)

    print(f"[SERVO] ✓ Kırmızı yük → GPIO {SERVO_KIRMIZI_YUK_PIN}")
    print(f"[SERVO] ✓ Mavi yük    → GPIO {SERVO_MAVI_YUK_PIN}")


def _drop_payload_sync(hedef_rengi):
    """Şartname: mavi hedefe kırmızı boyalı yük, kırmızı hedefe mavi boyalı yük."""
    global servo_kirmizi_pwm, servo_mavi_pwm

    if hedef_rengi == "mavi":
        pwm = servo_kirmizi_pwm
        yuk = "KIRMIZI"
    else:
        pwm = servo_mavi_pwm
        yuk = "MAVİ"

    if pwm:
        pwm.ChangeDutyCycle(12.5)
        time.sleep(0.5)          # sync sleep — sadece ayrı thread'den çağrılır
        pwm.ChangeDutyCycle(7.5)
        print(f"[SERVO] ✓ {yuk} boyalı yük bırakıldı ({hedef_rengi.upper()} hedefe)")


async def drop_payload(hedef_rengi):
    """Async wrapper — event loop'u bloklamaz."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _drop_payload_sync, hedef_rengi)


def cleanup_servo():
    global servo_kirmizi_pwm, servo_mavi_pwm
    if servo_kirmizi_pwm:
        servo_kirmizi_pwm.stop()
    if servo_mavi_pwm:
        servo_mavi_pwm.stop()
    GPIO.cleanup()


# ==================== MONITOR MODE ====================

def setup_monitor_mode():
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
    global wfb_process, rpicam_process, gst_process

    print("\n[PIPELINE] Başlatılıyor...")

    wfb_cmd = f"sudo wfb_tx -K {WFB_KEY_PATH} -i {WFB_LINK_ID} -p 0 -u {WFB_UDP_PORT} {iface}"
    wfb_process = subprocess.Popen(wfb_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[WFB-TX] ✓ PID {wfb_process.pid}")
    time.sleep(2)

    rpicam_cmd = f"""
    rpicam-vid -t 0 --inline --codec h264 \
      --width {WIDTH} --height {HEIGHT} --framerate {FPS} \
      --bitrate 3000000 --intra 30 \
      --listen -o tcp://127.0.0.1:{RPICAM_TCP_PORT}
    """
    rpicam_process = subprocess.Popen(rpicam_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[RPICAM] ✓ PID {rpicam_process.pid}, TCP {RPICAM_TCP_PORT}")
    time.sleep(5)

    # GStreamer asıl pipeline opencv_processing_thread içindeki ffmpeg_to_gstreamer()'da
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


# ==================== YARDIMCI FONKSİYONLAR ====================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, target_cx, target_cy):
    camera_pitch_rad = math.radians(CAMERA_PITCH)
    fov_x_rad = math.radians(CAMERA_FOV_H)
    fov_y_rad = math.radians(CAMERA_FOV_V)

    delta_x_pixel = target_cx - WIDTH / 2
    delta_y_pixel = HEIGHT / 2 - target_cy

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


def calculate_drop_point(target_lat, target_lon, alt, speed_ms, yaw_deg):
    """Balistik hesap: hedefe isabet için gereken release point"""
    g = 9.81
    fall_time = math.sqrt(2 * alt / g)
    horizontal_dist = speed_ms * fall_time

    yaw_rad = math.radians(yaw_deg)
    delta_north = -horizontal_dist * math.cos(yaw_rad)
    delta_east  = -horizontal_dist * math.sin(yaw_rad)

    release_lat = target_lat + (delta_north / 111320)
    release_lon = target_lon + (delta_east / (111320 * math.cos(math.radians(target_lat))))

    return release_lat, release_lon


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


# ==================== DETECTION + OVERLAY ====================

def opencv_processing_thread(queue):
    global ffmpeg_decode_process, ffmpeg_encode_process, detected_targets

    print("[OPENCV] Thread başlatıldı...")
    time.sleep(7)

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

    ffmpeg_encode_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "h264_v4l2m2m",
        "-b:v", "2000k",
        "-g", "15",
        "-f", "h264",
        "-",
    ]

    ffmpeg_encode_process = subprocess.Popen(
        ffmpeg_encode_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    print("[OPENCV] FFmpeg encode başladı (hardware)")

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
    gps_data = {}
    queued_colors = set()   # spam önlemi: her renk sadece 1 kez queue'ya girer
    frame_count = 0
    fps_time = time.time()
    current_fps = 0

    while True:
        try:
            raw_frame = ffmpeg_decode_process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                time.sleep(0.01)
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

            frame_count += 1
            if time.time() - fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_time = time.time()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # MAVİ KARE
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

                detected_targets["mavi"] = {"cx": cx, "cy": cy, "contour": cnt}
                blue_found = True

                if detection_active.is_set() and "mavi" not in queued_colors:
                    queue.put({"color": "mavi", "cx": cx, "cy": cy})
                    queued_colors.add("mavi")
                    print(f"✓ MAVİ KARE @ ({cx}, {cy}) [mission kuyruğuna eklendi]")
                break

            if not blue_found:
                detected_targets["mavi"] = None

            # KIRMIZI KARE
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

                if detection_active.is_set() and "kirmizi" not in queued_colors:
                    queue.put({"color": "kirmizi", "cx": cx, "cy": cy})
                    queued_colors.add("kirmizi")
                    print(f"✓ KIRMIZI KARE @ ({cx}, {cy}) [mission kuyruğuna eklendi]")
                break

            if not red_found:
                detected_targets["kirmizi"] = None

            # === OVERLAY ===
            for color, data in detected_targets.items():
                if data:
                    box_color = (255, 100, 0) if color == "mavi" else (0, 0, 255)
                    cv2.drawContours(frame, [data["contour"]], -1, box_color, 3)

                    if color in gps_data:
                        lat, lon = gps_data[color]
                        text = f"{color.upper()}: {lat:.6f}, {lon:.6f}"
                        cv2.putText(frame, text, (data["cx"] - 100, data["cy"] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            with telemetry_lock:
                tel = current_telemetry.copy()

            if tel["alt"]:
                cv2.putText(frame, f"ALT: {tel['alt']:.1f}m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            mavi_durum    = "MAVI: OK"    if detected_targets["mavi"]    else "MAVI: --"
            kirmizi_durum = "KIRMIZI: OK" if detected_targets["kirmizi"] else "KIRMIZI: --"
            mavi_renk    = (255, 100, 0) if detected_targets["mavi"]    else (128, 128, 128)
            kirmizi_renk = (0, 0, 255)   if detected_targets["kirmizi"] else (128, 128, 128)
            cv2.putText(frame, mavi_durum,    (WIDTH - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mavi_renk, 2)
            cv2.putText(frame, kirmizi_durum, (WIDTH - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kirmizi_renk, 2)

            ffmpeg_encode_process.stdin.write(frame.tobytes())

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


async def drop_trigger_task(drone, release_points):
    """GPS mesafesini izle, release point'e gelince doğru servoyu tetikle"""
    async for pos in drone.telemetry.position():
        for rp in release_points:
            if rp["dropped"]:
                continue

            dist = haversine(pos.latitude_deg, pos.longitude_deg, rp["lat"], rp["lon"])

            if dist < DROP_TRIGGER_RADIUS_M:
                print(f"[DROP] {rp['color'].upper()} hedefe {dist:.1f}m → BIRAKILIYOR")
                await drop_payload(rp["color"])
                rp["dropped"] = True

        if all(rp["dropped"] for rp in release_points):
            print("[DROP] ✓ Tüm yükler bırakıldı")
            break

        await asyncio.sleep(0.1)


async def rtl_after_mission(drone):
    async for finished in drone.mission.is_mission_finished():
        if finished:
            print("[MISSION] Görev tamamlandı → RTL")
            await drone.action.return_to_launch()
            return


async def takeover_with_drop_mission(drone, release_points):
    """
    GCS misyonunu devre dışı bırak, sadece drop WP'lerini yükle.
    Bitti → RTL (home pozisyonu FC'de kayıtlı).
    """
    items = []

    for rp in release_points:
        items.append(MissionItem(
            latitude_deg=rp["lat"],
            longitude_deg=rp["lon"],
            relative_altitude_m=rp["alt"],
            speed_m_s=DRONE_SPEED_MS,
            is_fly_through=True,
            gimbal_pitch_deg=float('nan'),
            gimbal_yaw_deg=float('nan'),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=float('nan'),
            camera_photo_interval_s=float('nan'),
            acceptance_radius_m=15.0,
            yaw_deg=float('nan'),
            camera_photo_distance_m=float('nan'),
            vehicle_action=MissionItem.VehicleAction.NONE
        ))

    await drone.mission.upload_mission(MissionPlan(items))
    await drone.mission.set_current_mission_item(0)
    await drone.mission.start_mission()

    asyncio.create_task(rtl_after_mission(drone))
    print(f"[MISSION] ✓ Drop misyonu aktif ({len(items)} WP) → Bitti → RTL")


async def mission_task(drone, queue):
    release_points = []
    first_detection_time = None

    # Her iki hedefi topla
    while len(release_points) < 2:
        try:
            target = queue.get(timeout=0.5)
        except Empty:
            # Tek hedef bulundu ama ikincisi SINGLE_TARGET_TIMEOUT_SEC'te gelmediyse devam et
            if len(release_points) == 1 and first_detection_time:
                elapsed = time.time() - first_detection_time
                if elapsed > SINGLE_TARGET_TIMEOUT_SEC:
                    print(f"[MISSION] ⚠ {SINGLE_TARGET_TIMEOUT_SEC}s içinde ikinci hedef bulunamadı, tek hedefle devam")
                    break
            await asyncio.sleep(0.1)
            continue

        # Telemetri hazır olana kadar item'ı geri koy (kaybetme)
        with telemetry_lock:
            tel = current_telemetry.copy()

        if None in tel.values():
            print("[MISSION] Telemetri hazır değil, item korunuyor...")
            queue.put(target)
            await asyncio.sleep(1)
            continue

        color = target["color"]

        if any(rp["color"] == color for rp in release_points):
            continue

        target_lat, target_lon = pixel_to_gps(
            tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
            target["cx"], target["cy"]
        )

        release_lat, release_lon = calculate_drop_point(
            target_lat, target_lon, tel["alt"], DRONE_SPEED_MS, tel["yaw"]
        )

        release_points.append({
            "color": color,
            "lat": release_lat,
            "lon": release_lon,
            "alt": tel["alt"],
            "dropped": False
        })

        if first_detection_time is None:
            first_detection_time = time.time()

        print(f"[MISSION] {color.upper()} release point: {release_lat:.6f}, {release_lon:.6f}")

    print(f"[MISSION] Tarama tamamlandı ({len(release_points)} hedef). {SCAN_EXIT_DELAY_SEC}s bekleniyor...")
    await asyncio.sleep(SCAN_EXIT_DELAY_SEC)

    await takeover_with_drop_mission(drone, release_points)
    asyncio.create_task(drop_trigger_task(drone, release_points))


# ==================== MAIN ====================

async def run():
    print("=" * 60)
    print("TEKNOFEST 2026 - Sabit Kanat - Görev 2")
    print("rpicam → OpenCV (overlay) → GStreamer → WFB-ng")
    print("=" * 60)

    init_servo()

    iface = setup_monitor_mode()
    if not iface:
        cleanup_servo()
        sys.exit(1)

    if not start_pipeline(iface):
        cleanup_servo()
        sys.exit(1)

    opencv_t = threading.Thread(target=opencv_processing_thread, args=(target_queue,), daemon=True)
    opencv_t.start()

    # Şartname: Direk 2 geçildikten sonra detection aktif olmalı
    def activate_detection():
        time.sleep(DETECTION_ACTIVE_DELAY_SEC)
        detection_active.set()
        print(f"[DETECTION] ✓ Aktif — Direk 2 geçildi varsayımı ({DETECTION_ACTIVE_DELAY_SEC}s)")

    threading.Thread(target=activate_detection, daemon=True).start()

    drone = System()
    print(f"\n[MAVSDK] {FC_PORT} bağlanıyor...")
    await drone.connect(system_address=f"serial://{FC_PORT}:{FC_BAUDRATE}")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAVSDK] ✓")
            break

    print("\n" + "=" * 60)
    print("✓ SİSTEM HAZIR")
    print(f"Tarama sonrası bekleme: {SCAN_EXIT_DELAY_SEC}s | Drop tetik: {DROP_TRIGGER_RADIUS_M}m")
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
        cleanup_servo()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n✓ Durduruldu")
        stop_pipeline()
        cleanup_servo()

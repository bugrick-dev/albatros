import math
import asyncio
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from collections import deque


WIDTH = 320
HEIGHT = 240
FPS = 15
MIN_AREA = max(200, int(0.0005 * WIDTH * HEIGHT))
MAX_AREA = int(0.2 * WIDTH * HEIGHT)

H_MIN, H_MAX = 100, 140
S_MIN, S_MAX = 90, 255
V_MIN, V_MAX = 50, 255

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={
    "format": "RGB888",
    "size": (WIDTH, HEIGHT)
})
picam2.configure(config)
picam2.start()

kernel = np.ones((5, 5), np.uint8)

target_queue = Queue()

telemetry_lock = threading.Lock()
current_telemetry = {"lat": None, "lon": None, "alt": None, "yaw": None}


def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, target_cx, target_cy):
    camera_angle_deg = math.radians(60.0)

    fov_x_rad = math.radians(62.2)
    fov_y_rad = math.radians(48.8)

    center_x = WIDTH / 2
    center_y = HEIGHT / 2

    delta_x_pixel = target_cx - center_x
    delta_y_pixel = center_y - target_cy

    offset_angle_x = delta_x_pixel * (fov_x_rad / WIDTH)
    offset_angle_y = delta_y_pixel * (fov_y_rad / HEIGHT)

    total_pitch = camera_angle_deg + offset_angle_y

    dist_y = alt * math.tan(total_pitch)
    dist_x = (alt / math.cos(total_pitch)) * math.tan(offset_angle_x)

    yaw_rad = math.radians(yaw_deg)
    delta_north = (dist_y * math.cos(yaw_rad)) - (dist_x * math.sin(yaw_rad))
    delta_east = (dist_y * math.sin(yaw_rad)) + (dist_x * math.cos(yaw_rad))

    target_lat = drone_lat + (delta_north / 111320)
    target_lon = drone_lon + (delta_east / (111320 * math.cos(math.radians(drone_lat))))

    return target_lat, target_lon


def detection_thread(queue):
    seen_targets = {}

    while True:
        frame = picam2.capture_array()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_blue = cv2.inRange(hsv, np.array([H_MIN, S_MIN, V_MIN]), np.array([H_MAX, S_MAX, V_MAX]))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        mask_red1 = cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([10, S_MAX, V_MAX]))
        mask_red2 = cv2.inRange(hsv, np.array([170, S_MIN, V_MIN]), np.array([180, S_MAX, V_MAX]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        for color, mask in [("mavi", mask_blue), ("kirmizi", mask_red)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA < area < MAX_AREA):
                    continue

                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
                corners = len(approx)

                # Sadece kare
                if corners != 4:
                    continue

                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width <= 0 or height <= 0:
                    continue
                aspectRatio = max(width, height) / min(width, height)
                if not (0.85 < aspectRatio < 1.15):
                    continue  # kare değil, dikdörtgen

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cx_bucket = cx // 50
                cy_bucket = cy // 50
                key = (color, cx_bucket, cy_bucket)
                now_check = time.time()

                if key in seen_targets and now_check - seen_targets[key] < 2.0:
                    continue

                seen_targets[key] = now_check
                queue.put({"color": color, "cx": cx, "cy": cy})
                print(f"Tespit: {color} kare @ ({cx}, {cy})")
                
# Getting telemetry info
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


# Assigning missions BUT NOT STARTING
async def mission_task(drone, queue):
    waypoints = {}  # key = color
    ELLIPSE_POINTS = 8
    ELLIPSE_RADIUS_M = 10.0  # hedef etrafında kaç metre yarıçap

    while True:
        try:
            target = queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.05)
            continue

        with telemetry_lock:
            tel = current_telemetry.copy()

        if None in tel.values():
            continue

        color = target["color"]
        if color in waypoints:
            continue

        center_lat, center_lon = pixel_to_gps(
            tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
            target["cx"], target["cy"]
        )

        # Hedef merkezi etrafında elips wp dizisi oluştur
        ellipse_items = []
        for i in range(ELLIPSE_POINTS):
            angle_rad = 2 * math.pi * i / ELLIPSE_POINTS
            # Kuzey-güney ekseni biraz daha geniş (1.5x) → elips efekti
            d_north = ELLIPSE_RADIUS_M * math.cos(angle_rad) * 1.5
            d_east  = ELLIPSE_RADIUS_M * math.sin(angle_rad)

            wp_lat = center_lat + (d_north / 111320)
            wp_lon = center_lon + (d_east  / (111320 * math.cos(math.radians(center_lat))))

            ellipse_items.append(MissionItem(
                latitude_deg=wp_lat,
                longitude_deg=wp_lon,
                relative_altitude_m=tel["alt"],
                speed_m_s=10.0,
                is_fly_through=True,
                gimbal_pitch_deg=float('nan'),
                gimbal_yaw_deg=float('nan'),
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=float('nan'),
                camera_photo_interval_s=float('nan'),
                acceptance_radius_m=3.0,
                yaw_deg=float('nan'),
                camera_photo_distance_m=float('nan'),
                vehicle_action=MissionItem.VehicleAction.NONE
            ))

        waypoints[color] = ellipse_items

        # Tüm renklerin wp'lerini düz listeye dök
        all_items = []
        for items in waypoints.values():
            all_items.extend(items)

        try:
            await drone.mission.upload_mission(MissionPlan(all_items))
            print(f"✓ Upload OK: {color} kare → merkez {center_lat:.6f}, {center_lon:.6f} | {ELLIPSE_POINTS} wp")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"✗ Upload FAILED: {e}")
            del waypoints[color]

        await asyncio.sleep(1.0)

async def run():
    drone = System()
    
    print("Cube Orange Plus'a ttyAMA3 üzerinden bağlanılıyor...")
    await drone.connect(system_address="serial:///dev/ttyAMA3:115200")
    
    print("Bağlantı bekleniyor...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone bağlandı!")
            break
    
    print("Veri akışı başlıyor... Çıkmak için Ctrl+C'ye basın.\n")
    
    t = threading.Thread(target=detection_thread, args=(target_queue,), daemon=True)
    t.start()

    try:
        await asyncio.gather(
            telemetry_task(drone),
            attitude_task(drone),
            mission_task(drone, target_queue),
        )
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\nProgram kullanıcı tarafından sonlandırıldı.")
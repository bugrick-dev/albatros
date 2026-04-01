import math
import asyncio
from picamera2 import Picamera2
import cv2
import numpy as np
import subprocess
import time
import os
import threading
from queue import Queue
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from collections import deque


DEST_IP = "192.168.50.23" #Ground Adapter IP
DEST_PORT = 5500
WIDTH = 640
HEIGHT = 480
FPS = 30
SNAPSHOT_DIR = "/home/albatros/albatros/snapshots"
SNAPSHOT_INTERVAL = 1 #Photos per second
snapshot_history = deque()

H_MIN, H_MAX = 100, 140
S_MIN, S_MAX = 90, 255
V_MIN, V_MAX = 50, 255

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

process_container = [process]

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={
    "format": "RGB888",
    "size": (WIDTH, HEIGHT)
})
picam2.configure(config)
picam2.start()


kernel = np.ones((5, 5), np.uint8)
last_snapshot = time.time()

target_queue = Queue()

telemetry_lock = threading.Lock()
current_telemetry = {"lat": None, "lon": None, "alt": None, "yaw": None}


def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, target_cx, target_cy):
    camera_angle_deg = 60.0

    fov_x_rad = math.radians(62.2)
    fov_y_rad = math.radians(48.8)

    center_x = WIDTH / 2
    center_y = HEIGHT / 2

    delta_x_pixel =  target_cx - center_x
    delta_y_pixel =  center_y - target_cy

    offset_angle_x = delta_x_pixel * (fov_x_rad / WIDTH)
    offset_angle_y = delta_y_pixel * (fov_y_rad / HEIGHT)

    total_pitch = camera_angle_deg + offset_angle_y

    dist_y = alt * math.tan(total_pitch)
    dist_x = (alt / math.cos(total_pitch)) * math.tan(offset_angle_x)
    
    yaw_rad = math.radians(yaw_deg)

    yaw_rad = math.radians(yaw_deg)
    delta_north = (dist_y * math.cos(yaw_rad)) - (dist_x * math.sin(yaw_rad))
    delta_east  = (dist_y * math.sin(yaw_rad)) + (dist_x * math.cos(yaw_rad))

    target_lat = drone_lat + (delta_north / 111320)
    target_lon = drone_lon + (delta_east  / (111320 * math.cos(math.radians(drone_lat))))

    return target_lat, target_lon


def detection_thread(queue, picam2, process):
    last_snapshot = time.time()
    while True:
        frame = picam2.capture_array()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mavi maske
        mask_blue = cv2.inRange(hsv, np.array([H_MIN, S_MIN, V_MIN]), np.array([H_MAX, S_MAX, V_MAX]))
        mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        # Kırmızı maske
        mask_red1 = cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([10, S_MAX, V_MAX]))
        mask_red2 = cv2.inRange(hsv, np.array([170, S_MIN, V_MIN]), np.array([180, S_MAX, V_MAX]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red1 = cv2.erode(mask_red1, kernel, iterations=2)
        mask_red2 = cv2.erode(mask_red2, kernel, iterations=2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

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

                        queue.put({
                            "color": color,
                            "isim": isim,
                            "cx": cx,
                            "cy": cy
                        })

        # UDP stream
        try:
            process_container[0].stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg koptu, yeniden başlatılıyor...")
            try:
                process_container[0] = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            except Exception as e:
                print(f"FFmpeg başlatılamadı: {e}")
                time.sleep(1)

        # Saniye başı snapshot
        now = time.time()
        if now - last_snapshot >= SNAPSHOT_INTERVAL:
            small = cv2.resize(frame, (320, 240))
            filename = os.path.join(SNAPSHOT_DIR, f"{int(now)}.jpg")
            ret = cv2.imwrite(filename, small)
            print(f"Snapshot: {filename} - {'OK' if ret else 'FAILED'}")
            
            if ret:
                snapshot_history.append(filename)
            
            while (len(snapshot_history)>100):
                oldest = snapshot_history.popleft()
                if os.path.exists(oldest):
                    os.remove(oldest)
            last_snapshot = now



#Getting telemetry info
async def telemetry_task(drone):
    async for position in drone.telemetry.position():
        with telemetry_lock:
            current_telemetry["lat"] = position.latitude_deg
            current_telemetry["lon"] = position.longitude_deg
            current_telemetry["alt"] = position.relative_altitude_m
#Getting telemety info
async def attitude_task(drone):
    async for attitude in drone.telemetry.attitude_euler():
        with telemetry_lock:
            current_telemetry["yaw"] = attitude.yaw_deg

#assigning missions BUT NOT STARTING
async def mission_task(drone, queue):
    waypoints = {}  # liste değil dict, key = renk
    while True:
        if not queue.empty():
            target = queue.get()
            with telemetry_lock:
                tel = current_telemetry.copy()
            
            if None not in tel.values():
                color = target["color"]
                
                # bu renk için zaten waypoint var mı
                if color in waypoints:
                    await asyncio.sleep(0.05)
                    continue
                
                lat, lon = pixel_to_gps(tel["lat"], tel["lon"], tel["alt"], tel["yaw"])
                
                item = MissionItem(
                    latitude_deg=lat,
                    longitude_deg=lon,
                    relative_altitude_m=tel["alt"],
                    speed_m_s=15.0,
                    is_fly_through=True,
                    gimbal_pitch_deg=float('nan'),
                    gimbal_yaw_deg=float('nan'),
                    camera_action=MissionItem.CameraAction.NONE,
                    loiter_time_s=float('nan'),
                    camera_photo_interval_s=float('nan'),
                    acceptance_radius_m=5.0,
                    yaw_deg=float('nan'),
                    camera_photo_distance_m=float('nan'),
                    vehicle_action=MissionItem.VehicleAction.NONE
                )
                waypoints[color] = item
                
                await drone.mission.upload_mission(MissionPlan(list(waypoints.values())))
                print(f"Waypoint eklendi: {color} {target['isim']} → {lat:.6f}, {lon:.6f}")
        
        await asyncio.sleep(0.05)

        
async def run():
    drone = System()
    
    print("Cube Orange Plus'a ttyAMA3 üzerinden bağlanılıyor...")
    await drone.connect(system_address="serial:///dev/ttyAMA3:57600")
    
    print("Bağlantı bekleniyor...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone bağlandı!")
            break
    
    print("Veri akışı başlıyor... Çıkmak için Ctrl+C'ye basın.\n")
    
    t = threading.Thread(target=detection_thread, args=(target_queue, picam2, process_container), daemon=True)
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
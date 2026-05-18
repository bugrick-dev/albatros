
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
import subprocess
import sys

# ==================== KONFİGÜRASYON ====================

# Kamera ayarları
WIDTH = 640  # Yüksek çözünürlük (hedef tespit için)
HEIGHT = 480
FPS = 30

# Hedef tespit parametreleri
MIN_AREA = int(0.001 * WIDTH * HEIGHT)  # Min 640 pixel
MAX_AREA = int(0.3 * WIDTH * HEIGHT)    # Max area

# MAVİ KARE (4x4m) - HSV değerleri
BLUE_H_MIN, BLUE_H_MAX = 100, 130
BLUE_S_MIN, BLUE_S_MAX = 80, 255
BLUE_V_MIN, BLUE_V_MAX = 50, 255

# KIRMIZI KARE (2x2m) - HSV değerleri (kırmızı 2 range'de)
RED1_H_MIN, RED1_H_MAX = 0, 10
RED2_H_MIN, RED2_H_MAX = 170, 180
RED_S_MIN, RED_S_MAX = 80, 255
RED_V_MIN, RED_V_MAX = 50, 255

# Şekil filtreleme (sadece KARE kabul et, altıgen/üçgen RED)
SQUARE_CORNER_TOLERANCE = 4  # 4 köşe = kare
ASPECT_RATIO_MIN = 0.7  # Kare aspect ratio toleransı
ASPECT_RATIO_MAX = 1.3

# WFB-ng ayarları
WFB_INTERFACE = "wlan0"
WFB_KEY_PATH = "/home/albatros/gs.key"
WFB_LINK_ID = "7669206"
WFB_UDP_PORT = 5600

# Flight Controller
FC_PORT = "/dev/ttyAMA3"
FC_BAUDRATE = 115200

# Kamera FOV (Pi Camera v2 için)
CAMERA_FOV_H = 62.2  # Horizontal FOV (derece)
CAMERA_FOV_V = 48.8  # Vertical FOV (derece)
CAMERA_PITCH = 60.0  # Kamera pitch açısı (aşağı doğru)

# ==================== GLOBAL DEĞİŞKENLER ====================

picam2 = None
kernel = np.ones((5, 5), np.uint8)
target_queue = Queue()
telemetry_lock = threading.Lock()
current_telemetry = {"lat": None, "lon": None, "alt": None, "yaw": None}
detected_targets = {"mavi": None, "kirmizi": None}  # Her hedeften 1 tane
wfb_process = None
gst_process = None

# ==================== GÖRÜNTÜ İŞLEME ====================

def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, target_cx, target_cy):
    """
    Pixel koordinatlarını GPS koordinatlarına çevir
    """
    camera_pitch_rad = math.radians(CAMERA_PITCH)
    fov_x_rad = math.radians(CAMERA_FOV_H)
    fov_y_rad = math.radians(CAMERA_FOV_V)
    
    center_x = WIDTH / 2
    center_y = HEIGHT / 2
    
    # Pixel offsetleri
    delta_x_pixel = target_cx - center_x
    delta_y_pixel = center_y - target_cy
    
    # Açısal offset
    offset_angle_x = delta_x_pixel * (fov_x_rad / WIDTH)
    offset_angle_y = delta_y_pixel * (fov_y_rad / HEIGHT)
    
    # Total pitch
    total_pitch = camera_pitch_rad + offset_angle_y
    
    # Mesafe hesaplama
    dist_y = alt * math.tan(total_pitch)
    dist_x = (alt / math.cos(total_pitch)) * math.tan(offset_angle_x)
    
    # Yaw dönüşümü
    yaw_rad = math.radians(yaw_deg)
    delta_north = (dist_y * math.cos(yaw_rad)) - (dist_x * math.sin(yaw_rad))
    delta_east = (dist_y * math.sin(yaw_rad)) + (dist_x * math.cos(yaw_rad))
    
    # GPS koordinatları
    target_lat = drone_lat + (delta_north / 111320)
    target_lon = drone_lon + (delta_east / (111320 * math.cos(math.radians(drone_lat))))
    
    return target_lat, target_lon


def is_square(contour):

    #Konturun kare olup olmadığını kontrol et (altıgen/üçgen filtreleme)

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    corners = len(approx)
    
    # Sadece 4 köşe kabul et
    if corners != SQUARE_CORNER_TOLERANCE:
        return False
    
    # Aspect ratio kontrolü
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width <= 0 or height <= 0:
        return False
    
    aspect_ratio = max(width, height) / min(width, height)
    if not (ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX):
        return False
    
    return True


def detection_thread(queue):

    #Sürekli görüntü işleme thread'i

    global picam2, detected_targets
    
    print("[DETECTION] Thread başlatıldı...")
    
    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # ===== MAVİ KARE TESPİTİ (4x4m) =====
            if detected_targets["mavi"] is None:
                mask_blue = cv2.inRange(hsv, 
                    np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]), 
                    np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX]))
                mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
                mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
                
                contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours_blue:
                    area = cv2.contourArea(cnt)
                    if not (MIN_AREA < area < MAX_AREA):
                        continue
                    
                    # KARE kontrolü (altıgen/üçgen OLMAMALI)
                    if not is_square(cnt):
                        continue
                    
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    detected_targets["mavi"] = {"cx": cx, "cy": cy, "area": area}
                    queue.put({"color": "mavi", "cx": cx, "cy": cy, "area": area})
                    print(f"✓ MAVİ KARE tespit edildi @ ({cx}, {cy}) - Alan: {area}")
                    break
            
            # ===== KIRMIZI KARE TESPİTİ (2x2m) =====
            if detected_targets["kirmizi"] is None:
                mask_red1 = cv2.inRange(hsv, 
                    np.array([RED1_H_MIN, RED_S_MIN, RED_V_MIN]), 
                    np.array([RED1_H_MAX, RED_S_MAX, RED_V_MAX]))
                mask_red2 = cv2.inRange(hsv, 
                    np.array([RED2_H_MIN, RED_S_MIN, RED_V_MIN]), 
                    np.array([RED2_H_MAX, RED_S_MAX, RED_V_MAX]))
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
                mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
                
                contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours_red:
                    area = cv2.contourArea(cnt)
                    if not (MIN_AREA < area < MAX_AREA):
                        continue
                    
                    # KARE kontrolü
                    if not is_square(cnt):
                        continue
                    
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    detected_targets["kirmizi"] = {"cx": cx, "cy": cy, "area": area}
                    queue.put({"color": "kirmizi", "cx": cx, "cy": cy, "area": area})
                    print(f"✓ KIRMIZI KARE tespit edildi @ ({cx}, {cy}) - Alan: {area}")
                    break
            
            # Her iki hedef de bulundu mu?
            if detected_targets["mavi"] and detected_targets["kirmizi"]:
                print("[DETECTION] İki hedef de bulundu, thread durduruluyor...")
                break
            
            time.sleep(0.05)  # 20 FPS görüntü işleme
            
        except Exception as e:
            print(f"[DETECTION ERROR] {e}")
            time.sleep(0.1)


# ==================== MAVSDK GÖREVLERİ ====================

async def telemetry_task(drone):
    """Telemetri dinleme"""
    async for position in drone.telemetry.position():
        with telemetry_lock:
            current_telemetry["lat"] = position.latitude_deg
            current_telemetry["lon"] = position.longitude_deg
            current_telemetry["alt"] = position.relative_altitude_m


async def attitude_task(drone):
    """Yaw dinleme"""
    async for attitude in drone.telemetry.attitude_euler():
        with telemetry_lock:
            current_telemetry["yaw"] = attitude.yaw_deg


async def mission_task(drone, queue):
    """
    Hedef tespit edildiğinde waypoint ekle
    """
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
            print("[MISSION] Telemetri henüz hazır değil, bekleniyor...")
            await asyncio.sleep(1)
            continue
        
        color = target["color"]
        
        if color in waypoints:
            print(f"[MISSION] {color.upper()} hedef zaten işlendi, atlanıyor...")
            continue
        
        # Pixel → GPS
        target_lat, target_lon = pixel_to_gps(
            tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
            target["cx"], target["cy"]
        )
        
        print(f"[MISSION] {color.upper()} hedef GPS: {target_lat:.7f}, {target_lon:.7f}")
        
        # Waypoint oluştur (hedef üzerinden geçiş)
        wp = MissionItem(
            latitude_deg=target_lat,
            longitude_deg=target_lon,
            relative_altitude_m=tel["alt"],  # Aynı irtifada
            speed_m_s=15.0,
            is_fly_through=True,
            gimbal_pitch_deg=float('nan'),
            gimbal_yaw_deg=float('nan'),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=float('nan'),
            camera_photo_interval_s=float('nan'),
            acceptance_radius_m=10.0,  # 10m kabul yarıçapı
            yaw_deg=float('nan'),
            camera_photo_distance_m=float('nan'),
            vehicle_action=MissionItem.VehicleAction.NONE
        )
        
        waypoints[color] = wp
        
        # Tüm waypoint'leri upload et
        all_waypoints = list(waypoints.values())
        
        try:
            await drone.mission.upload_mission(MissionPlan(all_waypoints))
            print(f"✓ Mission uploaded: {len(all_waypoints)} waypoint")
        except Exception as e:
            print(f"✗ Mission upload FAILED: {e}")
            del waypoints[color]
        
        await asyncio.sleep(1)


# ==================== WFB-NG VIDEO BROADCAST ====================

def start_wfb_broadcast():
    """
    WFB-ng + GStreamer video broadcast başlat
    """
    global wfb_process, gst_process
    
    print("[WFB-NG] Video broadcast başlatılıyor...")
    
    # WFB-ng TX
    wfb_cmd = [
        "sudo", "wfb_tx",
        "-K", WFB_KEY_PATH,
        "-i", WFB_LINK_ID,
        "-p", "0",
        "-u", str(WFB_UDP_PORT),
        WFB_INTERFACE
    ]
    
    try:
        wfb_process = subprocess.Popen(wfb_cmd, stderr=subprocess.DEVNULL)
        print(f"[WFB-NG] TX başlatıldı (PID: {wfb_process.pid})")
        time.sleep(2)
    except Exception as e:
        print(f"[WFB-NG ERROR] {e}")
        return False
    
    # GStreamer pipeline
    gst_cmd = [
        "gst-launch-1.0", "-v",
        "libcamerasrc", "!",
        f"video/x-raw,width={WIDTH},height={HEIGHT},framerate={FPS}/1", "!",
        "v4l2h264enc", "extra-controls=controls,video_bitrate=3000000", "!",
        "video/x-h264,level=(string)4.1,profile=baseline", "!",
        "h264parse", "!",
        "rtph264pay", "config-interval=1", "pt=96", "!",
        "udpsink", f"host=127.0.0.1", f"port={WFB_UDP_PORT}"
    ]
    
    try:
        gst_process = subprocess.Popen(gst_cmd, stderr=subprocess.DEVNULL)
        print(f"[GSTREAMER] Pipeline başlatıldı (PID: {gst_process.pid})")
        return True
    except Exception as e:
        print(f"[GSTREAMER ERROR] {e}")
        return False


def stop_wfb_broadcast():
    """
    WFB-ng broadcast durdur
    """
    global wfb_process, gst_process
    
    if gst_process:
        gst_process.terminate()
        print("[GSTREAMER] Durduruldu")
    
    if wfb_process:
        wfb_process.terminate()
        print("[WFB-NG] Durduruldu")


# ==================== ANA PROGRAM ====================

async def run():
    global picam2
    
    print("=" * 60)
    print("TEKNOFEST 2026 - Sabit Kanat İHA Yarışması - Görev 2")
    print("=" * 60)
    
    # Kamera başlat
    print("\n[KAMERA] Başlatılıyor...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={
        "format": "RGB888",
        "size": (WIDTH, HEIGHT)
    })
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("[KAMERA] ✓ Hazır")
    
    # WFB-ng broadcast başlat
    if not start_wfb_broadcast():
        print("[ERROR] WFB-ng başlatılamadı!")
        return
    
    # Flight Controller bağlan
    drone = System()
    print(f"\n[MAVSDK] {FC_PORT} üzerinden bağlanılıyor...")
    await drone.connect(system_address=f"serial://{FC_PORT}:{FC_BAUDRATE}")
    
    print("[MAVSDK] Bağlantı bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAVSDK] ✓ Bağlandı!")
            break
    
    # Görüntü işleme thread'i başlat
    detection_t = threading.Thread(target=detection_thread, args=(target_queue,), daemon=True)
    detection_t.start()
    
    print("\n" + "=" * 60)
    print("SİSTEM HAZIR - Uçuş başlayabilir!")
    print("Hedef tespit edildiğinde otomatik waypoint eklenecek")
    print("Ctrl+C ile çıkış")
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
        stop_wfb_broadcast()
        picam2.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\n✓ Program durduruldu")
        stop_wfb_broadcast()
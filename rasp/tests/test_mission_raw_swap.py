#!/usr/bin/env python3
"""
MİSYON DEĞİŞİMİ (mission_raw SWAP) TESTİ — uçuş gerektirmez, elle çalıştırılır.
vision.py'deki tam pipeline (FFmpeg encode + GStreamer + radyo yayını) hiç
başlatılmaz — yalnızca kamera açılır ve kare tespiti yapılır.

Ne yapar:
  1. FC'ye bağlanır, GCS'in yüklediği MEVCUT misyonu indirip ekrana basar.
  2. rpicam-vid + FFmpeg decode ile sade bir kamera akışı açar (vision.py'nin
     HSV/kare tespit fonksiyonlarını olduğu gibi kullanır — encode/GStreamer/
     radyo YOK).
  3. Mavi ve kırmızı kareleri kameraya okuttukça gerçek zamanlı release
     point hesaplar (geo.pixel_to_gps + geo.calculate_drop_point).
  4. 2 hedef toplanınca mission.build_and_start_drop_mission() çağrılır;
     YENİ misyon (drop + iniş WP'leri) indirilip ekrana basılır.
  5. mission_progress() akışını izleyip aktif WP'nin 0'dan itibaren
     ilerlediğini (başa sıçramadığını) gösterir.

Çalıştırma:
    cd rasp/tests
    python3 test_mission_raw_swap.py [--conn serial:///dev/ttyAMA3:115200]

Notlar:
  - Gerçek Pi üzerinde çalıştırılmalı (kamera + RPi.GPIO gerekiyor).
  - FC bağlı olmalı: gerçek uçuş kontrolcüsü (bench'te, disarm) ya da SITL.
    SITL için: --conn udp://:14540 gibi bir adres verin.
  - GCS'ten önceden gerçek görev planı FC'ye yüklenmiş olmalı —
    config.SEARCH_LOOP_EXIT_WP bu plana göre ayarlanmış olmalı.
  - start_mission() çağrılacağı için, drone arm'lıysa gerçekten drop+iniş
    sekansına geçmeye çalışır. Sadece WP listesinin değişimini görmek
    istiyorsanız drone'u DISARM bırakın.
"""
import argparse
import asyncio
import concurrent.futures
import os
import subprocess
import sys
import time

# stdout bazen pipe/redirect durumunda buffer'lanıyor — print() çağrıları
# program sonraki adımda (telemetri/GPS bekleme) takılırsa hiç ekrana
# düşmeyebilir. Tüm satırların anında görünmesi için satır bazlı flush'a zorla.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from mavsdk import System

import config
import state
import geo
import vision   # yalnızca saf tespit fonksiyonları kullanılacak (is_square, _apply_morph, ...)
import pipeline
import servo
import mission


def print_mission(items, title):
    print(f"\n{'='*72}\n{title} — {len(items)} öğe\n{'='*72}")
    for it in items:
        if it.command == config.CMD_NAV_WAYPOINT:
            print(f"  WP{it.seq:>2}  NAV_WAYPOINT     lat={it.x/1e7:.6f}  lon={it.y/1e7:.6f}  alt={it.z:.1f}m")
        elif it.command == config.CMD_DO_CHANGE_SPEED:
            print(f"  WP{it.seq:>2}  DO_CHANGE_SPEED  hız={it.param2}m/s")
        elif it.command == config.CMD_DO_SET_SERVO:
            print(f"  WP{it.seq:>2}  DO_SET_SERVO     kanal={int(it.param1)}  pwm={int(it.param2)}")
        elif it.command == config.CMD_CONDITION_DIST:
            print(f"  WP{it.seq:>2}  CONDITION_DIST   mesafe={it.param1}m")
        elif it.command == config.CMD_RTL:
            print(f"  WP{it.seq:>2}  RTL")
        else:
            print(f"  WP{it.seq:>2}  CMD={it.command}  p1={it.param1} p2={it.param2}")
    print("=" * 72 + "\n")


def start_camera():
    """Yalnızca kamerayı açar — encode/GStreamer/radyo yok."""
    cmd = (
        f"rpicam-vid -t 0 --inline --codec h264 "
        f"--width {config.WIDTH} --height {config.HEIGHT} --framerate {config.FPS} "
        f"--bitrate 2000000 --intra 30 "
        f"--listen -o tcp://127.0.0.1:{config.RPICAM_TCP_PORT}"
    )
    print(f"[TEST] rpicam-vid başlatılıyor: {cmd}")
    state.rpicam_process = subprocess.Popen(cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[TEST] ✓ rpicam-vid PID={state.rpicam_process.pid} — 5s ısınma bekleniyor...")
    time.sleep(5)

    decode_cmd = [
        "ffmpeg",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-probesize", "32", "-analyzeduration", "0",
        "-i", f"tcp://127.0.0.1:{config.RPICAM_TCP_PORT}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.WIDTH}x{config.HEIGHT}",
        "-r", str(config.FPS),
        "-",
    ]
    print(f"[TEST] FFmpeg decode başlatılıyor: {' '.join(decode_cmd)}")
    state.ffmpeg_decode_process = subprocess.Popen(
        decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    print(f"[TEST] ✓ FFmpeg decode PID={state.ffmpeg_decode_process.pid}")


# cv2.imshow/waitKey aynı OS thread'inden çağrılmalı — asyncio'nun varsayılan
# thread pool'u her seferinde farklı worker verebileceği için tek-thread'lik
# özel bir executor kullanıyoruz.
_vision_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

WINDOW_NAME = "Kare Tespiti — q: cik"


def detect_one_frame():
    """
    Tek kareyi okur, mavi/kırmızı kare arar, işlenmiş görüntüyü (kutu + etiket)
    cv2.imshow ile ekrana basar.
    Dönüş: {"mavi": (cx,cy)|None, "kirmizi": (cx,cy)|None} veya None (eksik frame).
    """
    frame_size = config.WIDTH * config.HEIGHT * 3
    raw = state.ffmpeg_decode_process.stdout.read(frame_size)
    if len(raw) != frame_size:
        return None

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((config.HEIGHT, config.WIDTH, 3)).copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    found = {}

    mask_blue = vision._apply_morph(cv2.inRange(hsv, config.BLUE_HSV_LOWER, config.BLUE_HSV_UPPER))
    r = vision._detect_square_in_mask(mask_blue)
    if r:
        cx, cy, cnt = r
        found["mavi"] = (cx, cy)
        cv2.drawContours(frame, [cnt], -1, (255, 100, 0), 3)
        cv2.putText(frame, f"MAVI ({cx},{cy})", (max(cx - 70, 0), max(cy - 15, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    else:
        found["mavi"] = None

    mask_red = vision._apply_morph(cv2.bitwise_or(
        cv2.inRange(hsv, config.RED1_HSV_LOWER, config.RED1_HSV_UPPER),
        cv2.inRange(hsv, config.RED2_HSV_LOWER, config.RED2_HSV_UPPER),
    ))
    r = vision._detect_square_in_mask(mask_red)
    if r:
        cx, cy, cnt = r
        found["kirmizi"] = (cx, cy)
        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
        cv2.putText(frame, f"KIRMIZI ({cx},{cy})", (max(cx - 80, 0), max(cy - 15, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        found["kirmizi"] = None

    with state.telemetry_lock:
        tel_ready = None not in state.current_telemetry.values()
    status_color = (0, 255, 0) if tel_ready else (0, 0, 255)
    cv2.putText(frame, f"GPS: {'VAR' if tel_ready else 'YOK (bekleniyor)'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

    return found


async def collect_two_targets():
    """
    Canlı görüntü penceresini hemen açar (GPS/telemetri beklemez) ve kareleri
    kameraya okutarak iki farklı renk toplanana kadar bekler. Telemetri henüz
    hazır değilse kare ekranda görünür ama release point'e dönüştürülmez —
    GPS fix gelince otomatik işlenir.
    """
    release_points = []
    seen_colors = set()
    loop = asyncio.get_running_loop()
    last_no_gps_warn = 0.0
    print("[TEST] Canlı görüntü penceresi açıldı. Mavi ve kırmızı kareleri kameraya okutun.")

    while len(release_points) < 2:
        result = await loop.run_in_executor(_vision_executor, detect_one_frame)
        if result is None:
            await asyncio.sleep(0.01)
            continue

        for color, pos in result.items():
            if pos is None or color in seen_colors:
                continue
            cx, cy = pos

            with state.telemetry_lock:
                tel = state.current_telemetry.copy()

            if None in tel.values():
                now = time.time()
                if now - last_no_gps_warn >= 2.0:
                    print(f"[TEST] {color.upper()} görüldü ama GPS/telemetri henüz hazır değil — bekleniyor...")
                    last_no_gps_warn = now
                continue  # seen_colors'a eklenmiyor, GPS gelince tekrar denenecek

            seen_colors.add(color)
            target_lat, target_lon = geo.pixel_to_gps(
                tel["lat"], tel["lon"], tel["alt"], tel["yaw"], cx, cy,
            )
            # Balistik drop noktası burada hesaplanmıyor — mission.drop_trigger_task
            # her tik'te güncel hızla taze hesaplıyor, burada hedefin kendi
            # GPS konumu yeterli.
            release_points.append({
                "color": color, "lat": target_lat, "lon": target_lon,
                "alt": tel["alt"], "dropped": False,
            })
            print(f"[TEST] ✓ {color.upper()} okundu → "
                  f"hedef=({target_lat:.6f},{target_lon:.6f}) | {len(release_points)}/2 hedef")

    return release_points


async def main(conn):
    print(f"[TEST] FC'ye bağlanılıyor: {conn}")
    drone = System()
    await drone.connect(system_address=conn)
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    print("[TEST] ✓ FC bağlandı")

    print("[TEST] Mevcut misyon FC'den indiriliyor...")
    try:
        existing = list(await drone.mission_raw.download_mission())
    except Exception as e:
        print(f"[TEST] ✗ HATA: download_mission() başarısız: {e}")
        print("[TEST] ✗ FC'de yüklü bir misyon yok olabilir — önce GCS'ten misyon yükleyin.")
        raise
    print_mission(existing, "GCS'TEN YÜKLÜ MEVCUT MİSYON (test öncesi)")

    # Telemetri arka planda akmaya başlar — kamerayı GPS fix'i BEKLEMEDEN açıyoruz,
    # GPS gelene kadar pencerede "GPS: YOK" görünür, gelince otomatik devam eder.
    asyncio.create_task(mission.telemetry_task(drone))
    asyncio.create_task(mission.attitude_task(drone))
    asyncio.create_task(mission.speed_track_task(drone))

    servo.init_servo()
    start_camera()

    try:
        release_points = await collect_two_targets()

        print("\n[TEST] 2 hedef toplandı — build_and_start_drop_mission() çağrılıyor...\n")
        await mission.build_and_start_drop_mission(drone, release_points)

        new_mission = list(await drone.mission_raw.download_mission())
        print_mission(new_mission, "YENİ YÜKLENEN MİSYON (drop + iniş) — test sonrası")

        print("[TEST] mission_progress izleniyor — WP'nin 0'dan itibaren "
              "ilerlediğini (başa SIÇRAMADIĞINI) doğrulayın. Ctrl+C ile çıkın.\n")
        async for progress in drone.mission_raw.mission_progress():
            print(f"[TEST] aktif WP {progress.current}/{progress.total}")

    finally:
        print("\n[TEST] Temizleniyor...")
        cv2.destroyAllWindows()
        pipeline.stop_pipeline()
        servo.cleanup_servo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conn", default=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}",
        help="MAVSDK bağlantı adresi (örn. serial://%(default)s veya udp://:14540 SITL için)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.conn))
    except KeyboardInterrupt:
        print("\n[TEST] Ctrl+C — durduruldu")
        pipeline.stop_pipeline()
        servo.cleanup_servo()

#!/usr/bin/env python3
"""
VİDEO PİPELİNE + TELEMETRİ TESTİ — uçuş gerektirmez, elle çalıştırılır.
main.py ile AYNI gerçek video pipeline'ını kullanır:
  rpicam-vid (TCP) → FFmpeg decode → OpenCV (tespit+overlay) → FFmpeg encode → GStreamer → WFB-ng
(vision.opencv_processing_thread olduğu gibi çalışır — kare tespiti, bounding box
ve telemetri overlay'i dahil tüm görüntü işleme asıl koddaki gibidir.)

asıl koddan (main.py) TEK FARK: GÖREV ATAMASI YOK.
  - mission_task çalışmıyor          → tespit edilen kareler release point'e
                                        dönüştürülmüyor, drop misyonu kurulmuyor.
  - speed_management_task çalışmıyor → DO_CHANGE_SPEED gönderilmiyor.
  - detection_activation_task çalışmıyor → state.detection_active hiç set
                                        edilmiyor, bu yüzden vision thread
                                        kareleri mission kuyruğuna da atmıyor
                                        (yine de ekranda bounding box + GPS
                                        etiketiyle gösteriliyor).

Çalışan görevler: yalnızca telemetry_task + attitude_task + waypoint_tracking_task
(ekranda lat/lon/alt/yaw/WP göstermek için) — FC üzerinde hiçbir komut/eylem
tetiklenmez.

Ne yapar:
  1. Servo GPIO başlatır (asıl koddaki pipeline init'iyle birebir aynı).
  2. WiFi monitor mode + WFB-ng + rpicam-vid başlatır (pipeline.start_pipeline).
  3. vision.opencv_processing_thread'i başlatır — FFmpeg decode/encode +
     GStreamer + HSV kare tespiti + bounding box + telemetri overlay'i.
  4. FC'ye bağlanır, yalnızca telemetri/yaw/WP akışlarını başlatır.
  5. Ctrl+C ile çıkılana kadar görüntü WFB üzerinden yayınlanır.

Çalıştırma:
    cd rasp/tests
    python3 test_video_pipeline_telemetry.py [--conn serial:///dev/ttyAMA3:115200]

Notlar:
  - Gerçek Pi üzerinde çalıştırılmalı (kamera + RPi.GPIO + WiFi adaptörü gerekiyor).
  - FC bağlantısı yalnızca telemetri okumak için kullanılır; FC'ye hiçbir
    misyon/komut gönderilmez. FC bağlanamazsa script yalnızca video modunda
    devam eder (telemetri overlay'i "YOK" gösterir).
"""
import argparse
import asyncio
import sys
import threading

sys.stdout.reconfigure(line_buffering=True)

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mavsdk import System

import config
import state
import servo
import pipeline
import vision
import mission


async def _wait_fc_connect(drone):
    async for conn in drone.core.connection_state():
        if conn.is_connected:
            return True
    return False


async def main(conn):
    print("=" * 60)
    print("GÖRÜNTÜ ALIM TESTİ — görev ataması YOK")
    print(f"  Video   : {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
    print(f"  FC      : {conn}")
    print(f"  WFB     : MAC={config.WFB_MAC}  kanal={config.WFB_CHANNEL}")
    print("=" * 60)

    # 1. GPIO servo başlat (asıl koddaki init'le birebir aynı)
    servo.init_servo()

    # 2. WiFi monitor mode
    iface = pipeline.setup_monitor_mode()
    if not iface:
        print("[TEST] Monitor mode kurulamadı — çıkılıyor")
        servo.cleanup_servo()
        sys.exit(1)

    # 3. Video pipeline (WFB-ng + rpicam) — asıl koddaki gibi
    if not pipeline.start_pipeline(iface):
        print("[TEST] Pipeline başlatılamadı — çıkılıyor")
        servo.cleanup_servo()
        sys.exit(1)

    # 4. OpenCV thread — tespit + bounding box + telemetri overlay (asıl koddaki gibi)
    print("[TEST] OpenCV thread başlatılıyor...")
    opencv_thread = threading.Thread(
        target=vision.opencv_processing_thread,
        args=(state.target_queue,),  # kuyruk dolduruluyor ama tüketen yok — görev ataması bilinçli olarak çalışmıyor
        daemon=True,
    )
    opencv_thread.start()
    print(f"[TEST] OpenCV thread başladı (TID={opencv_thread.ident})")

    # 5. MAVSDK bağlantısı — yalnızca telemetri için
    print(f"\n[TEST] MAVSDK bağlanıyor: {conn}")
    drone = System()
    await drone.connect(system_address=conn)

    print(f"[TEST] FC bağlantısı bekleniyor (max {config.FC_CONNECT_TIMEOUT_SEC}s)...")
    fc_connected = False
    try:
        fc_connected = await asyncio.wait_for(
            _wait_fc_connect(drone), timeout=config.FC_CONNECT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        pass

    tasks = []
    if fc_connected:
        print("[TEST] ✓ FC bağlandı — telemetri/WP akışları başlatılıyor (görev ataması YOK)")
        tasks = [
            asyncio.create_task(mission.telemetry_task(drone)),
            asyncio.create_task(mission.attitude_task(drone)),
            asyncio.create_task(mission.waypoint_tracking_task(drone)),
        ]
    else:
        print("[TEST] ⚠ FC bağlantısı kurulamadı — yalnızca video modu aktif "
              "(telemetri overlay'i 'YOK' gösterecek)")

    print("\n" + "=" * 60)
    print("✓ GÖRÜNTÜ AKIŞI HAZIR — Ctrl+C ile çıkın")
    print("=" * 60 + "\n")

    try:
        if tasks:
            await asyncio.gather(*tasks)
        else:
            await asyncio.Event().wait()
    except asyncio.CancelledError:
        print("[TEST] asyncio.CancelledError — durduruldu")
    finally:
        print("[TEST] Temizleniyor...")
        pipeline.stop_pipeline()
        servo.cleanup_servo()
        print("[TEST] ✓ Temizlik tamamlandı")


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

#!/usr/bin/env python3
"""
TEKNOFEST 2026 - Sabit Kanat - Görev 2
Giriş noktası.

Pipeline:
  rpicam-vid (TCP) → FFmpeg decode → OpenCV (tespit+overlay) → FFmpeg encode → GStreamer → WFB-ng

Modüller:
  config   — tüm sabitler
  state    — paylaşılan değişken global durum
  geo      — coğrafi hesaplamalar
  servo    — GPIO servo kontrolü
  pipeline — WFB-ng / rpicam-vid süreçleri
  vision   — OpenCV thread
  mission  — MAVSDK görevleri
"""
import asyncio
import sys
import threading

from mavsdk import System

import config
import state
import servo
import pipeline
import vision
import mission


async def run():
    print("=" * 60)
    print("TEKNOFEST 2026 - Sabit Kanat - Görev 2")
    print(f"  Video   : {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
    print(f"  FC      : {config.FC_PORT} @ {config.FC_BAUDRATE}baud")
    print(f"  WFB     : MAC={config.WFB_MAC}  kanal={config.WFB_CHANNEL}")
    print(f"  Tarama  : WP {config.SEARCH_START_WP} → {config.SEARCH_SPEED_MS}m/s | "
          f"loop çıkış WP={config.SEARCH_LOOP_EXIT_WP}")
    print(f"  Drop    : tetik={config.DROP_TRIGGER_RADIUS_M}m | "
          f"USE_FC_SERVO={config.USE_FC_SERVO}")
    print(f"  Tespit  : WP {config.DETECTION_ACTIVE_WP}'de aktif")
    print("=" * 60)

    # 1. GPIO servo başlat
    servo.init_servo()

    # 2. WiFi monitor mode
    iface = pipeline.setup_monitor_mode()
    if not iface:
        print("[MAIN] Monitor mode kurulamadı — çıkılıyor")
        servo.cleanup_servo()
        sys.exit(1)

    # 3. Video pipeline (WFB-ng + rpicam)
    if not pipeline.start_pipeline(iface):
        print("[MAIN] Pipeline başlatılamadı — çıkılıyor")
        servo.cleanup_servo()
        sys.exit(1)

    # 4. OpenCV thread
    print("[MAIN] OpenCV thread başlatılıyor...")
    opencv_thread = threading.Thread(
        target=vision.opencv_processing_thread,
        args=(state.target_queue,),
        daemon=True,
    )
    opencv_thread.start()
    print(f"[MAIN] OpenCV thread başladı (TID={opencv_thread.ident})")

    # 5. MAVSDK bağlantısı
    print(f"\n[MAIN] MAVSDK bağlanıyor: serial://{config.FC_PORT}:{config.FC_BAUDRATE}")

    async def _wait_fc_connect(drone):
        async for conn in drone.core.connection_state():
            if conn.is_connected:
                return True
        return False

    drone = System()
    await drone.connect(system_address=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}")

    print(f"[MAIN] FC bağlantısı bekleniyor (max {config.FC_CONNECT_TIMEOUT_SEC}s)...")
    fc_connected = False
    try:
        fc_connected = await asyncio.wait_for(
            _wait_fc_connect(drone),
            timeout=config.FC_CONNECT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        pass

    if not fc_connected:
        print(f"[MAIN] ⚠ FC bağlantısı kurulamadı — yalnızca video modu aktif")
        print("[MAIN] Görüntü yayını devam ediyor, Ctrl+C ile çıkın")
        await asyncio.Event().wait()
        return

    print("\n" + "=" * 60)
    print("✓ SİSTEM HAZIR — tüm görevler başlıyor")
    print(f"  Tarama sonrası bekleme : {config.SCAN_EXIT_DELAY_SEC}s")
    print(f"  Tek hedef timeout      : {config.SINGLE_TARGET_TIMEOUT_SEC}s")
    print(f"  Drop tetik mesafesi    : {config.DROP_TRIGGER_RADIUS_M}m")
    print("=" * 60 + "\n")

    try:
        await asyncio.gather(
            mission.telemetry_task(drone),
            mission.attitude_task(drone),
            mission.speed_track_task(drone),
            mission.mission_task(drone, state.target_queue),
            mission.speed_management_task(drone),
            mission.detection_activation_task(drone),
            mission.waypoint_tracking_task(drone),
        )
    except asyncio.CancelledError:
        print("[MAIN] asyncio.CancelledError — görev iptal edildi")
    finally:
        print("[MAIN] Temizleniyor...")
        pipeline.stop_pipeline()
        servo.cleanup_servo()
        print("[MAIN] ✓ Temizlik tamamlandı")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C — durduruldu")
        pipeline.stop_pipeline()
        servo.cleanup_servo()

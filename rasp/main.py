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
import logging
import sys
import threading

from mavsdk import System

import config
import state
import servo
import pipeline
import vision
import mission
import logsetup

log = logging.getLogger("main")


async def run():
    log.info("=" * 60)
    log.info("TEKNOFEST 2026 - Sabit Kanat - Görev 2")
    log.info(f"  Video   : {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
    log.info(f"  FC      : {config.FC_PORT} @ {config.FC_BAUDRATE}baud")
    log.info(f"  WFB     : MAC={config.WFB_MAC}  kanal={config.WFB_CHANNEL}")
    log.info(f"  Tarama  : WP {config.SEARCH_START_WP} → {config.SEARCH_SPEED_MS}m/s | "
          f"loop çıkış WP={config.SEARCH_LOOP_EXIT_WP}")
    log.info(f"  Drop    : tetik={config.DROP_TRIGGER_RADIUS_M}m | "
          f"USE_FC_SERVO={config.USE_FC_SERVO}")
    log.info(f"  Tespit  : WP {config.DETECTION_ACTIVE_WP}'de aktif")
    log.info("=" * 60)

    # 1. GPIO servo başlat
    servo.init_servo()

    # 2. WiFi monitor mode
    iface = pipeline.setup_monitor_mode()
    if not iface:
        log.info("[MAIN] Monitor mode kurulamadı — çıkılıyor")
        servo.cleanup_servo()
        sys.exit(1)

    # 3. Video pipeline (WFB-ng + rpicam)
    if not pipeline.start_pipeline(iface):
        log.info("[MAIN] Pipeline başlatılamadı — çıkılıyor")
        servo.cleanup_servo()
        sys.exit(1)

    # 4. OpenCV thread
    log.info("[MAIN] OpenCV thread başlatılıyor...")
    opencv_thread = threading.Thread(
        target=vision.opencv_processing_thread,
        args=(state.target_queue,),
        daemon=True,
    )
    opencv_thread.start()
    log.info(f"[MAIN] OpenCV thread başladı (TID={opencv_thread.ident})")

    # 5. MAVSDK bağlantısı
    log.info(f"\n[MAIN] MAVSDK bağlanıyor: serial://{config.FC_PORT}:{config.FC_BAUDRATE}")

    async def _wait_fc_connect(drone):
        async for conn in drone.core.connection_state():
            if conn.is_connected:
                return True
        return False

    drone = System()
    await drone.connect(system_address=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}")

    log.info(f"[MAIN] FC bağlantısı bekleniyor (max {config.FC_CONNECT_TIMEOUT_SEC}s)...")
    fc_connected = False
    try:
        fc_connected = await asyncio.wait_for(
            _wait_fc_connect(drone),
            timeout=config.FC_CONNECT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        pass

    state.fc_connected = fc_connected

    if not fc_connected:
        log.info(f"[MAIN] ⚠ FC bağlantısı kurulamadı — yalnızca video modu aktif")
        log.info("[MAIN] Görüntü yayını devam ediyor, Ctrl+C ile çıkın")
        asyncio.create_task(mission.fc_connection_task(drone))
        await asyncio.Event().wait()
        return

    log.info("\n" + "=" * 60)
    log.info("✓ SİSTEM HAZIR — tüm görevler başlıyor")
    log.info(f"  Tarama sonrası bekleme : {config.SCAN_EXIT_DELAY_SEC}s")
    log.info(f"  Tek hedef timeout      : {config.SINGLE_TARGET_TIMEOUT_SEC}s")
    log.info(f"  Drop tetik mesafesi    : {config.DROP_TRIGGER_RADIUS_M}m")
    log.info("=" * 60 + "\n")

    try:
        await asyncio.gather(
            mission.fc_connection_task(drone),
            mission.telemetry_task(drone),
            mission.attitude_task(drone),
            mission.speed_track_task(drone),
            mission.mission_task(drone, state.target_queue),
            mission.speed_management_task(drone),
            mission.detection_activation_task(drone),
            mission.waypoint_tracking_task(drone),
        )
    except asyncio.CancelledError:
        log.info("[MAIN] asyncio.CancelledError — görev iptal edildi")
    finally:
        log.info("[MAIN] Temizleniyor...")
        state.shutdown_requested.set()
        pipeline.stop_pipeline()
        servo.cleanup_servo()
        log.info("[MAIN] ✓ Temizlik tamamlandı")


if __name__ == "__main__":
    logsetup.setup_logging()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("\n[MAIN] Ctrl+C — durduruldu")
        state.shutdown_requested.set()
        pipeline.stop_pipeline()
        servo.cleanup_servo()

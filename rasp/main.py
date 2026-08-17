#!/usr/bin/env python3
"""
TEKNOFEST 2026 - Sabit Kanat - Görev 2
Giriş noktası.

Pipeline:
  rpicam-vid (raw stdout) → OpenCV (tespit+overlay) → FFmpeg encode → GStreamer → WFB-ng
  (2026-08-16: ara FFmpeg decode adımı kaldırıldı — bkz. vision.py notu)

Modüller:
  config   — tüm sabitler
  state    — paylaşılan değişken global durum
  geo      — coğrafi hesaplamalar
  pipeline — WFB-ng / rpicam-vid süreçleri
  vision   — OpenCV thread
  mission  — MAVSDK görevleri (servo tetikleme dahil — FC üzerinden, GPIO yok)
"""
import asyncio
import logging
import sys
import threading

from mavsdk import System

import config
import state
import pipeline
import vision
import mission
import logsetup

log = logging.getLogger("main")


async def run():
    log.info("=" * 60)
    log.info("TEKNOFEST 2026 - Sabit Kanat - Görev 2")
    log.info(f"  Video   : {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
    if config.CAMERA_CALIBRATED:
        log.info(f"  Kamera  : kalibreli (fx={config.CAMERA_FX:.1f} fy={config.CAMERA_FY:.1f}) ✓")
    elif config.CAMERA_CALIB_ERROR:
        log.info(f"  Kamera  : ‼ KALİBRASYON DOSYASI BOZUK ({config.CAMERA_CALIB_ERROR}) — "
                  f"FOV tahminine düşüldü, GPS hesabı BÜYÜK hata verir! "
                  f"camera_calib.json onarılmadan UÇMAYIN")
    else:
        log.info(f"  Kamera  : ⚠ KALİBRASYON YOK — FOV tahmini ({config.CAMERA_FOV_H}°/{config.CAMERA_FOV_V}°) kullanılıyor, "
                  f"GPS hesabı hatalı olabilir! bkz. tools/camera_calibrate.py")
    log.info(f"  FC      : {config.FC_PORT} @ {config.FC_BAUDRATE}baud")
    log.info(f"  WFB     : MAC={config.WFB_MAC}  kanal={config.WFB_CHANNEL}")
    log.info(f"  Tarama  : WP {config.SEARCH_START_WP} → {config.SEARCH_SPEED_MS}m/s | "
          f"loop çıkış WP={config.SEARCH_LOOP_EXIT_WP}")
    log.info(f"  Drop    : kurma={config.DROP_TRIGGER_RADIUS_M}m "
             f"cross-max={config.DROP_MAX_CROSS_TRACK_M}m t-tetik={config.DROP_ALONG_TRIGGER_S}s (FC servo)")
    log.info(f"  Tarama zaman aşımı : {config.SEARCH_TOTAL_TIMEOUT_SEC}s (hedefsiz kalınırsa inişe geç)")
    log.info(f"  Tespit  : WP {config.DETECTION_ACTIVE_WP}'de aktif")
    log.info("=" * 60)

    # 1. WiFi monitor mode
    # NOT (2026-08-16): GPIO servo yolu kaldırıldı — yükler artık sadece FC
    # üzerinden (DO_SET_SERVO) tetikleniyor, RPi GPIO'ya hiçbir servo
    # bağlanmayacak (bkz. servo.py silindi, mission.py drop_trigger_task).
    iface = pipeline.setup_monitor_mode()
    if not iface:
        log.info("[MAIN] Monitor mode kurulamadı — çıkılıyor")
        sys.exit(1)

    # 2. Video pipeline (WFB-ng + rpicam)
    if not pipeline.start_pipeline(iface):
        log.info("[MAIN] Pipeline başlatılamadı — çıkılıyor")
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
            mission.gps_health_task(drone),
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
        log.info("[MAIN] ✓ Temizlik tamamlandı")


if __name__ == "__main__":
    logsetup.setup_logging()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("\n[MAIN] Ctrl+C — durduruldu")
        state.shutdown_requested.set()
        pipeline.stop_pipeline()

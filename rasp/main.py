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
    log.info(f"  Tarama  : WP {config.SEARCH_START_WP} (hız yönetimi YOK — kendi hızında) | "
          f"loop çıkış WP={config.SEARCH_LOOP_EXIT_WP}")
    log.info(f"  Drop    : kurma={config.DROP_TRIGGER_RADIUS_M}m "
             f"cross-max={config.DROP_MAX_CROSS_TRACK_M}m t-tetik={config.DROP_ALONG_TRIGGER_S}s (FC servo)")
    log.info(f"  Tarama zaman aşımı : {config.SEARCH_TOTAL_TIMEOUT_SEC}s (hedefsiz kalınırsa inişe geç)")
    log.info(f"  Tespit  : WP {config.DETECTION_ACTIVE_WP}'de aktif")
    log.info("=" * 60)

    # 1-2. Video pipeline (rpicam-vid + WFB-ng)
    # NOT (2026-08-16): GPIO servo yolu kaldırıldı — yükler artık sadece FC
    # üzerinden (DO_SET_SERVO) tetikleniyor, RPi GPIO'ya hiçbir servo
    # bağlanmayacak (bkz. servo.py silindi, mission.py drop_trigger_task).
    # NOT (2026-08-19): WFB dongle bulunamazsa artık süreç sonlanmıyor —
    # rpicam-vid + OpenCV tespiti dongle'dan bağımsız hemen başlar, WFB
    # dongle arka planda süresiz aranır (bkz. pipeline._wfb_startup_retry_loop).
    # Eskiden dongle yokken (takılı değil / undervoltage ile USB'den düşmüş)
    # sys.exit(1) tüm süreci kapatıp systemd'yi 3sn'de bir crash-loop'a
    # sokuyordu, görüntü işleme ve mission görevleri de dongle bekleyerek hiç
    # başlamıyordu (sahada tekrarlanan restart sorunu).
    pipeline.start_pipeline()

    # 4. OpenCV thread
    log.info("[MAIN] OpenCV thread başlatılıyor...")
    opencv_thread = threading.Thread(
        target=vision.opencv_processing_thread,
        args=(state.target_queue,),
        daemon=True,
    )
    opencv_thread.start()
    log.info(f"[MAIN] OpenCV thread başladı (TID={opencv_thread.ident})")

    # 5. MAVSDK bağlantısı — FC bulunana kadar FAILSAFE ile SÜRESİZ dener.
    # NOT (2026-08-20 kök neden): drone.connect()'in KENDİSİ bazen hiç
    # dönmüyor (ör. seri port o an yoksa mavsdk_server içeride askıda
    # kalabiliyor). Eskiden bu çağrı sarmalanmamıştı — tek bir askıda kalan
    # deneme mission kodunu (WP takibi, tespit, servo/drop) saatlerce (bir
    # sahada gözlemlenen: 19 saat) hiç çalıştırmadan durduruyordu, üstelik
    # sessizce (bir sonraki log satırı bile basılmıyordu). Artık HEM
    # connect() çağrısının kendisi HEM de sonrasındaki is_connected bekleyişi
    # ayrı ayrı timeout'a sarılı; biri takılır/başarısız olursa YENİ bir
    # System() ile baştan denenir. Video akışı bu döngüden bağımsız zaten
    # sürüyor (ayrı thread) — bu döngü yalnızca mission görevlerinin ne
    # zaman başlayacağını belirliyor, o yüzden süresiz beklemek video
    # yayınını ETKİLEMEZ.
    log.info(f"\n[MAIN] MAVSDK bağlanıyor: serial://{config.FC_PORT}:{config.FC_BAUDRATE}")

    async def _wait_fc_connect(drone):
        async for conn in drone.core.connection_state():
            if conn.is_connected:
                return True
        return False

    drone = None
    attempt = 0
    while not state.shutdown_requested.is_set():
        attempt += 1
        candidate = System()
        try:
            await asyncio.wait_for(
                candidate.connect(system_address=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}"),
                timeout=config.FC_CONNECT_ATTEMPT_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.info(f"[MAIN] ⚠ FC connect() #{attempt} {config.FC_CONNECT_ATTEMPT_TIMEOUT_SEC}s içinde "
                     f"dönmedi (askıda kaldı) — yeni denemeye geçiliyor")
            await asyncio.sleep(config.FC_RECONNECT_INTERVAL_SEC)
            continue
        except Exception as e:
            log.info(f"[MAIN] ⚠ FC connect() #{attempt} hata: {e} — yeni denemeye geçiliyor")
            await asyncio.sleep(config.FC_RECONNECT_INTERVAL_SEC)
            continue

        log.info(f"[MAIN] FC bağlantısı bekleniyor (deneme #{attempt}, max {config.FC_CONNECT_TIMEOUT_SEC}s)...")
        try:
            fc_connected = await asyncio.wait_for(
                _wait_fc_connect(candidate),
                timeout=config.FC_CONNECT_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            fc_connected = False

        if fc_connected:
            drone = candidate
            state.fc_connected = True
            log.info(f"[MAIN] ✓ FC bağlandı (deneme #{attempt})")
            break

        log.info(f"[MAIN] ⚠ FC {config.FC_CONNECT_TIMEOUT_SEC}s içinde bağlanamadı "
                 f"(deneme #{attempt}) — yeniden denenecek")
        await asyncio.sleep(config.FC_RECONNECT_INTERVAL_SEC)

    if drone is None:
        # Buraya SADECE state.shutdown_requested (Ctrl+C) ile düşülür.
        log.info("[MAIN] Kapanış istendi — FC bağlantı döngüsünden çıkılıyor")
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

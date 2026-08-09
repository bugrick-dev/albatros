#!/usr/bin/env python3
"""
MİSYON DEĞİŞİMİ (mission_raw SWAP) TESTİ — ORBIT VARYASYONU + WFB YAYINI
uçuş gerektirmez, elle çalıştırılır. test_mission_raw_swap.py ile AYNI akış,
TEK FARK: 2 hedef toplanınca yük bırakma (servo) YOK — bunun yerine her
hedefin üstüne gidilip ORBIT_TURNS tur boyunca ORBIT_RADIUS_M yarıçapında
dönülür (NAV_WAYPOINT + NAV_LOITER_TURNS). Gerçek göreve yakın (kamera +
FC + mission_raw upload) ama yük mekanizması hiç devreye girmiyor.

main.py ile AYNI gerçek video pipeline'ını kullanır:
  rpicam-vid (TCP) → FFmpeg decode → OpenCV (tespit+overlay) → FFmpeg encode → GStreamer → WFB-ng
(önceki sürümden FARK: kare tespiti artık yer istasyonuna WFB üzerinden
yayınlanıyor — yerel cv2.imshow penceresi YOK, Pi'ye ekran bağlamaya gerek yok.)

Ne yapar:
  1. FC'ye bağlanır, GCS'in yüklediği MEVCUT misyonu indirip ekrana basar.
  2. WiFi monitor mode + WFB-ng + rpicam-vid başlatır (pipeline.start_pipeline)
     ve vision.opencv_processing_thread'i başlatır — HSV kare tespiti,
     bounding box + telemetri overlay'i WFB üzerinden yer istasyonuna akar.
  3. state.detection_active hemen set edilir (WP'ye bağlı aktivasyon YOK —
     test başladığı andan itibaren tespit aktif).
  4. Mavi ve kırmızı kareler vision thread tarafından state.target_queue'ya
     düştükçe gerçek zamanlı hedef GPS konumu hesaplanır (geo.pixel_to_gps)
     — release point/balistik hesap YOK, doğrudan hedefin üstünde
     dönüleceği için gerek yok.
  5. 2 hedef toplanınca build_and_start_orbit_mission() çağrılır; YENİ
     misyon (orbit öğeleri + iniş WP'leri) indirilip ekrana basılır.
  6. mission_progress() akışını izleyip aktif WP'nin 0'dan itibaren
     ilerlediğini (başa sıçramadığını) gösterir.

Çalıştırma:
    cd rasp/tests
    python3 test_mission_orbit_swap.py [--conn serial:///dev/ttyAMA3:921600]

Notlar:
  - Gerçek Pi üzerinde çalıştırılmalı (kamera + WiFi adaptörü gerekiyor,
    servo/GPIO YOK).
  - FC bağlı olmalı: gerçek uçuş kontrolcüsü (bench'te, disarm) ya da SITL.
    SITL için: --conn udp://:14540 gibi bir adres verin.
  - GCS'ten önceden gerçek görev planı FC'ye yüklenmiş olmalı —
    config.SEARCH_LOOP_EXIT_WP bu plana göre ayarlanmış olmalı.
  - start_mission() çağrılacağı için, drone arm'lıysa gerçekten orbit+iniş
    sekansına geçmeye çalışır. Sadece WP listesinin değişimini görmek
    istiyorsanız drone'u DISARM bırakın.
"""
import argparse
import asyncio
import os
import sys
import threading
import time
from queue import Empty

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mavsdk import System
from mavsdk.mission_raw import MissionItem as RawMissionItem

import config
import state
import geo
import pipeline
import vision
import mission  # _make_mission_item yardımcı fonksiyonu için


def print_mission(items, title):
    print(f"\n{'='*72}\n{title} — {len(items)} öğe\n{'='*72}")
    for it in items:
        if it.command == config.CMD_NAV_WAYPOINT:
            print(f"  WP{it.seq:>2}  NAV_WAYPOINT     lat={it.x/1e7:.6f}  lon={it.y/1e7:.6f}  alt={it.z:.1f}m")
        elif it.command == config.CMD_NAV_LOITER_TURNS:
            print(f"  WP{it.seq:>2}  NAV_LOITER_TURNS tur={it.param1:.0f}  yarıçap={it.param3:.0f}m  "
                  f"lat={it.x/1e7:.6f}  lon={it.y/1e7:.6f}  alt={it.z:.1f}m")
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


# ==================== ORBIT MİSYONU (drop yerine) ====================

def _build_orbit_items(targets, approach_alt_m, approach_frame=3):
    """
    Her hedef için: NAV_WAYPOINT ile hedefe git, ardından NAV_LOITER_TURNS
    ile ORBIT_TURNS tur boyunca ORBIT_RADIUS_M yarıçapında hedef üstünde dön.
    Servo/GPIO YOK — mission._build_drop_items'ın orbit karşılığı.

    approach_alt_m/approach_frame: t["alt"] (tespit anındaki ANLIK telemetri
    irtifası) DEĞİL, GCS planındaki SEARCH_START_WP öğesinden — aynı sebep
    mission._build_drop_items'ta (2026-08-09: sahada gözlenen ani dalış).
    """
    items = []
    for i, t in enumerate(targets):
        print(f"[TEST] === Orbit öğeleri: Hedef {i+1} {t['color'].upper()} "
              f"({t['lat']:.6f},{t['lon']:.6f}) tespit-anı-irtifası={t['alt']:.1f}m "
              f"| WP irtifası (planlı)={approach_alt_m:.1f}m ===")

        items.append(mission._make_mission_item(
            0, config.CMD_NAV_WAYPOINT, param2=15.0,
            lat=t["lat"], lon=t["lon"], alt=approach_alt_m, frame=approach_frame,
        ))
        items.append(mission._make_mission_item(
            0, config.CMD_NAV_LOITER_TURNS,
            param1=config.ORBIT_TURNS, param3=config.ORBIT_RADIUS_M,
            lat=t["lat"], lon=t["lon"], alt=approach_alt_m, frame=approach_frame,
        ))
    return items


async def build_and_start_orbit_mission(drone, targets):
    """
    mission.build_and_start_drop_mission ile aynı iskelet: eski misyonun
    iniş sekansı (SEARCH_LOOP_EXIT_WP'den itibaren) korunur, ama yük bırakma
    yerine orbit öğeleri eklenir.

    ArduPilot, misyon öğesi seq=0'ı ne yüklenirse yüklensin FC'nin kayıtlı
    HOME konumuyla değiştirir (doğrulanmış davranış) — bu yüzden seq 0'a bir
    HOME dolgu öğesi konur, gerçek navigasyon (orbit + iniş) seq 1'den başlar
    ve set_current_mission_item(1) ile FC doğrudan oraya yönlendirilir.
    """
    print("[TEST] Mevcut misyon indiriliyor (iniş WP'lerini almak için)...")
    existing = list(await drone.mission_raw.download_mission())
    print(f"[TEST] {len(existing)} öğe indirildi")

    landing_items = existing[config.SEARCH_LOOP_EXIT_WP:]
    print(f"[TEST] İniş sekansı: WP {config.SEARCH_LOOP_EXIT_WP}'den itibaren "
          f"{len(landing_items)} öğe alındı")

    approach_alt_m = existing[config.SEARCH_START_WP].z
    approach_frame = existing[config.SEARCH_START_WP].frame
    orbit_items = _build_orbit_items(targets, approach_alt_m, approach_frame)
    print(f"[TEST] {len(orbit_items)} orbit öğesi oluşturuldu | yaklaşma irtifası={approach_alt_m:.1f}m")

    home_placeholder = mission._make_mission_item(0, config.CMD_NAV_WAYPOINT, frame=3)
    new_mission = [home_placeholder] + orbit_items + landing_items
    resequenced = [
        RawMissionItem(
            i,
            item.frame, item.command,
            1 if i == 0 else 0,
            item.autocontinue,
            item.param1, item.param2, item.param3, item.param4,
            item.x, item.y, item.z,
            item.mission_type,
        )
        for i, item in enumerate(new_mission)
    ]
    print(f"[TEST] Yeni misyon: {len(resequenced)} öğe "
          f"(1 home dolgu + {len(orbit_items)} orbit + {len(landing_items)} iniş)")

    await drone.mission_raw.upload_mission(resequenced)
    print("[TEST] ✓ upload_mission() tamamlandı")

    await drone.mission_raw.set_current_mission_item(1)
    print("[TEST] ✓ set_current_mission_item(1) — seq 0 (home) atlandı, ilk orbit hedefi aktif")

    await drone.mission_raw.start_mission()
    print("[TEST] ✓ start_mission() — orbit sekansı başladı (başa sıçrama yok)")


# ==================== KARE TESPİTİ (WFB video pipeline üzerinden) ====================

def start_video_pipeline():
    """main.py ile AYNI: monitor mode + WFB-ng + rpicam-vid + OpenCV thread."""
    iface = pipeline.setup_monitor_mode()
    if not iface:
        print("[TEST] Monitor mode kurulamadı — çıkılıyor")
        sys.exit(1)

    if not pipeline.start_pipeline(iface):
        print("[TEST] Pipeline başlatılamadı — çıkılıyor")
        sys.exit(1)

    print("[TEST] OpenCV thread başlatılıyor...")
    opencv_thread = threading.Thread(
        target=vision.opencv_processing_thread,
        args=(state.target_queue,),
        daemon=True,
    )
    opencv_thread.start()
    print(f"[TEST] OpenCV thread başladı (TID={opencv_thread.ident})")

    # Bu testte WP'ye bağlı aktivasyon yok — tespit baştan itibaren aktif.
    state.detection_active.set()
    print("[TEST] ✓ state.detection_active SET — tespit baştan itibaren aktif")


async def collect_two_targets():
    """
    state.target_queue'dan (vision thread'in doldurduğu) mavi/kırmızı
    piksel tespitlerini okuyup, o anki telemetriyle gerçek zamanlı GPS
    konumuna çevirir. Release point/balistik hesap YOK — üstünde
    dönüleceği için hedefin kendi GPS konumu yeterli.
    Görüntüyü artık burada DEĞİL, yer istasyonunda WFB üzerinden izlersiniz.
    """
    targets = []
    seen_colors = set()
    print("[TEST] Mavi ve kırmızı kareleri kameraya okutun — WFB yayınından izleyin.")

    while len(targets) < 2:
        try:
            item = state.target_queue.get(timeout=0.5)
        except Empty:
            await asyncio.sleep(0.05)
            continue

        color = item["color"]
        if color in seen_colors:
            continue

        with state.telemetry_lock:
            tel = state.current_telemetry.copy()

        if None in tel.values():
            print(f"[TEST] {color.upper()} görüldü ama GPS/telemetri henüz hazır değil — "
                  f"hedef geri kuyruğa alınıyor")
            state.target_queue.put(item)
            await asyncio.sleep(1)
            continue

        gps_result = geo.pixel_to_gps(
            tel["lat"], tel["lon"], tel["alt"], tel["yaw"], tel["roll"], tel["pitch"], item["cx"], item["cy"],
        )
        if gps_result is None:
            print(f"[TEST] {color.upper()} REDDEDİLDİ (roll={tel['roll']:.1f}°) — "
                  f"hedef geri kuyruğa alınıyor")
            state.target_queue.put(item)
            await asyncio.sleep(0.5)
            continue
        target_lat, target_lon = gps_result
        seen_colors.add(color)
        targets.append({
            "color": color, "lat": target_lat, "lon": target_lon, "alt": tel["alt"],
        })
        print(f"[TEST] ✓ {color.upper()} okundu → "
              f"hedef=({target_lat:.6f},{target_lon:.6f}) | {len(targets)}/2 hedef")

    return targets


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

    # Telemetri arka planda akmaya başlar
    asyncio.create_task(mission.telemetry_task(drone))
    asyncio.create_task(mission.attitude_task(drone))

    start_video_pipeline()

    try:
        targets = await collect_two_targets()

        print("\n[TEST] 2 hedef toplandı — build_and_start_orbit_mission() çağrılıyor...\n")
        await build_and_start_orbit_mission(drone, targets)

        new_mission = list(await drone.mission_raw.download_mission())
        print_mission(new_mission, "YENİ YÜKLENEN MİSYON (orbit + iniş) — test sonrası")

        print("[TEST] mission_progress izleniyor — WP'nin 0'dan itibaren "
              "ilerlediğini (başa SIÇRAMADIĞINI) doğrulayın. Ctrl+C ile çıkın.\n")
        async for progress in drone.mission_raw.mission_progress():
            print(f"[TEST] aktif WP {progress.current}/{progress.total}")

    finally:
        print("\n[TEST] Temizleniyor...")
        pipeline.stop_pipeline()


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

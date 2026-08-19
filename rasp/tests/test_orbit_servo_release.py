#!/usr/bin/env python3
"""
LOITER + SERVO AÇMA TESTİ (birleşik) — uçuş gerektirmez, elle çalıştırılır.
test_mission_orbit_swap.py ile AYNI temel: WFB video pipeline + kare tespiti +
mission_raw swap ile hedef üstünde LOITER. TEK FARK: test_servo_color_trigger.py
gibi ayrı bir pigpio servo YOK — kare GÖRÜLÜP GPS'e çevrildiği anda (mission.py'nin
gerçek drop_trigger_task'ıyla AYNI yöntemle) FC'ye DO_SET_SERVO gönderilir; servolar
GERÇEKTEN Pixhawk'a bağlı (AUX1/AUX2), pigpio/RPi GPIO devrede DEĞİL.

Akış:
  1. Mavi/kırmızı kare görülüp GPS konumuna çevrildiği anda:
     a. İlgili servo kanalına DO_SET_SERVO gönderilir (RELEASE → 0.5s → NEUTRAL,
        mission.py'deki gerçek drop tetikleme paterniyle birebir aynı)
     b. Hedef, orbit listesine eklenir
  2. 2 hedef toplanınca build_and_start_orbit_mission() çağrılır: her hedefin
     üstüne config.ORBIT_ALTITUDE_M (60m, SABİT — anlık uçuş irtifasından
     BAĞIMSIZ) irtifasında gidilip ORBIT_TURNS tur ORBIT_RADIUS_M yarıçapında
     dönülür, ardından mevcut iniş sekansına devam edilir.

Servo kanalları (config.py, main.py ile AYNI):
  mavi hedef  → SERVO_KIRMIZI_FC_NO (9)  → AUX OUT 1
  kırmızı hedef → SERVO_MAVI_FC_NO (10)  → AUX OUT 2

Çalıştırma:
    cd rasp/tests
    python3 test_orbit_servo_release.py [--conn serial:///dev/ttyAMA3:921600]

Notlar:
  - Gerçek Pi üzerinde çalıştırılmalı (kamera + WiFi adaptörü gerekiyor).
  - Servolar FİZİKSEL olarak Pixhawk AUX1/AUX2'ye bağlı olmalı — pigpio/GPIO YOK.
  - FC'de SERVO9_FUNCTION ve SERVO10_FUNCTION = 0 (Disabled) olmalı, aksi halde
    DO_SET_SERVO görmezden gelinebilir.
  - GCS'ten önceden gerçek görev planı FC'ye yüklenmiş olmalı —
    config.SEARCH_LOOP_EXIT_WP bu plana göre ayarlanmış olmalı.
  - start_mission() çağrılacağı için, drone arm'lıysa gerçekten orbit+iniş
    sekansına geçmeye çalışır. Sadece davranışı görmek istiyorsanız DISARM bırakın.
"""
import argparse
import asyncio
import os
import sys
import threading
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
import mission  # _make_mission_item ve _make_servo_command yardımcı fonksiyonları için


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


# ==================== SERVO TETİKLEME (FC üzerinden, gerçek drop paterniyle) ====================

async def trigger_release_servo(drone, color):
    """
    mission.drop_trigger_task ile AYNI yöntem: DO_SET_SERVO RELEASE → 0.5s → NEUTRAL.
    Şartname: mavi hedefe kırmızı boyalı yük, kırmızı hedefe mavi boyalı yük.
    """
    servo_no = config.SERVO_KIRMIZI_FC_NO if color == "mavi" else config.SERVO_MAVI_FC_NO
    yuk = "KIRMIZI" if color == "mavi" else "MAVİ"
    print(f"[TEST] *** {color.upper()} kare GÖRÜLDÜ → {yuk} servo (kanal={servo_no}) AÇILIYOR ***")
    release_pwm, neutral_pwm = mission._servo_pwm_for(servo_no)
    await drone.mavlink_direct.send_message(mission._make_servo_command(servo_no, release_pwm))
    await asyncio.sleep(0.5)
    await drone.mavlink_direct.send_message(mission._make_servo_command(servo_no, neutral_pwm))
    print(f"[TEST] ✓ {yuk} servo NEUTRAL'e döndü (kanal={servo_no})")


# ==================== ORBIT MİSYONU (sabit irtifada) ====================

def _build_orbit_items(targets):
    """
    Her hedef için: NAV_WAYPOINT ile hedefe git, ardından NAV_LOITER_TURNS ile
    ORBIT_TURNS tur boyunca ORBIT_RADIUS_M yarıçapında hedef üstünde dön.
    İrtifa SABİT config.ORBIT_ALTITUDE_M — tespit anındaki uçuş irtifasından
    BAĞIMSIZ (test_mission_orbit_swap.py'den FARK).
    """
    items = []
    for i, t in enumerate(targets):
        print(f"[TEST] === Orbit öğeleri: Hedef {i+1} {t['color'].upper()} "
              f"({t['lat']:.6f},{t['lon']:.6f}) alt={config.ORBIT_ALTITUDE_M:.1f}m (sabit) ===")

        items.append(mission._make_mission_item(
            0, config.CMD_NAV_WAYPOINT, param2=15.0,
            lat=t["lat"], lon=t["lon"], alt=config.ORBIT_ALTITUDE_M, frame=3,
        ))
        items.append(mission._make_mission_item(
            0, config.CMD_NAV_LOITER_TURNS,
            param1=config.ORBIT_TURNS, param3=config.ORBIT_RADIUS_M,
            lat=t["lat"], lon=t["lon"], alt=config.ORBIT_ALTITUDE_M, frame=3,
        ))
    return items


async def build_and_start_orbit_mission(drone, targets):
    """
    mission.build_and_start_drop_mission ile aynı iskelet: eski misyonun
    iniş sekansı (SEARCH_LOOP_EXIT_WP'den itibaren) korunur, ama yük bırakma
    yerine sabit irtifalı orbit öğeleri eklenir.

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

    orbit_items = _build_orbit_items(targets)
    print(f"[TEST] {len(orbit_items)} orbit öğesi oluşturuldu (irtifa={config.ORBIT_ALTITUDE_M}m)")

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
    """main.py ile AYNI: rpicam-vid + WFB-ng + OpenCV thread. WFB dongle
    bulunamazsa artık çıkılmıyor — main.py'deki 2026-08-19 düzeltmesiyle
    aynı: rpicam/OpenCV hemen başlar, dongle arka planda aranır."""
    pipeline.start_pipeline()

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


async def collect_two_targets(drone):
    """
    state.target_queue'dan (vision thread'in doldurduğu) mavi/kırmızı piksel
    tespitlerini okuyup, o anki telemetriyle gerçek zamanlı GPS konumuna çevirir.
    Her hedef GPS'e çevrildiği anda İLGİLİ SERVO FC ÜZERİNDEN TETİKLENİR
    (trigger_release_servo) — orbit'e geçmeden ÖNCE, kare görülür görülmez.
    """
    targets = []
    seen_colors = set()
    print("[TEST] Mavi ve kırmızı kareleri kameraya okutun — WFB yayınından izleyin.")
    print("[TEST] Kare GÖRÜLÜR GÖRÜLMEZ ilgili servo FC üzerinden tetiklenecek.")

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

        # Kare görülür görülmez servo tetiklenir — orbit misyonu yüklenmeden önce.
        await trigger_release_servo(drone, color)

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
    asyncio.create_task(mission.speed_track_task(drone))

    start_video_pipeline()

    try:
        targets = await collect_two_targets(drone)

        print("\n[TEST] 2 hedef toplandı (servolar tetiklendi) — "
              "build_and_start_orbit_mission() çağrılıyor...\n")
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

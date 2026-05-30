"""
MAVSDK görev yönetimi:
  - Telemetri ve yaw akışları
  - Hız yönetimi (DO_CHANGE_SPEED)
  - Arama döngüsünden çıkış (DO_JUMP exit)
  - Drop misyonu oluşturma ve yükleme (mission_raw)
  - GPIO drop tetikleme (USE_FC_SERVO=False durumu)
"""
import asyncio
import json
import time
from queue import Empty
from mavsdk.mission_raw import MissionItem as RawMissionItem
from mavsdk.mavlink_direct import MavlinkMessage
import config
import state
import geo
import servo


# ==================== TELEMETRİ ====================

async def telemetry_task(drone):
    print("[TELEMETRY] Konum akışı başlatıldı")
    count = 0
    async for position in drone.telemetry.position():
        with state.telemetry_lock:
            state.current_telemetry["lat"] = position.latitude_deg
            state.current_telemetry["lon"] = position.longitude_deg
            state.current_telemetry["alt"] = position.relative_altitude_m
        count += 1
        if count % 50 == 0:
            print(f"[TELEMETRY] #{count}: lat={position.latitude_deg:.6f} "
                  f"lon={position.longitude_deg:.6f} alt={position.relative_altitude_m:.1f}m")


async def attitude_task(drone):
    print("[ATTITUDE] Yaw akışı başlatıldı")
    count = 0
    async for attitude in drone.telemetry.attitude_euler():
        with state.telemetry_lock:
            state.current_telemetry["yaw"] = attitude.yaw_deg
        count += 1
        if count % 50 == 0:
            print(f"[ATTITUDE] #{count}: yaw={attitude.yaw_deg:.1f}°")


# ==================== GPIO DROP TETİKLEME ====================

async def drop_trigger_task(drone, release_points):
    """
    USE_FC_SERVO=False durumunda: GPS mesafesiyle GPIO servo tetikler.
    Her 0.1 saniyede mesafeleri kontrol eder.
    """
    print(f"[DROP] GPIO trigger başlatıldı — {len(release_points)} release point izleniyor")
    for i, rp in enumerate(release_points):
        print(f"[DROP]   Hedef {i+1}: {rp['color'].upper()} → "
              f"({rp['lat']:.6f},{rp['lon']:.6f}) alt={rp['alt']:.1f}m")

    check_count = 0
    async for pos in drone.telemetry.position():
        check_count += 1
        for rp in release_points:
            if rp["dropped"]:
                continue
            dist = geo.haversine(pos.latitude_deg, pos.longitude_deg, rp["lat"], rp["lon"])
            if check_count % 10 == 0:
                print(f"[DROP] {rp['color'].upper()} mesafe: {dist:.1f}m "
                      f"(eşik: {config.DROP_TRIGGER_RADIUS_M}m)")
            if dist < config.DROP_TRIGGER_RADIUS_M:
                print(f"[DROP] *** {rp['color'].upper()} TETİKLENİYOR! "
                      f"mesafe={dist:.1f}m < {config.DROP_TRIGGER_RADIUS_M}m ***")
                await servo.drop_payload(rp["color"])
                rp["dropped"] = True
                print(f"[DROP] {rp['color'].upper()} bırakıldı ✓ "
                      f"(kalan: {sum(1 for r in release_points if not r['dropped'])})")

        if all(rp["dropped"] for rp in release_points):
            print("[DROP] ✓ Tüm yükler bırakıldı — drop_trigger_task sonlanıyor")
            break
        await asyncio.sleep(0.1)


# ==================== HIZ YÖNETİMİ ====================

def _make_speed_command(speed_ms):
    """COMMAND_LONG / DO_CHANGE_SPEED MAVLink mesajı oluşturur."""
    return MavlinkMessage(
        message_name="COMMAND_LONG",
        system_id=255,
        component_id=190,
        target_system_id=1,
        target_component_id=1,
        fields_json=json.dumps({
            "target_system": 1,
            "target_component": 1,
            "command": config.CMD_DO_CHANGE_SPEED,
            "confirmation": 0,
            "param1": 1.0,       # groundspeed
            "param2": speed_ms,
            "param3": -1.0,      # throttle değişikliği yok
            "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0,
        })
    )


async def speed_management_task(drone):
    """
    Misyon ilerlemesini izler; SEARCH_START_WP'ye gelince
    mavlink_direct üzerinden DO_CHANGE_SPEED gönderir.
    """
    print(f"[SPEED] WP takibi başladı — WP {config.SEARCH_START_WP}'de "
          f"{config.SEARCH_SPEED_MS}m/s'ye düşülecek")
    async for progress in drone.mission_raw.mission_progress():
        print(f"[SPEED] Misyon ilerleme: WP {progress.current}/{progress.total}")
        if progress.current == config.SEARCH_START_WP:
            print(f"[SPEED] WP {config.SEARCH_START_WP} ulaşıldı → "
                  f"DO_CHANGE_SPEED={config.SEARCH_SPEED_MS}m/s gönderiliyor...")
            try:
                await drone.mavlink_direct.send_message(_make_speed_command(config.SEARCH_SPEED_MS))
                print(f"[SPEED] ✓ Hız başarıyla {config.SEARCH_SPEED_MS}m/s'ye düşürüldü")
            except Exception as e:
                print(f"[SPEED] HATA: Hız komutu gönderilemedi: {e}")
            print("[SPEED] speed_management_task görevi tamamlandı, sonlanıyor")
            break


# ==================== SEARCH LOOP ÇIKIŞI ====================

async def exit_search_loop(drone):
    """DO_JUMP döngüsünden çık: misyon item'ını SEARCH_LOOP_EXIT_WP'ye al."""
    print(f"[MISSION] DO_JUMP döngüsünden çıkılıyor → "
          f"set_current_mission_item({config.SEARCH_LOOP_EXIT_WP})")
    try:
        await drone.mission_raw.set_current_mission_item(config.SEARCH_LOOP_EXIT_WP)
        print(f"[MISSION] ✓ Aktif WP artık {config.SEARCH_LOOP_EXIT_WP}")
    except Exception as e:
        print(f"[MISSION] ⚠ set_current_mission_item HATA: {e}")
        print(f"[MISSION] ⚠ Devam ediliyor — FC'de misyon yüklü mü? "
              f"SEARCH_LOOP_EXIT_WP={config.SEARCH_LOOP_EXIT_WP} doğru mu?")


# ==================== DROP MİSYONU ====================

def _make_mission_item(seq, command, param1=0.0, param2=0.0, param3=0.0, param4=0.0,
                       lat=0.0, lon=0.0, alt=0.0, frame=2):
    """
    mission_raw.MissionItem oluşturma yardımcısı.
    frame=2 → MAV_FRAME_MISSION (komutlar için)
    frame=3 → MAV_FRAME_GLOBAL_RELATIVE_ALT (GPS waypoint'ler için)
    """
    return RawMissionItem(
        seq, frame, command, 1 if seq == 0 else 0, 1,
        param1, param2, param3, param4,
        int(lat * 1e7), int(lon * 1e7), alt, 0,
    )


def _build_drop_items(release_points):
    """
    Drop waypoint'lerini oluşturur.
    Seq geçici 0 — insert sonrası yeniden numaralandırılır.
    """
    items = []
    for i, rp in enumerate(release_points):
        servo_no = config.SERVO_KIRMIZI_FC_NO if rp["color"] == "mavi" else config.SERVO_MAVI_FC_NO
        print(f"[MISSION] === Drop öğeleri: Hedef {i+1} {rp['color'].upper()} "
              f"({rp['lat']:.6f},{rp['lon']:.6f}) alt={rp['alt']:.1f}m servo={servo_no} ===")

        items.append(_make_mission_item(
            0, config.CMD_DO_CHANGE_SPEED,
            param1=1.0, param2=config.DROP_SPEED_MS, param3=-1.0,
        ))
        items.append(_make_mission_item(
            0, config.CMD_NAV_WAYPOINT, param2=15.0,
            lat=rp["lat"], lon=rp["lon"], alt=rp["alt"], frame=3,
        ))

        if config.USE_FC_SERVO:
            items.append(_make_mission_item(
                0, config.CMD_CONDITION_DIST,
                param1=float(config.DROP_TRIGGER_RADIUS_M),
            ))
            items.append(_make_mission_item(
                0, config.CMD_DO_SET_SERVO,
                param1=float(servo_no), param2=float(config.PWM_RELEASE),
            ))
            items.append(_make_mission_item(
                0, config.CMD_DO_SET_SERVO,
                param1=float(servo_no), param2=float(config.PWM_NEUTRAL),
            ))

        items.append(_make_mission_item(
            0, config.CMD_DO_CHANGE_SPEED,
            param1=1.0, param2=config.DRONE_SPEED_MS, param3=-1.0,
        ))
    return items


async def build_and_insert_drop_mission(drone, release_points):
    """
    FC'deki mevcut misyonu indirir, drop waypoint'lerini SEARCH_LOOP_EXIT_WP'den
    sonraya ekler, kalkış ve iniş sekansı korunur.
    """
    print(f"[MISSION] Mevcut misyon indiriliyor...")
    existing = list(await drone.mission_raw.download_mission())
    print(f"[MISSION] {len(existing)} öğe indirildi")

    drop_items = _build_drop_items(release_points)
    print(f"[MISSION] {len(drop_items)} drop öğesi oluşturuldu | USE_FC_SERVO={config.USE_FC_SERVO}")

    insert_at = config.SEARCH_LOOP_EXIT_WP + 1
    merged    = existing[:insert_at] + drop_items + existing[insert_at:]

    # Tüm seq numaralarını sıfırdan yeniden düzenle
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
        for i, item in enumerate(merged)
    ]
    print(f"[MISSION] Yeni misyon: {len(resequenced)} öğe "
          f"({len(existing)} mevcut + {len(drop_items)} drop)")

    await drone.mission_raw.upload_mission(resequenced)
    print("[MISSION] upload_mission() tamamlandı")

    first_drop = insert_at
    await drone.mission_raw.set_current_mission_item(first_drop)
    print(f"[MISSION] ✓ Drop başlıyor WP {first_drop}'den — iniş sekansı korundu")

    await drone.mission_raw.start_mission()

    if not config.USE_FC_SERVO:
        print("[MISSION] USE_FC_SERVO=False → GPIO drop_trigger_task başlatılıyor")
        asyncio.create_task(drop_trigger_task(drone, release_points))
    else:
        print("[MISSION] USE_FC_SERVO=True → servo FC misyon öğesiyle tetiklenecek")


# ==================== WP TAKİBİ ====================

async def waypoint_tracking_task(drone):
    """Aktif WP index'ini state.current_wp'de tutar — overlay için."""
    async for progress in drone.mission_raw.mission_progress():
        state.current_wp["index"] = progress.current
        state.current_wp["total"] = progress.total


# ==================== TESPİT AKTİVASYONU ====================

async def detection_activation_task(drone):
    """DETECTION_ACTIVE_WP'ye gelince tespiti aktif eder."""
    print(f"[DETECTION] Tespit aktivasyonu bekleniyor — WP {config.DETECTION_ACTIVE_WP}'de aktif olacak")
    async for progress in drone.mission_raw.mission_progress():
        if progress.current >= config.DETECTION_ACTIVE_WP:
            state.detection_active.set()
            print(f"[DETECTION] ✓ Tespit AKTİF — WP {progress.current}")
            break


# ==================== ANA GÖREV ====================

async def mission_task(drone, queue):
    print("[MISSION] mission_task başladı — hedef tespiti bekleniyor")
    release_points       = []
    first_detection_time = None

    while len(release_points) < 2:
        try:
            target = queue.get(timeout=0.5)
            print(f"[MISSION] Kuyruktan alındı: {target['color'].upper()} "
                  f"piksel=({target['cx']},{target['cy']})")
        except Empty:
            if len(release_points) == 1 and first_detection_time:
                elapsed   = time.time() - first_detection_time
                remaining = config.SINGLE_TARGET_TIMEOUT_SEC - elapsed
                if elapsed > config.SINGLE_TARGET_TIMEOUT_SEC:
                    print(f"[MISSION] ⚠ Timeout ({config.SINGLE_TARGET_TIMEOUT_SEC}s) doldu — "
                          f"tek hedefle devam ediliyor")
                    break
                if int(remaining) % 5 == 0:
                    print(f"[MISSION] İkinci hedef bekleniyor... (kalan≈{remaining:.0f}s)")
            await asyncio.sleep(0.1)
            continue

        with state.telemetry_lock:
            tel = state.current_telemetry.copy()

        if None in tel.values():
            print(f"[MISSION] Telemetri eksik {tel} — hedef geri kuyruğa alınıyor")
            queue.put(target)
            await asyncio.sleep(1)
            continue

        color = target["color"]
        if any(rp["color"] == color for rp in release_points):
            print(f"[MISSION] {color.upper()} zaten işlendi — tekrar atlandı")
            continue

        print(f"[MISSION] {color.upper()} işleniyor | telemetri: {tel}")
        target_lat, target_lon = geo.pixel_to_gps(
            tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
            target["cx"], target["cy"],
        )

        release_lat, release_lon = geo.calculate_drop_point(
            target_lat, target_lon, tel["alt"], config.DRONE_SPEED_MS, tel["yaw"],
        )

        release_points.append({
            "color":   color,
            "lat":     release_lat,
            "lon":     release_lon,
            "alt":     tel["alt"],
            "dropped": False,
        })
        if first_detection_time is None:
            first_detection_time = time.time()

        print(f"[MISSION] ✓ {color.upper()} release point kaydedildi: "
              f"({release_lat:.6f},{release_lon:.6f}) — "
              f"{len(release_points)}/2 hedef toplandı")

    print(f"[MISSION] Tarama tamamlandı ({len(release_points)} hedef). "
          f"{config.SCAN_EXIT_DELAY_SEC}s bekleniyor...")
    await asyncio.sleep(config.SCAN_EXIT_DELAY_SEC)

    print("[MISSION] Arama döngüsünden çıkılıyor...")
    await exit_search_loop(drone)
    print("[MISSION] 2s bekleniyor (FC'nin item değişikliğini işlemesi için)...")
    await asyncio.sleep(2)

    await build_and_insert_drop_mission(drone, release_points)

    if not config.USE_FC_SERVO:
        print("[MISSION] USE_FC_SERVO=False → GPIO drop_trigger_task başlatılıyor")
        asyncio.create_task(drop_trigger_task(drone, release_points))
    else:
        print("[MISSION] USE_FC_SERVO=True → servo FC misyon öğesiyle tetiklenecek, ek task yok")

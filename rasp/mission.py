"""
MAVSDK görev yönetimi:
  - Telemetri, yaw ve hız akışları
  - Hız yönetimi (DO_CHANGE_SPEED)
  - Drop misyonu oluşturma ve yükleme (mission_raw) — yalnızca YÖNLENDİRME
  - Canlı balistik drop tetikleme (drop_trigger_task): her telemetri tik'inde
    güncel hız/irtifa ile calculate_drop_point() yeniden hesaplanır, doğru
    anda FC'ye DO_SET_SERVO gönderilir (GPIO servo yolu kaldırıldı, 2026-08-16).
"""
import asyncio
import json
import logging
import math
import time
from queue import Empty
from mavsdk.mission_raw import MissionItem as RawMissionItem
from mavsdk.mavlink_direct import MavlinkMessage
import config
import state
import geo

log = logging.getLogger("mission")


# ==================== FC BAĞLANTI DURUMU ====================

async def _fc_reconnect_loop(drone):
    """
    Bağlantı koptuğunda (ör. kalibrasyon sonrası FC reboot atılması — USB
    CDC-ACM cihazı kısa süreliğine kaybolup by-id sembolik linki aynı yoldan
    geri gelir) periyodik olarak drone.connect()'i tekrar çağırır.

    Sadece TEK bir döngü aynı anda çalışsın diye state.fc_reconnecting ile
    korunur — art arda gelen connection_state() olayları ikinci bir döngü
    başlatmasın.
    """
    if state.fc_reconnecting:
        return
    state.fc_reconnecting = True
    attempt = 0
    try:
        while not state.fc_connected and not state.shutdown_requested.is_set():
            attempt += 1
            log.info(f"[FC] Yeniden bağlanma denemesi #{attempt} "
                     f"(serial://{config.FC_PORT}:{config.FC_BAUDRATE})...")
            try:
                await drone.connect(system_address=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}")
            except Exception as e:
                log.info(f"[FC] Yeniden bağlanma denemesi HATA: {e}")
            await asyncio.sleep(config.FC_RECONNECT_INTERVAL_SEC)
        if state.fc_connected:
            log.info(f"[FC] ✓ Yeniden bağlantı başarılı ({attempt} deneme sonrası)")
    finally:
        state.fc_reconnecting = False


async def fc_connection_task(drone):
    """FC bağlantısını surekli izler (HUD icin) — ilk baglanti sonrası kopma/
    geri gelme de burada yakalanir, main.py'deki tek seferlik kontrolden farklı.
    Bağlantı koparsa _fc_reconnect_loop ile otomatik yeniden bağlanma dener."""
    log.info("[FC] Bağlantı durumu izleyici başlatıldı")
    async for conn in drone.core.connection_state():
        was_connected = state.fc_connected
        state.fc_connected = conn.is_connected
        if conn.is_connected and not was_connected:
            log.info("[FC] ✓ Bağlantı kuruldu")
        elif not conn.is_connected and was_connected:
            log.info("[FC] ⚠ Bağlantı KOPTU — yeniden bağlanma denenecek")
            asyncio.create_task(_fc_reconnect_loop(drone))


# ==================== TELEMETRİ ====================

async def telemetry_task(drone):
    log.info("[TELEMETRY] Konum akışı başlatıldı")
    count = 0
    async for position in drone.telemetry.position():
        now = time.monotonic()
        with state.telemetry_lock:
            state.current_telemetry["lat"] = position.latitude_deg
            state.current_telemetry["lon"] = position.longitude_deg
            state.current_telemetry["alt"] = position.relative_altitude_m
            state.position_history.append((
                now, position.latitude_deg, position.longitude_deg,
                position.relative_altitude_m,
            ))
        count += 1
        if count % 50 == 0:
            log.info(f"[TELEMETRY] #{count}: lat={position.latitude_deg:.6f} "
                  f"lon={position.longitude_deg:.6f} alt={position.relative_altitude_m:.1f}m")


async def attitude_task(drone):
    """Yaw + roll + pitch akışı — roll/pitch, pixel_to_gps()'in tespit anındaki
    gerçek uçak duruşunu (özellikle bank/roll) hesaba katabilmesi için gerekli
    (bkz. geo.pixel_to_gps, 2026-08-08: roll bankında tespit hatası bulundu/düzeltildi)."""
    log.info("[ATTITUDE] Yaw/roll/pitch akışı başlatıldı")
    count = 0
    async for attitude in drone.telemetry.attitude_euler():
        now = time.monotonic()
        with state.telemetry_lock:
            state.current_telemetry["yaw"]   = attitude.yaw_deg
            state.current_telemetry["roll"]  = attitude.roll_deg
            state.current_telemetry["pitch"] = attitude.pitch_deg
            state.attitude_history.append((
                now, attitude.yaw_deg, attitude.roll_deg, attitude.pitch_deg,
            ))
        count += 1
        if count % 50 == 0:
            log.info(f"[ATTITUDE] #{count}: yaw={attitude.yaw_deg:.1f}° "
                     f"roll={attitude.roll_deg:.1f}° pitch={attitude.pitch_deg:.1f}°")


async def speed_track_task(drone):
    """
    Yer hızını (ground speed) sürekli günceller — drop_trigger_task'ın her
    tik'te güncel hızla balistik drop noktasını yeniden hesaplayabilmesi için.
    """
    log.info("[SPEED_TRACK] Yer hızı akışı başlatıldı")
    async for vel in drone.telemetry.velocity_ned():
        now = time.monotonic()
        speed = math.hypot(vel.north_m_s, vel.east_m_s)
        with state.telemetry_lock:
            state.current_telemetry["speed"] = speed
            state.current_telemetry["vel_n"] = vel.north_m_s
            state.current_telemetry["vel_e"] = vel.east_m_s
            state.speed_history.append((now, speed, vel.north_m_s, vel.east_m_s))


async def gps_health_task(drone):
    """
    Uydu sayısı/fix tipi + EKF-eşdeğeri (Health) durumunu izler ve loglar
    (checklist "GPS ve heading" — 2026-08-16). Tek başına HDOP/uydu sayısı
    yüzlerce metrelik hatayı açıklamaz (bkz. commit notu) ama fix_type
    düşerse (ör. RTK_FIXED -> FIX_3D) veya global_position_ok False olursa
    bu, o anki tespitlerin şüpheli sayılması için güçlü bir sinyaldir.
    """
    log.info("[GPS_HEALTH] Uydu/EKF izleyici başlatıldı")

    async def _gps_info_loop():
        prev_fix = None
        async for info in drone.telemetry.gps_info():
            with state.telemetry_lock:
                state.gps_health["num_satellites"] = info.num_satellites
                state.gps_health["fix_type"]       = str(info.fix_type)
            if info.fix_type != prev_fix:
                log.info(f"[GPS_HEALTH] fix_type değişti: {prev_fix} -> {info.fix_type} "
                         f"(uydu={info.num_satellites})")
                prev_fix = info.fix_type

    async def _health_loop():
        prev = None
        async for h in drone.telemetry.health():
            snap = (h.is_global_position_ok, h.is_local_position_ok, h.is_home_position_ok)
            with state.telemetry_lock:
                state.ekf_health["global_position_ok"] = h.is_global_position_ok
                state.ekf_health["local_position_ok"]  = h.is_local_position_ok
                state.ekf_health["home_position_ok"]   = h.is_home_position_ok
            if snap != prev:
                log.info(f"[GPS_HEALTH] health değişti: global_ok={h.is_global_position_ok} "
                         f"local_ok={h.is_local_position_ok} home_ok={h.is_home_position_ok}")
                prev = snap

    await asyncio.gather(_gps_info_loop(), _health_loop())


# ==================== CANLI BALİSTİK DROP TETİKLEME ====================

async def _trigger_release(drone, rp, release_points, aircraft_lat=None, aircraft_lon=None):
    """FC'ye DO_SET_SERVO gönderip yükü bırakır (şartname eşlemesi: MAVİ hedefe
    KIRMIZI yük, KIRMIZI hedefe MAVİ yük — çapraz eşleme KASITLI)."""
    # NOT (2026-08-16): GPIO servo yolu tamamen kaldırıldı — RPi GPIO'ya
    # hiçbir servo bağlanmayacak, tek yol FC (DO_SET_SERVO). Eskiden 0.5sn
    # sonra otomatik nötr PWM'e dönülüyordu — tek seferlik bırakma için
    # gereksiz, kasıtlı olarak bırak konumunda BIRAKILIYOR.
    servo_no = config.SERVO_KIRMIZI_FC_NO if rp["color"] == "mavi" else config.SERVO_MAVI_FC_NO
    release_pwm, _neutral_pwm = _servo_pwm_for(servo_no)
    await drone.mavlink_direct.send_message(_make_servo_command(servo_no, release_pwm))
    log.info(f"[DROP] ✓ FC'ye DO_SET_SERVO gönderildi: kanal={servo_no} "
             f"pwm={release_pwm} (nötre dönülmüyor, açık kalıyor)")
    rp["dropped"] = True
    # HUD'da "SERVO AÇILDI" kalıcı bilgisi için — komutun gönderildiği ANDAKİ
    # uçak konumu (2026-08-20, bkz. vision.py overlay, state.servo_events).
    state.servo_events[rp["color"]] = {
        "channel": servo_no,
        "lat": aircraft_lat,
        "lon": aircraft_lon,
        "ts": time.monotonic(),
    }
    log.info(f"[DROP] {rp['color'].upper()} bırakıldı ✓ "
          f"(kalan: {sum(1 for r in release_points if not r['dropped'])})")


async def drop_trigger_task(drone, release_points):
    """
    Her pozisyon tik'inde GÜNCEL hız/irtifa ile calculate_drop_point()'i
    yeniden hesaplar (hedefin kendisi rp['lat']/rp['lon']'da sabit, balistik
    ofset taze) ve bırakma kararını along/cross-track ayrışımıyla verir
    (2026-08-17 — eski davranış: release noktasına 40m'ye girildiği AN
    bırakılıyordu; bu (a) 40m'ye kadar konum hatasına izin veriyordu
    (şartname 20m dışını 0 puan sayar), (b) rota başka bacaktayken —
    ör. öteki hedefe ya da dönüş WP'sine giderken — noktanın 40m yakınından
    geçilirse YANLIŞ tetikliyordu):

      1. KURMA: release noktasına mesafe DROP_TRIGGER_RADIUS_M içine girince
         o hedef için değerlendirme başlar.
      2. Rota (course, hız vektöründen — rüzgar crab'i nedeniyle yaw DEĞİL)
         ile release noktasına kerteriz ayrışımı: along = ileri kalan mesafe,
         cross = dik sapma.
      3. TETİK: along-track kalan süre DROP_ALONG_TRIGGER_S altına inince VE
         |cross| <= DROP_MAX_CROSS_TRACK_M ise bırak. Cross fazlaysa bırakma
         (log'la) — 0 puanlık atış yerine sonraki geçişi bekle: misyonda aynı
         WP'ye tekrar gelinmez ama uçuş süresi elverdikçe GCS'den yeni geçiş
         komutu verilebilir; kötü atışı erken sabitlemekten iyidir.
      4. Emniyet ağı: nokta değerlendirme penceresindeyken along işaret
         değiştirir (nokta geçilmiş) ve cross uygunsa hemen bırak — akış
         tik'i tam tetik anını kaçırdıysa bir tik gecikmeli telafi.

    Balistik ofset yönü de artık yaw yerine course ile hesaplanıyor: yük
    bırakıldığında uçağın burnu değil, yer hızı vektörü yönünde savrulur.
    """
    log.info(f"[DROP] Canlı balistik trigger başlatıldı — {len(release_points)} hedef izleniyor")
    for i, rp in enumerate(release_points):
        log.info(f"[DROP]   Hedef {i+1}: {rp['color'].upper()} → ({rp['lat']:.6f},{rp['lon']:.6f})")
        rp["armed"] = False
        rp["prev_along"] = None

    check_count = 0
    async for pos in drone.telemetry.position():
        check_count += 1
        with state.telemetry_lock:
            tel = state.current_telemetry.copy()
        if tel["speed"] is None or tel["vel_n"] is None or tel["speed"] < 1.0:
            continue

        course_deg = math.degrees(math.atan2(tel["vel_e"], tel["vel_n"]))

        for rp in release_points:
            if rp["dropped"]:
                continue
            release_lat, release_lon = geo.calculate_drop_point(
                rp["lat"], rp["lon"], pos.relative_altitude_m, tel["speed"], course_deg,
            )
            dist = geo.haversine(pos.latitude_deg, pos.longitude_deg, release_lat, release_lon)

            if not rp["armed"]:
                if dist < config.DROP_TRIGGER_RADIUS_M:
                    rp["armed"] = True
                    log.info(f"[DROP] {rp['color'].upper()} KURULDU (mesafe={dist:.1f}m < "
                             f"{config.DROP_TRIGGER_RADIUS_M}m) — along/cross izleniyor")
                else:
                    if check_count % 10 == 0:
                        log.info(f"[DROP] {rp['color'].upper()} drop noktasına {dist:.1f}m "
                              f"(kurma eşiği {config.DROP_TRIGGER_RADIUS_M}m)")
                    continue

            bearing = geo.bearing_deg(pos.latitude_deg, pos.longitude_deg,
                                      release_lat, release_lon)
            rel_rad = math.radians(geo.wrap180(bearing - course_deg))
            along   = dist * math.cos(rel_rad)   # + : nokta önümüzde
            cross   = abs(dist * math.sin(rel_rad))
            time_to = along / tel["speed"]

            log.info(f"[DROP] {rp['color'].upper()} along={along:.1f}m cross={cross:.1f}m "
                     f"t={time_to:.2f}s (hız={tel['speed']:.1f}m/s alt={pos.relative_altitude_m:.1f}m)")

            passed = rp["prev_along"] is not None and rp["prev_along"] > 0 and along <= 0
            rp["prev_along"] = along

            if cross > config.DROP_MAX_CROSS_TRACK_M:
                if along <= 0:
                    # Noktayı yandan kaçırarak geçtik — bırakmadık, yeniden kur.
                    log.info(f"[DROP] ⚠ {rp['color'].upper()} geçiş ISKA: cross={cross:.1f}m > "
                             f"{config.DROP_MAX_CROSS_TRACK_M}m — bırakılmadı, sonraki geçiş bekleniyor")
                    rp["armed"] = False
                    rp["prev_along"] = None
                continue

            if (0 <= time_to <= config.DROP_ALONG_TRIGGER_S) or passed:
                log.info(f"[DROP] *** {rp['color'].upper()} TETİKLENİYOR! "
                      f"along={along:.1f}m cross={cross:.1f}m t={time_to:.2f}s "
                      f"{'(geçiş telafisi)' if passed else ''} ***")
                await _trigger_release(drone, rp, release_points,
                                        pos.latitude_deg, pos.longitude_deg)

        if all(rp["dropped"] for rp in release_points):
            log.info("[DROP] ✓ Tüm yükler bırakıldı — drop_trigger_task sonlanıyor")
            break


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


def _servo_pwm_for(servo_no):
    """
    Her servo kendi güvenli PWM aralığıyla ayrı tanımlı (sahada doğrulandı).
    Dönüş: (release_pwm, neutral_pwm).
    """
    if servo_no == config.SERVO_MAVI_FC_NO:
        return config.PWM_MAVI_RELEASE, config.PWM_MAVI_NEUTRAL
    return config.PWM_KIRMIZI_RELEASE, config.PWM_KIRMIZI_NEUTRAL


def _make_servo_command(servo_no, pwm):
    """COMMAND_LONG / DO_SET_SERVO MAVLink mesajı oluşturur (canlı balistik drop tetikleme için)."""
    return MavlinkMessage(
        message_name="COMMAND_LONG",
        system_id=255,
        component_id=190,
        target_system_id=1,
        target_component_id=1,
        fields_json=json.dumps({
            "target_system": 1,
            "target_component": 1,
            "command": config.CMD_DO_SET_SERVO,
            "confirmation": 0,
            "param1": float(servo_no),
            "param2": float(pwm),
            "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0,
        })
    )


async def speed_management_task(drone):
    """
    Misyon ilerlemesini izler; SEARCH_START_WP'ye gelince
    mavlink_direct üzerinden DO_CHANGE_SPEED gönderir.
    """
    log.info(f"[SPEED] WP takibi başladı — WP {config.SEARCH_START_WP}'de "
          f"{config.SEARCH_SPEED_MS}m/s'ye düşülecek")
    async for progress in drone.mission_raw.mission_progress():
        log.info(f"[SPEED] Misyon ilerleme: WP {progress.current}/{progress.total}")
        # >= (== değil, 2026-08-17): tek bir progress olayı kaçarsa/atlanırsa
        # (ör. 3→5) hız düşürme HİÇ tetiklenmiyordu ve tarama 15m/s'de
        # yapılıyordu — detection_activation_task'taki mantıkla aynı hizaya
        # getirildi.
        if progress.current >= config.SEARCH_START_WP:
            log.info(f"[SPEED] WP {config.SEARCH_START_WP} ulaşıldı → "
                  f"DO_CHANGE_SPEED={config.SEARCH_SPEED_MS}m/s gönderiliyor...")
            try:
                await drone.mavlink_direct.send_message(_make_speed_command(config.SEARCH_SPEED_MS))
                log.info(f"[SPEED] ✓ Hız başarıyla {config.SEARCH_SPEED_MS}m/s'ye düşürüldü")
            except Exception as e:
                log.info(f"[SPEED] HATA: Hız komutu gönderilemedi: {e}")
            log.info("[SPEED] speed_management_task görevi tamamlandı, sonlanıyor")
            break


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


def _make_return_to_search_entry_item(existing):
    """
    Tarama girişine (SEARCH_START_WP) dönüş waypoint'i.

    Arama döngüsü kod tarafından (2 hedef bulununca) rastgele bir noktada
    kesintiye uğruyor — uçak oradan doğrudan ilk hedefe dönerse sabit kanat
    için dar açılı/öngörülemez bir dönüş olabilir. Bunun yerine önce bilinen,
    sabit bir noktaya (tarama bacağının giriş WP'si) dönülüp yaklaşma oradan
    başlatılır. Seq geçici 0 — insert sonrası yeniden numaralandırılır.
    """
    entry = existing[config.SEARCH_START_WP]
    if entry.x == 0 and entry.y == 0:
        log.info(f"[MISSION] ⚠ UYARI: SEARCH_START_WP={config.SEARCH_START_WP} öğesinin koordinatı "
                 f"(0,0) — bu index yanlış olabilir (command={entry.command}), GCS planını kontrol edin!")
    return RawMissionItem(
        0, entry.frame, config.CMD_NAV_WAYPOINT, 0, 1,
        0.0, 15.0, 0.0, 0.0,
        entry.x, entry.y, entry.z, 0,
    )


def _build_drop_items(release_points, approach_alt_m, approach_frame=3):
    """
    Hedeflere YÖNLENDİRME waypoint'lerini oluşturur (sade NAV_WAYPOINT —
    direkt WP'ye gidilir, servo AÇMA burada değil). Asıl drop tetikleme
    burada YOK — drop_trigger_task her telemetri tik'inde güncel hız/irtifa
    ile balistik hesap yapıp doğru anda tetikliyor (bkz.
    build_and_start_drop_mission), bu sayede tespit anındaki eski/varsayılan
    hıza değil, bırakma anındaki gerçek hıza göre atılıyor.
    Seq geçici 0 — insert sonrası yeniden numaralandırılır.

    NOT (2026-08-20): waypoint çevresindeki DO_CHANGE_SPEED (yavaşla/hızlan)
    çifti kaldırıldı — drop_trigger_task zaten GÜNCEL anlık hıza göre
    balistik hesap yapıyor, mission'ın kendi hız komutuna ihtiyacı yok;
    tek amacı uçağı doğrudan hedef koordinatına götürmek.

    approach_alt_m/approach_frame: waypoint'lerin İRTİFASI ve FRAME'i — GCS
    planındaki SEARCH_START_WP öğesinden (build_and_start_drop_mission'dan
    geliyor, entry.z/entry.frame — _make_return_to_search_entry_item'ın
    kullandığıyla AYNI kaynak), rp["alt"] (tespit anındaki ANLIK telemetri
    irtifası, her zaman relative) DEĞİL. rp["alt"] tek bir telemetri örneği —
    bank/tur sırasında, GPS/baro gürültüsünde ya da zamanla gerçek irtifadan
    sapmış olabilir; bunu NAV_WAYPOINT hedefi yaparsak uçak o waypoint'e
    ulaştığında FC ani tırmanış/DALIŞ komutu üretebilir (2026-08-09: sahada
    gözlenen ani dalış sonrası bulundu/düzeltildi). Frame'i de entry'den
    almak (sabit 3 yerine), planın hangi frame'de yazıldığından bağımsız
    tutarlılık sağlıyor.
    """
    items = []
    for i, rp in enumerate(release_points):
        log.info(f"[MISSION] === Yönlendirme öğesi: Hedef {i+1} {rp['color'].upper()} "
              f"({rp['lat']:.6f},{rp['lon']:.6f}) tespit-anı-irtifası={rp['alt']:.1f}m "
              f"| WP irtifası (planlı)={approach_alt_m:.1f}m frame={approach_frame} ===")

        items.append(_make_mission_item(
            0, config.CMD_NAV_WAYPOINT, param2=15.0,
            lat=rp["lat"], lon=rp["lon"], alt=approach_alt_m, frame=approach_frame,
        ))
    return items


async def build_and_start_drop_mission(drone, release_points):
    """
    Eski misyonun ortasına item sıkıştırıp set_current_mission_item ile oraya
    atlamak yerine TAMAMEN YENİ bir misyon oluşturur: drop waypoint'leri +
    mevcut misyonun iniş sekansı (SEARCH_LOOP_EXIT_WP'den itibaren).

    ArduPilot, misyon öğesi seq=0'ı ne yüklenirse yüklensin FC'nin kayıtlı
    HOME konumuyla değiştirir (doğrulanmış davranış) — ama bu davranışa körü
    körüne güvenip (0,0,0) gibi anlamsız bir dolgu koymak yerine, indirilen
    ESKİ misyonun kendi seq=0 öğesi (GCS'in yüklediği/FC'nin doldurduğu
    GERÇEK home) AYNEN yeniden kullanılır (2026-08-20 düzeltmesi — eskiden
    burada (0,0,0) koordinatlı sahte bir NAV_WAYPOINT vardı; ArduPilot'un
    onu gerçekten ezdiğini varsaymak yerine artık zaten doğru veriyle
    başlıyoruz). Gerçek navigasyon (drop + iniş) seq 1'den başlar ve
    set_current_mission_item(1) ile FC doğrudan oraya yönlendirilir —
    aksi halde ilk drop hedefi sessizce home ile değiştirilip atlanabilirdi.
    """
    log.info(f"[MISSION] Mevcut misyon indiriliyor (iniş WP'lerini almak için)...")
    existing = list(await drone.mission_raw.download_mission())
    log.info(f"[MISSION] {len(existing)} öğe indirildi")

    landing_items = existing[config.SEARCH_LOOP_EXIT_WP:]
    log.info(f"[MISSION] İniş sekansı: WP {config.SEARCH_LOOP_EXIT_WP}'den itibaren "
          f"{len(landing_items)} öğe alındı")

    # DO_JUMP koruması (2026-08-17): kopyalanan iniş sekansında DO_JUMP varsa
    # param1'i ESKİ misyonun index'ini gösterir — yeni misyonda tüm seq'ler
    # kaydığı için uçak öngörülemez bir öğeye atlar. Böyle bir öğe iniş
    # sekansında zaten olmamalı (arama döngüsünün DO_JUMP'ı SEARCH_LOOP_EXIT_WP
    # ÖNCESİNDE kalır); yine de GCS planı yanlış kurulursa sessiz felaket
    # olmasın diye öğe atılır ve yüksek sesle loglanır.
    jumps = [it for it in landing_items if it.command == config.CMD_DO_JUMP]
    if jumps:
        log.info(f"[MISSION] ⚠ UYARI: iniş sekansında {len(jumps)} DO_JUMP öğesi bulundu — "
                 f"index'ler yeni misyonda geçersiz olacağından bu öğeler ATILDI. "
                 f"GCS planını ve SEARCH_LOOP_EXIT_WP'yi kontrol edin!")
        landing_items = [it for it in landing_items if it.command != config.CMD_DO_JUMP]

    return_item = _make_return_to_search_entry_item(existing)
    log.info(f"[MISSION] Tarama girişine dönüş öğesi eklendi (SEARCH_START_WP={config.SEARCH_START_WP} konumu)")

    # Drop yaklaşma waypoint'leri için irtifa+frame: GCS planındaki
    # SEARCH_START_WP öğesinden (return_item'ınkiyle AYNI kaynak) — rp["alt"]
    # (tespit anındaki anlık telemetri) DEĞİL, bkz. _build_drop_items docstring.
    approach_alt_m   = existing[config.SEARCH_START_WP].z
    approach_frame   = existing[config.SEARCH_START_WP].frame
    drop_items = _build_drop_items(release_points, approach_alt_m, approach_frame)
    log.info(f"[MISSION] {len(drop_items)} drop öğesi oluşturuldu "
             f"| yaklaşma irtifası={approach_alt_m:.1f}m frame={approach_frame}")

    home_item = existing[0]  # gerçek HOME — indirilen eski misyonun seq=0 öğesi, AYNEN korunur
    new_mission = [home_item, return_item] + drop_items + landing_items

    # Tüm seq numaralarını sıfırdan yeniden düzenle — item 0 = gerçek home,
    # item 1 = tarama girişine dönüş, item 2 = ilk drop öğesi
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
    log.info(f"[MISSION] Yeni misyon: {len(resequenced)} öğe "
          f"(1 home + 1 dönüş + {len(drop_items)} drop + {len(landing_items)} iniş)")

    await drone.mission_raw.upload_mission(resequenced)
    log.info("[MISSION] ✓ upload_mission() tamamlandı")

    await drone.mission_raw.set_current_mission_item(1)
    log.info("[MISSION] ✓ set_current_mission_item(1) — seq 0 (home) atlandı, tarama girişine dönüş aktif")

    await drone.mission_raw.start_mission()
    log.info("[MISSION] ✓ start_mission() — drop sekansı başladı (başa sıçrama yok)")

    if release_points:
        log.info("[MISSION] Canlı balistik drop_trigger_task başlatılıyor")
        asyncio.create_task(drop_trigger_task(drone, release_points))
    else:
        log.info("[MISSION] Hedef yok — drop_trigger_task başlatılmadı (yalnızca iniş)")


# ==================== WP TAKİBİ ====================

async def waypoint_tracking_task(drone):
    """Aktif WP index'ini state.current_wp'de tutar — overlay için.
    WP değiştiği ANDAKİ uçak GPS konumu da ayrıca saklanır (2026-08-20,
    bkz. vision.py HUD) — "şu an neredeyiz" (POS) değil, "bu WP'ye
    GEÇERKEN neredeydik" bilgisi."""
    last_index = None
    async for progress in drone.mission_raw.mission_progress():
        state.current_wp["index"] = progress.current
        state.current_wp["total"] = progress.total
        if progress.current != last_index:
            last_index = progress.current
            with state.telemetry_lock:
                state.current_wp["lat"] = state.current_telemetry["lat"]
                state.current_wp["lon"] = state.current_telemetry["lon"]


# ==================== TESPİT AKTİVASYONU ====================

async def detection_activation_task(drone):
    """DETECTION_ACTIVE_WP'ye gelince tespiti aktif eder."""
    log.info(f"[DETECTION] Tespit aktivasyonu bekleniyor — WP {config.DETECTION_ACTIVE_WP}'de aktif olacak")
    async for progress in drone.mission_raw.mission_progress():
        if progress.current >= config.DETECTION_ACTIVE_WP:
            state.detection_active.set()
            log.info(f"[DETECTION] ✓ Tespit AKTİF — WP {progress.current}")
            break


# ==================== ANA GÖREV ====================

async def mission_task(drone, queue):
    log.info("[MISSION] mission_task başladı — hedef tespiti bekleniyor")
    release_points       = []
    first_detection_time = None
    search_deadline      = None
    last_wait_log        = 0.0

    while len(release_points) < 2:
        # Genel tarama zaman aşımı (2026-08-17): tespit aktifleştikten sonra
        # sayaç başlar. HİÇ hedef bulunamazsa mission_task (kod tarafı) süresiz
        # beklerdi — şartnamede Görev 2 azami 10 dk, aşımı uçuşu geçersiz
        # kılıyor. Süre dolunca eldeki hedeflerle (0 dahil) iniş sekansına
        # geçilir. NOT (2026-08-19): GCS planı artık DO_JUMP kullanmıyor
        # (tarama bacağı SEARCH_START_WP→SEARCH_LOOP_EXIT_WP arası sonlu düz bir dizi),
        # ama bu zaman aşımı hâlâ gerekli — FC'nin kendi ilerleyişi hiçbir
        # zaman build_and_start_drop_mission'ı tetiklemez, bu satır olmadan
        # kod tarafı hedef bulunamasa bile sonsuza dek beklerdi.
        now_m = time.monotonic()
        if search_deadline is None and state.detection_active.is_set():
            search_deadline = now_m + config.SEARCH_TOTAL_TIMEOUT_SEC
            log.info(f"[MISSION] Tarama sayacı başladı — azami "
                     f"{config.SEARCH_TOTAL_TIMEOUT_SEC}s sonra eldekiyle çıkılacak")
        if search_deadline is not None and now_m > search_deadline:
            log.info(f"[MISSION] ⚠ Tarama zaman aşımı ({config.SEARCH_TOTAL_TIMEOUT_SEC}s) — "
                     f"{len(release_points)} hedefle iniş sekansına geçiliyor")
            break

        try:
            # get_nowait (2026-08-17): eskiden queue.get(timeout=0.5) SENKRON
            # çağrısı asyncio event loop'unu her turda 0.5s BLOKLUYORDU — tüm
            # arama fazı boyunca telemetri/attitude akışları gecikip zaman
            # damgaları kayıyor, vision'ın frame-anı telemetri eşleşmesini
            # (nearest_telemetry_at) bozuyordu. GPS sapmalarının ("277m")
            # kök nedenlerinden biri.
            target = queue.get_nowait()
            log.info(f"[MISSION] Kuyruktan alındı: {target['color'].upper()} "
                  f"({target['lat']:.6f},{target['lon']:.6f})")
        except Empty:
            if len(release_points) == 1 and first_detection_time:
                elapsed   = time.time() - first_detection_time
                remaining = config.SINGLE_TARGET_TIMEOUT_SEC - elapsed
                if elapsed > config.SINGLE_TARGET_TIMEOUT_SEC:
                    log.info(f"[MISSION] ⚠ Timeout ({config.SINGLE_TARGET_TIMEOUT_SEC}s) doldu — "
                          f"tek hedefle devam ediliyor")
                    break
                if now_m - last_wait_log >= 5.0:
                    log.info(f"[MISSION] İkinci hedef bekleniyor... (kalan≈{remaining:.0f}s)")
                    last_wait_log = now_m
            await asyncio.sleep(0.1)
            continue

        color = target["color"]
        if any(rp["color"] == color for rp in release_points):
            log.info(f"[MISSION] {color.upper()} zaten işlendi — tekrar atlandı")
            continue

        # GPS koordinatı ARTIK BURADA hesaplanmıyor — vision.py'nin izleme
        # döngüsü (bkz. _update_detection/_finalize_track) hedefi kadraja
        # ilk girdiği andan itibaren TAKİP EDİP en yakın (en düşük hatalı)
        # örneği o karenin GERÇEK telemetrisiyle zaten hesaplayıp kilitledi;
        # kuyruğa hazır (lat, lon, alt) geliyor (2026-08-12, bkz. config.py
        # DETECTION_CONFIRM_STREAK notu). Roll/pitch/mesafe reddi de aynı
        # döngüde (geo.pixel_to_gps üzerinden) zaten uygulanmış durumda.
        #
        # Balistik drop noktası burada HESAPLANMIYOR — tespit anındaki hız
        # varsayımı bırakma anında geçersiz olabilir. Hedefin kendi GPS
        # konumu saklanır, drop_trigger_task her tik'te güncel hızla
        # calculate_drop_point()'i taze hesaplar.
        release_points.append({
            "color":   color,
            "lat":     target["lat"],
            "lon":     target["lon"],
            "alt":     target["alt"],
            "dropped": False,
        })
        if first_detection_time is None:
            first_detection_time = time.time()

        # HUD'da kalıcı "kilit" bilgisi için — canlı tespitten (detected_targets)
        # bağımsız, hedef kadrajdan çıksa/tracker kaybolsa bile burada kalır
        # (2026-08-20, bkz. state.locked_targets, vision.py overlay).
        state.locked_targets[color] = (target["lat"], target["lon"])

        log.info(f"[MISSION] ✓ {color.upper()} hedef konumu kaydedildi: "
              f"({target['lat']:.6f},{target['lon']:.6f}) — "
              f"{len(release_points)}/2 hedef toplandı")

    if release_points:
        log.info(f"[MISSION] Tarama tamamlandı ({len(release_points)} hedef). "
              f"{config.SCAN_EXIT_DELAY_SEC}s bekleniyor...")
        await asyncio.sleep(config.SCAN_EXIT_DELAY_SEC)
    else:
        # Hedefsiz çıkış (tarama zaman aşımı): beklemenin anlamı yok, uçuş
        # süresi puanı için doğrudan iniş sekansına geç.
        log.info("[MISSION] ⚠ Hiç hedef bulunamadı — drop öğesiz misyonla inişe geçiliyor")

    await build_and_start_drop_mission(drone, release_points)

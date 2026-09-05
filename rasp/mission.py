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

    TIMEOUT (2026-09-05, yarış öncesi inceleme): main.py'nin İLK bağlantı
    döngüsü drone.connect()'i asyncio.wait_for(FC_CONNECT_ATTEMPT_TIMEOUT_SEC)
    ile sarmalıyor — kendi yorumunda açıkça belirtildiği gibi bu çağrının
    KENDİSİ bazen hiç dönmüyor (seri port o an yoksa mavsdk_server içeride
    askıda kalabiliyor, gözlemlenen saha olayı). Bu fonksiyon (uçuş
    ORTASINDA kopma sonrası yeniden bağlanma) aynı riski taşıdığı hâlde
    eskiden bu sarmalamaya SAHİP DEĞİLDİ — tam da bu fonksiyonun var olma
    sebebi olan senaryoda (USB kısa süreli kaybolup geri gelirken tam bu
    ana denk gelen bir connect() çağrısı) süresiz askıda kalabilir, bu da
    state.fc_reconnecting'i SONSUZA DEK True'da bırakıp (finally hiç
    çalışmaz) bir daha HİÇBİR yeniden bağlanma denemesi başlamamasına yol
    açardı — uçuşun geri kalanında FC'ye bir daha bağlanılamazdı. Artık
    aynı timeout+wait_for deseni burada da uygulanıyor.
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
                await asyncio.wait_for(
                    drone.connect(system_address=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}"),
                    timeout=config.FC_CONNECT_ATTEMPT_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                log.info(f"[FC] ⚠ Yeniden bağlanma denemesi #{attempt} "
                         f"{config.FC_CONNECT_ATTEMPT_TIMEOUT_SEC}s içinde dönmedi (askıda kaldı) — "
                         f"yeni denemeye geçiliyor")
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


async def resilient_stream(name, coro_factory):
    """
    "Pasif akış aynalama" görevlerini (telemetri/durum — kendi durumu yok,
    sadece FC'den gelen bir MAVSDK akışını state.*'e yazar) main.py'deki TEK
    asyncio.gather'dan İZOLE eder (2026-09-05 düzeltmesi).

    KÖK NEDEN: bu görevlerin hepsi `async for x in drone.telemetry.foo():`
    şeklinde grpc.aio üzerinden mavsdk_server'a abone — FC bağlantısı
    GERÇEKTEN koparsa (ör. USB CDC-ACM'in kısa süreliğine kaybolması, bkz.
    _fc_reconnect_loop docstring) altındaki grpc stream'i genelde SESSİZCE
    takılı kalmaz, bir grpc.aio.AioRpcError (ör. UNAVAILABLE) fırlatarak
    SONLANIR. main.py'deki asyncio.gather(...) return_exceptions=YOK
    çağrıldığından buradan kaçan TEK bir istisna gather'ı düşürür, main'in
    finally'si pipeline.stop_pipeline() ile VİDEO YAYININI DA durdurur
    (video kendi başına ayrı thread + ayrı OS süreçleriyle FC'den TAMAMEN
    bağımsız çalışıyor olsa da), sonra istisna asyncio.run()'a kadar
    yükselip TÜM SÜREÇ ÇÖKER (main.py yalnızca CancelledError'ı yakalıyor).
    systemd (Restart=on-failure, RestartSec=3) süreci sıfırdan başlatır —
    bu da rpicam-vid/WFB-ng pipeline'ının TAMAMEN yeniden kurulmasını
    gerektirir. Sonuç, sahada gözlenen semptomun ta kendisi: kısa bir FC
    bağlantı pürüzü, MANTIKEN video akışını hiç etkilememesi gerekirken
    görüntüyü de kesiyordu.

    ÇÖZÜM: mission.wind_track_task'taki (2026-09-02) "istisnayı yut, sistemi
    ayakta tut" desenini burada GENELLEŞTİRİYORUZ — istisna loglanır, kısa
    bir bekleme (config.FC_RECONNECT_INTERVAL_SEC) sonrası coro_factory()
    YENİDEN çağrılıp akışa TAZE abone olunur; main.py'deki gather bunu HİÇ
    görmez, dolayısıyla video (ve diğer tüm görevler) etkilenmeden sürer.
    coro_factory HER ÇAĞRIDA TAZE bir coroutine üretmeli (ör. bir lambda) —
    aynı coroutine nesnesi iki kez await edilemez.

    Yalnızca DURUMSUZ akış-aynalama görevlerine uygundur. mission_task gibi
    durum taşıyan (release_points, arama ilerleyişi) bir görevi sıfırdan
    yeniden başlatmak GÜVENLİ DEĞİLDİR — bkz. main.py çağrı noktasındaki
    ayrı (yeniden başlatmayan) sarmalayıcı.
    """
    while not state.shutdown_requested.is_set():
        try:
            await coro_factory()
            log.info(f"[{name}] ⚠ akış istisnasız bitti (beklenmedik) — yeniden abone olunuyor")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.info(f"[{name}] ⚠ akış HATA verdi: {e!r} — {config.FC_RECONNECT_INTERVAL_SEC}s "
                     f"sonra yeniden abone olunacak (video ve diğer görevler ETKİLENMEDİ)")
        await asyncio.sleep(config.FC_RECONNECT_INTERVAL_SEC)


async def guarded_mission_task(drone, queue):
    """mission_task'ı main.py'deki asyncio.gather'dan İZOLE eder — genel
    gerekçe için bkz. resilient_stream docstring'i. mission_task DURUM
    TAŞIDIĞINDAN (release_points, kilitli hedefler) resilient_stream gibi
    sıfırdan YENİDEN BAŞLATILMAZ: bir istisna burada sadece loglanıp YUTULUR,
    görev sonlanır — diğer tüm görevler (video dahil) etkilenmeden sürer.
    build_and_start_drop_mission zaten çalışıp drop_trigger_task'ı ayrı bir
    asyncio.create_task olarak başlattıysa (bkz. mission_task'ın sonu), o
    görev bu sarmalayıcıdan tamamen bağımsız kendi başına sürmeye devam eder.
    """
    try:
        await mission_task(drone, queue)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.info(f"[MISSION] ⚠ mission_task beklenmedik İSTİSNA ile sonlandı: {e!r} — "
                 f"diğer görevler (video dahil) etkilenmeden sürüyor; FC ile ilgiliyse "
                 f"o anki misyon adımı (ör. mission upload/servo) yarım kalmış olabilir, "
                 f"loglara ve FC'nin kendi misyon durumuna bakın")


# ==================== TELEMETRİ ====================

async def telemetry_task(drone):
    """Konum akışı (telemetry.position()) — ayrıca velocity_ned()'i de besleyen
    AYNI GLOBAL_POSITION_INT paketini FC'den daha sık istiyor (2026-09-05,
    bkz. config.POSITION_STREAM_HZ): drop_trigger_task'ın along/cross-track
    tetiklemesi her pozisyon tik'inde bir karar veriyor, tik'ler arası KÖR —
    varsayılan FC akış hızı (~2-5Hz, ölçülmüş) 15-20m/s uçuşta tik başına
    3-10m'lik konum sıçraması demekti (bkz. config.DROP_ALONG_TRIGGER_S notu).
    WIND'de zaten kullanılan SET_MESSAGE_INTERVAL tekniğiyle (bkz.
    wind_track_task/_make_message_interval_command) aynı isteği burada
    GLOBAL_POSITION_INT için gönderiyoruz — mavsdk_server bu TEK MAVLink
    mesajından hem Position hem VelocityNed'i türetiyor, yani tek istek
    telemetry_task VE speed_track_task'ı birden hızlandırıyor."""
    try:
        await drone.mavlink_direct.send_message(_make_message_interval_command(
            config.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, int(1_000_000 / config.POSITION_STREAM_HZ)))
        log.info(f"[TELEMETRY] FC'den GLOBAL_POSITION_INT {config.POSITION_STREAM_HZ:.0f}Hz istendi "
                 f"(SET_MESSAGE_INTERVAL) — drop tetikleme ve speed_track_task'ı da hızlandırır")
    except Exception as e:
        log.info(f"[TELEMETRY] ⚠ SET_MESSAGE_INTERVAL gönderilemedi: {e} — "
                 f"varsayılan akış hızına güveniliyor")

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


def _make_message_interval_command(msg_id, interval_us):
    """COMMAND_LONG / SET_MESSAGE_INTERVAL: FC'den belirli bir MAVLink mesajını
    verilen periyotla (µs) yayınlamasını ister (bkz. wind_track_task)."""
    return MavlinkMessage(
        message_name="COMMAND_LONG",
        system_id=255,
        component_id=190,
        target_system_id=1,
        target_component_id=1,
        fields_json=json.dumps({
            "target_system": 1,
            "target_component": 1,
            "command": config.CMD_SET_MESSAGE_INTERVAL,
            "confirmation": 0,
            "param1": float(msg_id),
            "param2": float(interval_us),
            "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0,
        })
    )


def _wind_from_to_ned(direction_from_deg, speed_ms):
    """ArduPilot WIND mesajı (direction = rüzgarın GELDİĞİ yön, derece, 0=kuzey;
    speed m/s) → hava kütlesinin GİTTİĞİ yön vektörü (wind_n, wind_e) m/s.
    geo.calculate_drop_point / wind_shifted_nav_point bu 'gittiği yön'
    (NED hız) kuralını bekler — işaret burada ters çevrilir."""
    r = math.radians(direction_from_deg)
    return -speed_ms * math.cos(r), -speed_ms * math.sin(r)


async def wind_track_task(drone):
    """
    FC'nin (ArduPilot EKF) rüzgar tahminini sürekli günceller (2026-09-02) —
    calculate_drop_point()'in ters/yanal rüzgarı telafi edebilmesi ve
    _build_drop_items'ın hedef WP'lerini rüzgara karşı kaydırabilmesi için
    (bkz. config.DROP_WIND_ALONG_GAIN/DROP_WIND_CROSS_GAIN). Anlık yer hızı
    eksi hava hızı farkını KENDİMİZ hesaplamak yerine EKF'nin zaman içinde
    filtrelediği tahmini kullanıyoruz — dönüşlerde sideslip nedeniyle anlık
    GS-TAS farkının gürültülü olmasından kaçınmak için.

    KAYNAK DEĞİŞİKLİĞİ (2026-09-02 akşam): eskiden MAVSDK telemetry.wind()
    kullanılıyordu; bu MAVLink WIND_COV mesajından beslenir ve ArduPilot
    WIND_COV GÖNDERMEZ (PX4 gönderir). Sonuç: bugünkü 3 uçuşun 21 tetik
    tik'inin HEPSİNDE rüzgar_n=rüzgar_e=0.0 loglandı — rüzgar telafisi
    (along+cross) deploy edildiği günden beri fiilen DEVRE DIŞIYDI, "sola
    atma"nın 3977b82 sonrası da sürmesinin bir nedeni bu. ArduPilot'un
    gerçekten yayınladığı mesaj WIND (id 168, ardupilotmega): direction =
    rüzgarın GELDİĞİ yön (derece), speed (m/s), speed_z. mavlink_direct ile
    doğrudan buna abone oluyoruz ve _wind_from_to_ned ile "gittiği yön"
    NED vektörüne çeviriyoruz (GCS'in rüzgar okuyla aynı kural: Mission
    Planner 'Wind 45° 5m/s' = kuzeydoğuDAN esiyor → wind_n,wind_e negatif).

    Başlangıçta SET_MESSAGE_INTERVAL ile WIND'i WIND_STREAM_HZ'de ister
    (varsayılan SRx_EXTRA3 akış hızı düşük/kapalı olabilir; zaten
    akıyorsa zararsız). İlk örnek WIND_FIRST_SAMPLE_TIMEOUT_S içinde
    gelmezse yüksek sesle uyarır — sessiz 0.0 bir daha olmasın diye.
    Ayrıca her WIND_LOG_EVERY_N örnekte bir değer loglanır ki uçuş
    sonrası log'dan rüzgarın gerçekten okunduğu görülsün.

    config.WIND_MANUAL_FROM_DEG/WIND_MANUAL_SPEED_MS doluysa FC tahmini
    YOK SAYILIR ve sabit elle girilen değer kullanılır (saha ölçümü / EKF
    tahmini güvenilmezse; uçuş öncesi GCS rüzgar okumasıyla girilir).
    """
    if config.WIND_MANUAL_FROM_DEG is not None and config.WIND_MANUAL_SPEED_MS is not None:
        wind_n, wind_e = _wind_from_to_ned(config.WIND_MANUAL_FROM_DEG, config.WIND_MANUAL_SPEED_MS)
        with state.telemetry_lock:
            state.current_telemetry["wind_n"] = wind_n
            state.current_telemetry["wind_e"] = wind_e
        log.info(f"[WIND] ELLE GİRİLEN rüzgar kullanılıyor (FC tahmini yok sayılıyor): "
                 f"{config.WIND_MANUAL_SPEED_MS:.1f}m/s, {config.WIND_MANUAL_FROM_DEG:.0f}°'DEN "
                 f"→ wind_n={wind_n:+.1f} wind_e={wind_e:+.1f} m/s")
        return

    try:
        await drone.mavlink_direct.send_message(_make_message_interval_command(
            config.MAVLINK_MSG_ID_WIND, int(1_000_000 / config.WIND_STREAM_HZ)))
        log.info(f"[WIND] FC'den WIND mesajı {config.WIND_STREAM_HZ:.0f}Hz istendi (SET_MESSAGE_INTERVAL)")
    except Exception as e:
        log.info(f"[WIND] ⚠ SET_MESSAGE_INTERVAL gönderilemedi: {e} — varsayılan akış hızına güveniliyor")

    log.info("[WIND] Rüzgar tahmini akışı başlatıldı (MAVLink WIND, mavlink_direct)")
    started = time.monotonic()
    stats = {"count": 0, "last_t": None}

    # Sessizlik bekçisi AYRI görevde: akışın kendisini wait_for ile
    # zamanlamak olmaz — timeout'ta iptal edilen anext() mavlink_direct
    # generator'ının finally bloğunu tetikleyip gRPC aboneliğini kapatır.
    async def _watchdog():
        while True:
            await asyncio.sleep(config.WIND_FIRST_SAMPLE_TIMEOUT_S)
            if stats["count"] == 0:
                log.info(f"[WIND] ⚠ {time.monotonic() - started:.0f}s'dir hiç WIND mesajı gelmedi — "
                         f"FC rüzgar tahmini yayınlamıyor olabilir (SRx_EXTRA3 / EKF). Rüzgar "
                         f"telafisi ve WP kaydırması bu uçuşta DEVRE DIŞI kalacak!")
            elif time.monotonic() - stats["last_t"] > config.WIND_FIRST_SAMPLE_TIMEOUT_S:
                log.info(f"[WIND] ⚠ WIND akışı {time.monotonic() - stats['last_t']:.0f}s'dir sessiz "
                         f"(son değer kullanılmaya devam ediyor)")

    watchdog = asyncio.create_task(_watchdog())
    # Bu görev main.py'de asyncio.gather içinde diğerleriyle birlikte koşuyor
    # (return_exceptions YOK): buradan kaçan bir istisna gather'ı düşürür ve
    # main'in finally'si UÇUŞ ORTASINDA tüm sistemi kapatır. Rüzgar
    # yardımcı bir bilgidir — kaynağı ne olursa olsun burada yutulur, rüzgar
    # None kalır (build_and_start_drop_mission bunu loglayıp kaydırmasız
    # devam eder).
    try:
        async for msg in drone.mavlink_direct.message("WIND"):
            try:
                f = json.loads(msg.fields_json)
                direction_from = float(f["direction"])
                speed = float(f["speed"])
            except (KeyError, ValueError, TypeError) as e:
                log.info(f"[WIND] ⚠ WIND mesajı çözümlenemedi: {e} — {msg.fields_json!r}")
                continue

            wind_n, wind_e = _wind_from_to_ned(direction_from, speed)
            with state.telemetry_lock:
                state.current_telemetry["wind_n"] = wind_n
                state.current_telemetry["wind_e"] = wind_e
            stats["count"] += 1
            stats["last_t"] = time.monotonic()
            if stats["count"] == 1:
                log.info(f"[WIND] ✓ İlk WIND örneği (+{stats['last_t'] - started:.0f}s): "
                         f"{speed:.1f}m/s {direction_from:.0f}°'DEN esiyor → "
                         f"wind_n={wind_n:+.1f} wind_e={wind_e:+.1f} m/s")
            elif stats["count"] % config.WIND_LOG_EVERY_N == 0:
                log.info(f"[WIND] #{stats['count']}: {speed:.1f}m/s {direction_from:.0f}°'DEN → "
                         f"wind_n={wind_n:+.1f} wind_e={wind_e:+.1f}")
        log.info("[WIND] ⚠ WIND akışı sonlandı")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.info(f"[WIND] ⚠ WIND aboneliği HATA verdi: {e!r} — rüzgar telafisi bu uçuşta devre dışı "
                 f"(sistem çalışmaya devam ediyor)")
    finally:
        watchdog.cancel()


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


async def drop_trigger_task(drone, release_points, landing_start_seq=None):
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

    landing_start_seq (2026-08-26): hedef WP bloğu sonuna eklenen DO_JUMP'ın
    (bkz. build_and_start_drop_mission/_make_drop_retry_jump_item) atladığı
    iniş sekansının NİHAİ seq'i. Yükler döngü tekrar hakkı bitmeden önce
    ikisi de bırakılırsa, uçağın kalan tekrar hakkını boşuna hedef bloğunu
    yeniden gezerek harcamasını beklemek yerine set_current_mission_item ile
    doğrudan bu seq'e atlatılır.
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
        wind_n = tel["wind_n"] or 0.0
        wind_e = tel["wind_e"] or 0.0

        for rp in release_points:
            if rp["dropped"]:
                continue
            release_lat, release_lon = geo.calculate_drop_point(
                rp["lat"], rp["lon"], pos.relative_altitude_m, tel["speed"], course_deg,
                wind_n=wind_n, wind_e=wind_e,
            )
            dist = geo.haversine(pos.latitude_deg, pos.longitude_deg, release_lat, release_lon)
            # HUD'da KILIT satırının yanında canlı gösterilir (2026-09-05,
            # bkz. state.drop_distance) — kurulma eşiğinden ÖNCE de, armed
            # olur olmaz da güncellenir; kullanıcı mesafenin azalışını
            # "SERVO AÇILDI" anıyla aynı ekranda karşılaştırabilsin diye.
            state.drop_distance[rp["color"]] = dist

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
                     f"t={time_to:.2f}s (hız={tel['speed']:.1f}m/s alt={pos.relative_altitude_m:.1f}m "
                     f"rüzgar_n={wind_n:.1f} rüzgar_e={wind_e:.1f})")

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
            if landing_start_seq is not None:
                # Hedef bloğu sonundaki DO_JUMP (bkz. build_and_start_drop_mission)
                # yükler zaten bırakılmış olsa da kalan tekrar hakkını boşuna
                # harcayıp hedef bloğunu yeniden gezebilir — FC'yi doğrudan
                # iniş sekansına atlatarak bu beklemeyi atlıyoruz.
                try:
                    await drone.mission_raw.set_current_mission_item(landing_start_seq)
                    log.info(f"[DROP] ✓ DO_JUMP döngüsü atlanıp doğrudan iniş sekansına "
                             f"geçildi (WP{landing_start_seq})")
                except Exception as e:
                    log.info(f"[DROP] ⚠ iniş sekansına atlama HATASI: {e} — "
                             f"uçak DO_JUMP döngüsünü kendi haline tamamlayacak")
            break


async def _resilient_drop_trigger(drone, release_points, landing_start_seq):
    """
    drop_trigger_task'ı build_and_start_drop_mission'ın kendi asyncio.create_task
    çağrısıyla sarmalar (2026-09-05, yarış öncesi inceleme) — bu görev main.py'nin
    ana asyncio.gather'ının DIŞINDA, tamamen ayrı ateşlenmiş bir task olduğundan
    resilient_stream/guarded_mission_task'ın kapsamadığı AYRI bir kırılganlık
    noktası: drop_trigger_task da diğerleri gibi `async for pos in
    drone.telemetry.position():` ile bir MAVSDK akışını tüketiyor — FC bağlantısı
    tam da DROP YAKLAŞMASI SIRASINDA (uçuşun en kritik anı) hıçkırık yaparsa
    (bkz. resilient_stream docstring — aynı grpc.aio.AioRpcError riski) bu
    fonksiyon istisna fırlatıp SESSİZCE ölür; onu hiçbir şey izlemediğinden bir
    daha HİÇBİR hedef için servo tetiklenmez, kalan DO_JUMP retry geçişleri bile
    boşa gider (kod tarafı artık değerlendirme yapmıyor) — sonuç: yükler hiç
    bırakılmaz.

    ÇÖZÜM: istisna yakalanır, henüz bırakılmamış hedef varsa kısa bir bekleme
    sonrası drop_trigger_task YENİDEN başlatılır. rp['dropped'] bayrakları
    release_points üzerinde (referansla paylaşılan, drop_trigger_task'ın kendi
    yerel değişkeni DEĞİL) kalıcı olduğundan zaten bırakılmış yükler yeniden
    TETİKLENMEZ — yalnızca rp['armed']/prev_along (drop_trigger_task'ın kendi
    başlangıcında sıfırlanıyor) sıfırlanır, bu da o hedefin yeniden KURULMASINI
    (arming, DROP_TRIGGER_RADIUS_M'ye tekrar girmesi) bir tık geciktirir —
    yanlış/tekrar bırakmaya yol AÇMAZ. Tüm yükler zaten bırakılmışken (normal
    tamamlanma, ya da tamamlanma sonrası tesadüfen bir istisna) YENİDEN
    BAŞLATILMAZ — sonsuz anlamsız yeniden başlatma döngüsüne girilmesin diye.
    """
    while not state.shutdown_requested.is_set():
        try:
            await drop_trigger_task(drone, release_points, landing_start_seq)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if all(rp["dropped"] for rp in release_points):
                log.info(f"[DROP] drop_trigger_task istisna ile bitti ama tüm yükler zaten "
                         f"bırakılmıştı ({e!r}) — yeniden başlatılmıyor")
                return
            log.info(f"[DROP] ⚠ drop_trigger_task HATA verdi: {e!r} — "
                     f"{config.FC_RECONNECT_INTERVAL_SEC}s sonra YENİDEN başlatılacak "
                     f"(henüz bırakılmamış hedef(ler) var)")
            await asyncio.sleep(config.FC_RECONNECT_INTERVAL_SEC)
            continue
        return  # istisnasız (normal) dönüş — tüm yükler bırakıldı, tekrara gerek yok


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


# ==================== DROP MİSYONU ====================
# NOT (2026-08-20): speed_management_task (SEARCH_START_WP'de DO_CHANGE_SPEED
# gönderen görev) kaldırıldı — hız yönetimi kod tarafının işi değil, uçak
# kendi (GCS/FC'de ayarlı) hızında uçar. Kod tarafı yalnızca WP koyar ve
# doğru anda servo tetikler (bkz. drop_trigger_task).

# build_and_start_drop_mission() yeni misyonu HER ZAMAN aynı sırayla kurar:
# [home_item, return_item, drop_item(ler)i, ...] — yani drop bloğu HER ZAMAN
# seq 2'den başlar (bkz. build_and_start_drop_mission _drop_seq_offset).
# mission_task bu sabiti, henüz yeni misyon yüklenmeden ÖNCE bir hedef
# kilitlenir kilitlenmez o hedefin YENİ plandaki WP numarasını HUD için
# TAHMİN ETMEK üzere kullanır (bkz. mission_task, state.locked_target_wp).
_DROP_BLOCK_START_SEQ = 2  # len([home_item, return_item])

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

    İrtifa (2026-09-01): lat/lon GCS planındaki WP{SEARCH_START_WP}'den
    (entry.x/y) AYNEN alınır ama irtifa entry.z (tarama irtifası, GCS'te
    50m) DEĞİL, drop WP'leriyle AYNI config.DROP_TARGET_ALT_M kullanılır.
    Eskiden entry.z (50m) korunuyordu — uçak bu dönüş noktasından ilk drop
    WP'sine (25m) çok kısa bir bacakta inmek zorunda kalıyor, saha
    gözlemi: hedeflerde 25m'ye alçalma yetişmiyordu. Artık alçalma tarama
    bacağının kesildiği yerden (çok daha uzun mesafe/süre) itibaren
    kademeli başlıyor, dönüş noktasına vardığında uçak zaten hedef
    irtifasında oluyor.
    """
    entry = existing[config.SEARCH_START_WP]
    if entry.x == 0 and entry.y == 0:
        log.info(f"[MISSION] ⚠ UYARI: SEARCH_START_WP={config.SEARCH_START_WP} öğesinin koordinatı "
                 f"(0,0) — bu index yanlış olabilir (command={entry.command}), GCS planını kontrol edin!")
    return RawMissionItem(
        0, entry.frame, config.CMD_NAV_WAYPOINT, 0, 1,
        0.0, 15.0, 0.0, 0.0,
        entry.x, entry.y, config.DROP_TARGET_ALT_M, 0,
    )


def _build_drop_items(release_points, wind_n=0.0, wind_e=0.0):
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

    İrtifa (2026-08-27): hedef WP'leri artık SABİT config.DROP_TARGET_ALT_M
    (relative, MAV_FRAME_GLOBAL_RELATIVE_ALT) taşıyor — tarama irtifasından
    (SEARCH_START_WP'nin planlı irtifası, hâlâ daha yüksek) bu WP'ye doğru
    gerçek bir ALÇALMA/yaklaşma emri oluşturuluyor, uçak artık hedefin
    üstüne tarama irtifasında düz geçmiyor. Önceki tasarımda (2026-08-20…26)
    hedef WP'ler SEARCH_START_WP'nin irtifasını devralıyordu — irtifa hiç
    değişmediği için "yaklaşma" gerçekleşmiyordu, yalnızca yatay yönlendirme
    vardı.
    rp["alt"] (tespit anındaki ANLIK telemetri irtifası) BİLEREK
    kullanılmıyor — tek bir telemetri örneği, bank/tur sırasında GPS/baro
    gürültüsünde ya da zamanla gerçek irtifadan sapmış olabilir; bunu
    NAV_WAYPOINT hedefi yaparsak uçak o waypoint'e ulaştığında FC ani
    tırmanış/DALIŞ komutu üretebilir (2026-08-09: sahada gözlenen ani dalış
    sonrası bulunup DEĞİL rp["alt"] KULLANILMAYARAK düzeltildi — sabit
    DROP_TARGET_ALT_M bu riski taşımıyor, telemetriden bağımsız).

    Rüzgar kaydırması (2026-09-02, "sola atma" düzeltmesi): WP koordinatı
    artık ham hedef DEĞİL, geo.wind_shifted_nav_point() ile o anki EKF
    rüzgar tahminine göre rüzgarın GELDİĞİ yöne kaydırılmış nokta
    (rp["nav_lat"/"nav_lon"]). rp["lat"/"lon"] ham hedef olarak KALIR —
    drop_trigger_task bırakma anını hâlâ ham hedefe göre hesaplar (release
    noktası zaten aynı rüzgarla kaydırılıyor, ikisi tutarlı; neden çift
    telafi olmadığı geo.wind_shifted_nav_point docstring'inde). Misyon uçuşta
    yeniden yüklenemediği için bu tek seferlik bir anlık görüntüdür; rüzgar
    sonradan değişirse fark drop_trigger_task'ın cross kontrolüne yansır.
    """
    items = []
    for i, rp in enumerate(release_points):
        nav_lat, nav_lon, shift_n, shift_e = geo.wind_shifted_nav_point(
            rp["lat"], rp["lon"], config.DROP_TARGET_ALT_M, wind_n=wind_n, wind_e=wind_e,
        )
        rp["nav_lat"], rp["nav_lon"] = nav_lat, nav_lon
        log.info(f"[MISSION] === Yönlendirme öğesi: Hedef {i+1} {rp['color'].upper()} "
              f"({rp['lat']:.6f},{rp['lon']:.6f}) tespit-anı-irtifası={rp['alt']:.1f}m "
              f"| WP irtifası (sabit, yaklaşma)={config.DROP_TARGET_ALT_M:.1f}m ===")
        log.info(f"[MISSION]     rüzgar kaydırması: kuzey={shift_n:+.1f}m doğu={shift_e:+.1f}m "
                 f"(|{math.hypot(shift_n, shift_e):.1f}m|, rüzgar_n={wind_n:.1f} rüzgar_e={wind_e:.1f} m/s) "
                 f"→ WP ({nav_lat:.6f},{nav_lon:.6f})")

        items.append(_make_mission_item(
            0, config.CMD_NAV_WAYPOINT, param2=config.DROP_WP_ACCEPT_RADIUS_M,
            lat=nav_lat, lon=nav_lon, alt=config.DROP_TARGET_ALT_M, frame=3,
        ))
    return items


def _make_drop_retry_jump_item(target_seq, repeat_count):
    """
    Hedef WP bloğunun (mavi+kırmızı drop WP'leri) hemen sonuna eklenen
    DO_JUMP (2026-08-26): bir hedef WP'sinin kabul yarıçapına hiç
    girilemediği (dar dönüş/rüzgar) ya da along/cross-track penceresinin
    kaçırıldığı (bkz. drop_trigger_task docstring — eskiden bu durumda
    manuel GCS müdahalesi bekleniyordu) durumda uçak hedef bloğunu
    config.DROP_RETRY_PASS_COUNT kadar DAHA tekrar dener. Süresiz döngü
    OLMASIN diye SINIRLI — bir hedef gerçekten ulaşılamazsa sayaç bitince
    iniş sekansına düşülür. Yükler döngü bitmeden erken bırakılırsa
    drop_trigger_task bu bekleyişi ATLAYIP set_current_mission_item ile
    uçağı doğrudan iniş sekansına yönlendirir (bkz. drop_trigger_task).

    target_seq: hedef bloğun İLK öğesinin NİHAİ (resequenced) seq'i —
    çağıran taraf (build_and_start_drop_mission) new_mission'a eklemeden
    ÖNCE bu index'i zaten biliyor (home+dönüş sayısı kadar offset).
    Seq geçici 0 — insert sonrası yeniden numaralandırılır.
    """
    return _make_mission_item(0, config.CMD_DO_JUMP, param1=float(target_seq),
                               param2=float(repeat_count))


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

    Hedef WP bloğu (mavi+kırmızı drop öğeleri) sonuna, kendi bloğunun BAŞINA
    dönen bir DO_JUMP eklenir (2026-08-26, bkz. _make_drop_retry_jump_item):
    bir hedefe kabul yarıçapına hiç girilemez ya da along/cross-track
    penceresi kaçırılırsa (bkz. drop_trigger_task) uçak bloğu
    config.DROP_RETRY_PASS_COUNT kadar daha tekrar dener, sonsuza dek
    dönmez. Bu, iniş sekansında bulunan DO_JUMP'ları ATAN yukarıdaki
    korumayla ÇELİŞMİYOR — korunan `landing_items` ESKİ (GCS) misyondan
    kopyalanıyor ve yeni seq'lerde geçersiz olurdu, bu DO_JUMP ise koddan
    YENİ üretiliyor ve hedef seq'i zaten NİHAİ numaralamaya göre hesaplanıyor.
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
    log.info(f"[MISSION] Tarama girişine dönüş öğesi eklendi (SEARCH_START_WP={config.SEARCH_START_WP} konumu, "
             f"irtifa={config.DROP_TARGET_ALT_M:.1f}m — tarama irtifası DEĞİL, hedef irtifası)")

    # Drop yaklaşma waypoint'leri artık SABİT config.DROP_TARGET_ALT_M taşıyor
    # (yere yaklaşma davranışı) — GCS planındaki SEARCH_START_WP irtifasını
    # DEVRALMIYOR, bkz. _build_drop_items docstring.
    # Rüzgar anlık görüntüsü (2026-09-02): hedef WP'leri rüzgarın geldiği yöne
    # kaydırılıyor (bkz. _build_drop_items / geo.wind_shifted_nav_point).
    # wind_track_task main.py'de baştan beri koşuyor, normalde dolu; hiç
    # örnek gelmediyse kaydırma yapılmaz (0) ve yüksek sesle loglanır —
    # sessizce eski (kaydırmasız) davranışa düşmek yerine.
    with state.telemetry_lock:
        wind_n = state.current_telemetry["wind_n"]
        wind_e = state.current_telemetry["wind_e"]
    if wind_n is None or wind_e is None:
        log.info("[MISSION] ⚠ EKF rüzgar tahmini yok (telemetry.wind hiç örnek vermedi) — "
                 "hedef WP'leri rüzgara göre KAYDIRILMADI, yanal telafi bu uçuşta devre dışı")
        wind_n, wind_e = 0.0, 0.0
    drop_items = _build_drop_items(release_points, wind_n=wind_n, wind_e=wind_e)
    log.info(f"[MISSION] {len(drop_items)} drop öğesi oluşturuldu "
             f"| hedef irtifası={config.DROP_TARGET_ALT_M:.1f}m (sabit, yaklaşma)")

    home_item = existing[0]  # gerçek HOME — indirilen eski misyonun seq=0 öğesi, AYNEN korunur

    # Hedef WP bloğunun (mavi+kırmızı) NİHAİ (resequenced) başlangıç index'i —
    # DO_JUMP hedefi (aşağıda) ve drop_trigger_task'ın erken-iniş atlaması
    # (2026-08-26) bu değere göre hesaplanıyor; item'lar new_mission'a
    # eklenmeden ÖNCE biliniyor olması gerekiyor çünkü DO_JUMP param1 bu
    # index'i taşıyor.
    _drop_seq_offset = _DROP_BLOCK_START_SEQ
    # DO_JUMP hedefi (2026-08-27): 2+ hedefte bloğun BAŞINA (_drop_seq_offset,
    # ilk drop öğesi) atlamak sorun değil — uçak mavi→kırmızı sırasıyla baştan
    # düzgün bir yaklaşma bacağı katediyor. TEK hedefte ise bloğun başı ZATEN
    # o tek drop öğesinin kendisi — uçak WP'yi YENİ geçmişken kendi üstüne
    # atlarsa hizalanacak bacağı olmadan anlık keskin dönüşe zorlanıyor
    # (kararsız/dar dönüş, tam da bu DO_JUMP'ın çözmeye çalıştığı sorunu
    # büyütüyor). Bu yüzden tek hedefte atlama noktası bir öncekine,
    # return_item'a (tarama girişine dönüş, seq=1) çekilir — uçak ilk
    # yaklaşmasındakiyle AYNI bacağı kullanarak hedefe yeniden yönelir.
    search_entry_seq = len([home_item])  # return_item'ın nihai seq'i (=1)
    jump_items = []
    if drop_items:
        jump_target_seq = _drop_seq_offset if len(release_points) > 1 else search_entry_seq
        jump_items = [_make_drop_retry_jump_item(jump_target_seq, config.DROP_RETRY_PASS_COUNT)]
        log.info(f"[MISSION] Hedef bloğu sonuna DO_JUMP eklendi: WP{jump_target_seq}'e "
                 f"{config.DROP_RETRY_PASS_COUNT} tekrar hakkıyla "
                 f"({'tarama girişine dönüş — tek hedef' if len(release_points) == 1 else 'hedef bloğu başı'}) "
                 f"— WP'ye ulaşılamama/geçiş ıskası durumunda otomatik yeniden deneme "
                 f"(bkz. config.DROP_RETRY_PASS_COUNT)")
    landing_start_seq = _drop_seq_offset + len(drop_items) + len(jump_items)

    new_mission = [home_item, return_item] + drop_items + jump_items + landing_items

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
          f"(1 home + 1 dönüş + {len(drop_items)} drop + {len(jump_items)} do_jump + "
          f"{len(landing_items)} iniş)")

    # HUD'daki hedef WP numarasını TEYİT eder (2026-08-26, güncelleme 2026-09-05):
    # mission_task artık kilit ANINDA aynı formülle (_DROP_BLOCK_START_SEQ +
    # release_points sırası) TAHMİN yazıyor, bu yüzden burada normalde
    # değişiklik olmaz — ama tek doğruluk kaynağı burası: drop_items sırası
    # release_points sırasıyla birebir aynı (bkz. _build_drop_items), offset
    # [home_item, return_item] uzunluğu kadar (_DROP_BLOCK_START_SEQ). Bu
    # satır, formül ileride burada değişip mission_task'takiyle senkron
    # kalmazsa HUD'un yine de FİİLEN yüklenen misyona göre doğru numarayı
    # göstermesini garantiler.
    for i, rp in enumerate(release_points):
        state.locked_target_wp[rp["color"]] = _drop_seq_offset + i
    log.info(f"[MISSION] Kilitli hedef WP numaraları yeni plana göre güncellendi: "
             + ", ".join(f"{rp['color'].upper()}=WP{_drop_seq_offset + i}"
                          for i, rp in enumerate(release_points)))

    await drone.mission_raw.upload_mission(resequenced)
    log.info("[MISSION] ✓ upload_mission() tamamlandı")

    await drone.mission_raw.set_current_mission_item(1)
    log.info("[MISSION] ✓ set_current_mission_item(1) — seq 0 (home) atlandı, tarama girişine dönüş aktif")

    await drone.mission_raw.start_mission()
    log.info("[MISSION] ✓ start_mission() — drop sekansı başladı (başa sıçrama yok)")

    if release_points:
        log.info(f"[MISSION] Canlı balistik drop_trigger_task başlatılıyor "
                 f"(iniş sekansı WP{landing_start_seq}'de, erken bitişte oraya atlanacak)")
        asyncio.create_task(_resilient_drop_trigger(drone, release_points, landing_start_seq))
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
    """Tespiti YALNIZCA DETECTION_ACTIVE_WP..SEARCH_LOOP_EXIT_WP aralığında aktif tutar.
    2026-08-24: eskiden DETECTION_ACTIVE_WP'ye ulaşınca bir kere set edilip task
    sonlanıyordu — tarama bacağı bittikten (SEARCH_LOOP_EXIT_WP) sonra, yaklaşma/
    iniş bacağı boyunca da AKTİF kalmaya devam ediyordu. Böylece uçak kendi
    bulduğu (transit/tırmanış veya iniş sekansındaki) sahte hedefleri de kabul
    edebiliyordu. Artık aralığın DIŞINA çıkınca bayrak PASİFE çekilir."""
    log.info(f"[DETECTION] Tespit aktivasyonu bekleniyor — WP "
             f"{config.DETECTION_ACTIVE_WP}-{config.SEARCH_LOOP_EXIT_WP} aralığında aktif olacak")
    async for progress in drone.mission_raw.mission_progress():
        in_range = config.DETECTION_ACTIVE_WP <= progress.current < config.SEARCH_LOOP_EXIT_WP
        if in_range and not state.detection_active.is_set():
            state.detection_active.set()
            log.info(f"[DETECTION] ✓ Tespit AKTİF — WP {progress.current}")
        elif not in_range and state.detection_active.is_set():
            state.detection_active.clear()
            log.info(f"[DETECTION] ✗ Tespit PASİF — WP {progress.current} (tarama aralığı dışı)")


# ==================== ANA GÖREV ====================

async def mission_task(drone, queue):
    log.info("[MISSION] mission_task başladı — hedef tespiti bekleniyor")
    release_points       = []
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
            # Süre bazlı SINGLE_TARGET_TIMEOUT_SEC kaldırıldı (2026-08-27) —
            # testte işe yaramadı (uçak WP30'u geçtikten çok önce/sonra süre
            # dolabiliyordu). Artık WP index'i doğrudan referans: tarama
            # bacağı SEARCH_LOOP_EXIT_WP'de (30) bitiyor, uçak o WP'yi
            # GEÇTİYSE artık ikinci hedef aranmaz, eldeki tek hedefle devam
            # edilir.
            if len(release_points) == 1:
                current_wp_idx = state.current_wp["index"]
                if current_wp_idx is not None and current_wp_idx >= config.SEARCH_LOOP_EXIT_WP:
                    log.info(f"[MISSION] ⚠ WP{current_wp_idx} — tarama bacağı sonu "
                          f"(SEARCH_LOOP_EXIT_WP={config.SEARCH_LOOP_EXIT_WP}) geçildi — "
                          f"tek hedefle devam ediliyor")
                    break
                if now_m - last_wait_log >= 5.0:
                    log.info(f"[MISSION] İkinci hedef bekleniyor... "
                          f"(WP {current_wp_idx}/{config.SEARCH_LOOP_EXIT_WP})")
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

        # HUD'da kalıcı "kilit" bilgisi için — canlı tespitten (detected_targets)
        # bağımsız, hedef kadrajdan çıksa/tracker kaybolsa bile burada kalır
        # (2026-08-20, bkz. state.locked_targets, vision.py overlay).
        state.locked_targets[color] = (target["lat"], target["lon"])
        # Bu hedefin YENİ (drop) plandaki WP numarası — kilit ANINDA TAHMİN
        # EDİLİR (2026-09-05 düzeltmesi): eskiden burada state.current_wp["index"]
        # (o an içinde bulunulan ESKİ/tarama misyonunun WP'si) yazılıyordu —
        # HUD, yeni misyon fiilen yüklenene kadar (genelde ikinci hedef de
        # kilitlenip tarama bitene dek, yani uçuşun BÜYÜK kısmında) YANLIŞ/
        # anlamsız bir WP numarası gösteriyordu. Drop bloğu HER ZAMAN
        # _DROP_BLOCK_START_SEQ'ten başlayıp release_points sırasıyla birebir
        # numaralandığından (bkz. _build_drop_items, build_and_start_drop_mission
        # _drop_seq_offset döngüsü), bu hedefin nihai WP'si release_points'teki
        # kendi sırasından (yukarıdaki append SONRASI son eleman) baştan bilinir
        # — mission build'ini beklemeye gerek yok. build_and_start_drop_mission
        # yeni misyonu yüklerken bu değeri yine de aynı formülle TEYİDEN yazar
        # (bkz. orada) — ikisi her zaman aynı sonucu verir.
        state.locked_target_wp[color] = _DROP_BLOCK_START_SEQ + len(release_points) - 1

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

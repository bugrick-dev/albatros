"""Coğrafi hesaplamalar: Haversine mesafesi, piksel→GPS dönüşümü, balistik drop point."""
import logging
import math
import numpy as np
import config

log = logging.getLogger("geo")


def _rx(phi):
    """Roll — gövde X (ileri) ekseni etrafında rotasyon."""
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(theta):
    """Pitch — gövde/kamera Y (sağ) ekseni etrafında rotasyon."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(psi):
    """Yaw — dünya Z (aşağı) ekseni etrafında rotasyon."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# NOT (2026-08-17): Bu modüldeki TÜM adım logları log.debug'a indirildi.
# pixel_to_gps tespit sürerken kare başına (30fps x 2 renk) 4-6 satır INFO
# basıyordu — konsol+dosya handler'larıyla görüntü işleme thread'ini
# yavaşlatıp frame backpressure'a (bayat kare → yanlış telemetri eşleşmesi
# → GPS sapması) katkı veriyor ve SD kartı dolduruyordu (vision.log 16MB).
# Kilit/karar olayları vision.py ve mission.py'de INFO olarak loglanmaya
# devam ediyor; ham projeksiyon adımları gerekirse logger seviyesi DEBUG'a
# çekilerek görülebilir.

# --- HUD için son reddedilme nedeni (2026-08-26) ---
# pixel_to_gps() None dönünce SEBEBİ yalnızca (yukarıdaki nottaki performans
# kaygısıyla) log.debug()'a yazıyor — DEBUG kapalıyken bu sebep hem logda hem
# HUD'da tamamen görünmez kalıyordu (kırmızı/mavi neden hiç kilitlenmiyor
# sorusuna canlı yayından cevap yoktu). Dönüş imzasını değiştirip her çağrı
# noktasını/testi güncellemek yerine, fonksiyon imzasını AYNI bırakıp kısa
# nedeni bu modül-seviyesi değişkene yazıyoruz — vision.py geo_result None
# döndüğünde bunu okuyup ekranın sağ tarafına küçük bir satır olarak basar
# (bkz. vision.py _update_detection, state.detection_reject_reason). Tek
# thread'den (opencv_processing_thread) çağrıldığı için race riski yok.
last_reject_reason = None


def wrap180(deg):
    """Açıyı [-180, 180) aralığına katlar (180 girişi -180 döner — eşdeğer açı)."""
    return (deg + 180.0) % 360.0 - 180.0


def bearing_deg(lat1, lon1, lat2, lon2):
    """1. noktadan 2. noktaya düz-dünya (küçük mesafe) kerterizi, derece (0=kuzey)."""
    dn = (lat2 - lat1) * 111320.0
    de = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat1))
    return math.degrees(math.atan2(de, dn))


def haversine(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki yüzey mesafesini metre cinsinden döner."""
    R    = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def point_in_polygon(lat, lon, polygon):
    """Ray-casting: (lat, lon) polygon (lat,lon köşe listesi) içinde mi?

    Alan küçük (birkaç yüz metre) olduğu için lat/lon düzlemsel kabul
    edilip doğrudan üzerinde çalışılıyor — enlem/boylam ölçek farkı
    (cos(lat) çarpanı) bu ölçekte içeride/dışarıda kararını değiştirecek
    kadar önemli değil, ray-casting yalnızca kesişim SAYISINA bakıyor.
    """
    n = len(polygon)
    inside = False
    x, y = lon, lat
    x1, y1 = polygon[0][1], polygon[0][0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n][1], polygon[i % n][0]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2) and y1 != y2:
            x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or x <= x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, roll_deg, pitch_deg, target_cx, target_cy):
    """
    Kamera frame piksel koordinatlarını GPS koordinatına dönüştürür.

    roll_deg/pitch_deg: FC'den okunan GERÇEK uçak duruşu (attitude_euler) —
    tespit anındaki bank/pitch'i hesaba katmak zorunlu; aksi halde bank
    yaparken yapılan tespitlerde onlarca-yüzlerce metre hata oluşuyor
    (2026-08-08 simülasyonla bulundu/doğrulandı — 15° roll'da dead-center
    tespit bile ~16m kayıyordu, eskiden roll hiç kullanılmıyordu).

    MAX_ROLL_FOR_DETECTION_DEG/MAX_PITCH_FOR_DETECTION_DEG aşılırsa, hesaplanan
    mesafe MAX_DETECTION_DISTANCE_M'yi aşarsa ya da ışın ufka çok yakın/üstünde
    çıkarsa (dejenere projeksiyon) None döner — çağıran taraf bu tespiti
    atlamalı, bir sonraki (daha düz/merkezi) anı beklemeli. Mesafe sınırı
    özellikle önemli: roll/pitch SIFIR olsa bile frame'in üst kenarına yakın
    pikseller ışını ufka doğru sığ bir açıyla göndermiş oluyor — bu bölgede
    küçük bir attitude hatası mesafeyi/yönü onlarca-yüzlerce metre oynatabiliyor
    (2026-08-12 masa/Gazebo testinde ~150-160m'lik "saçma" sapmalar bulundu,
    bkz. config.py MAX_DETECTION_DISTANCE_M notu).
    """
    global last_reject_reason
    last_reject_reason = None

    log.debug(f"[GEO][pixel_to_gps] Giriş: drone=({drone_lat:.6f},{drone_lon:.6f}) "
          f"alt={alt:.1f}m yaw={yaw_deg:.1f}° roll={roll_deg:.1f}° pitch={pitch_deg:.1f}° "
          f"piksel=({target_cx},{target_cy})")

    if abs(roll_deg) > config.MAX_ROLL_FOR_DETECTION_DEG:
        log.debug(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: |roll|={abs(roll_deg):.1f}° > "
                 f"MAX_ROLL_FOR_DETECTION_DEG={config.MAX_ROLL_FOR_DETECTION_DEG}° — tespit atlanıyor")
        # NOT: HUD (cv2.putText/Hershey font) UTF-8 "°" karakterini render
        # edemiyor ("??" basıyor, 2026-08-26 önizlemede görüldü) — bu yüzden
        # (log.debug'daki tam mesajın aksine) burada derece işareti YOK.
        last_reject_reason = f"ROLL {abs(roll_deg):.0f}>{config.MAX_ROLL_FOR_DETECTION_DEG:.0f}"
        return None

    if abs(pitch_deg) > config.MAX_PITCH_FOR_DETECTION_DEG:
        log.debug(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: |pitch|={abs(pitch_deg):.1f}° > "
                 f"MAX_PITCH_FOR_DETECTION_DEG={config.MAX_PITCH_FOR_DETECTION_DEG}° — tespit atlanıyor")
        last_reject_reason = f"PITCH {abs(pitch_deg):.0f}>{config.MAX_PITCH_FOR_DETECTION_DEG:.0f}"
        return None

    if config.CAMERA_CALIBRATED:
        # Gerçek kalibrasyondan fx/fy/cx/cy ile TAM (yaklaşıksız) pinhole açısı.
        # vision.py, frame'i HSV maskelemeden önce cv2.remap ile zaten distorsiyonu
        # düzeltiyor (aynı camera_matrix'i newCameraMatrix olarak kullanıyor), bu
        # yüzden burada ayrıca undistortPoints çağırmaya gerek yok — target_cx/cy
        # zaten düzeltilmiş frame'den geliyor.
        offset_angle_x = math.atan((target_cx - config.CAMERA_CX) / config.CAMERA_FX)
        offset_angle_y = math.atan((config.CAMERA_CY - target_cy) / config.CAMERA_FY)
    else:
        # Yedek: kalibrasyon yok — doğrusal FOV/piksel yaklaşıklığı (bkz.
        # config.py CAMERA_FOV_H/V notu, kenarlara doğru gerçek açıdan sapar).
        fov_x_rad = math.radians(config.CAMERA_FOV_H)
        fov_y_rad = math.radians(config.CAMERA_FOV_V)
        delta_x_pixel = target_cx - config.WIDTH  / 2
        delta_y_pixel = config.HEIGHT / 2 - target_cy
        offset_angle_x = delta_x_pixel * (fov_x_rad / config.WIDTH)
        offset_angle_y = delta_y_pixel * (fov_y_rad / config.HEIGHT)

    # Kamera kendi bakış (boresight) ekseni etrafında 180° ters monte edilmişse
    # (kablo yönü zorunluluğu, bkz. config.py CAMERA_ROTATION_DEG) capture
    # KASITLI olarak native/rotasyonsuz kalıyor (kalibrasyon değişmesin diye,
    # bkz. pipeline.py, tools/camera_calibrate.py) — bu yüzden ham frame'de
    # sağ/sol ve yukarı/aşağı fiilen ters okunuyor. Matematiksel olarak bu,
    # ışını kameranın kendi X (bakış) ekseni etrafında 180° döndürmeye denk
    # (Rx(180°): Y ve Z işaretini çevirir) — pratikte iki açıyı da negatiflemek
    # yeterli (2026-08-09).
    if config.CAMERA_ROTATION_DEG == 180:
        offset_angle_x = -offset_angle_x
        offset_angle_y = -offset_angle_y

    log.debug(f"[GEO][pixel_to_gps] Açısal ofset: ax={math.degrees(offset_angle_x):.3f}° "
          f"ay={math.degrees(offset_angle_y):.3f}° (kalibreli={config.CAMERA_CALIBRATED}, "
          f"kamera_rotasyon={config.CAMERA_ROTATION_DEG}°)")

    # --- Kamera ışını: kamera-lokal → gövde → dünya (tam 3B, roll+pitch+yaw) ---
    # Kamera-lokal eksende ışın (Xc=bakış ekseni, Yc=sağ, Zc=yukarı-yönlü ofset).
    ray_local = np.array([1.0, math.tan(offset_angle_x), -math.tan(offset_angle_y)])
    ray_local = ray_local / np.linalg.norm(ray_local)

    # Kamera montaj açısı (CAMERA_PITCH, nadir-referanslı sabit) → gövde ekseni.
    mount_theta = math.radians(config.CAMERA_PITCH - 90)
    ray_body = _ry(mount_theta) @ ray_local

    # Gövde → dünya: önce roll (gövde X ekseni), sonra pitch (gövde Y), sonra yaw
    # (dünya Z ekseni) — standart havacılık Euler sırası, ZYX'in tersten uygulanışı.
    #
    # DÜZELTME (2026-08-16): pitch_deg önünde gereksiz eksi işareti vardı —
    # burun yukarı (pitch>0) hedefi YAKINLAŞTIRIYORDU, oysa kamera zaten 45°
    # aşağı bakarken burun kalkınca ufka yaklaşır, hedef UZAKLAŞMALI. Test
    # scriptiyle doğrulandı: pitch=+15°/merkez piksel/roll=0 → düzeltmeden
    # önce 28.87m (yanlış, yakın), düzeltmeden sonra 86.60m (doğru, uzak).
    # 200-300m'lik saha sapmalarının olası kök nedenlerinden biri.
    ray_world = _rz(math.radians(yaw_deg)) @ _ry(math.radians(pitch_deg)) @ _rx(math.radians(roll_deg)) @ ray_body

    if ray_world[2] <= 0.05:
        log.debug(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: ışın ufka çok yakın/üstünde "
                 f"(dünya-aşağı bileşeni={ray_world[2]:.3f}) — dejenere projeksiyon")
        last_reject_reason = "UFUK/DEJENERE"
        return None

    t = alt / ray_world[2]
    delta_north = t * ray_world[0]
    delta_east  = t * ray_world[1]
    log.debug(f"[GEO][pixel_to_gps] Dünya koordinatları: kuzey={delta_north:.2f}m doğu={delta_east:.2f}m")

    horizontal_dist = math.hypot(delta_north, delta_east)
    if horizontal_dist > config.MAX_DETECTION_DISTANCE_M:
        log.debug(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: mesafe={horizontal_dist:.1f}m > "
                 f"MAX_DETECTION_DISTANCE_M={config.MAX_DETECTION_DISTANCE_M}m — sığ açı/dejenere "
                 f"projeksiyon şüphesi, tespit atlanıyor")
        last_reject_reason = f"MESAFE {horizontal_dist:.0f}m>{config.MAX_DETECTION_DISTANCE_M:.0f}m"
        return None

    target_lat = drone_lat + (delta_north / 111320)
    target_lon = drone_lon + (delta_east  / (111320 * math.cos(math.radians(drone_lat))))
    log.debug(f"[GEO][pixel_to_gps] Sonuç: hedef GPS=({target_lat:.6f}, {target_lon:.6f})")

    if config.GEOFENCE_POLYGON and not point_in_polygon(target_lat, target_lon, config.GEOFENCE_POLYGON):
        log.debug(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: hedef GPS=({target_lat:.6f}, {target_lon:.6f}) "
                 f"GEOFENCE_POLYGON dışında — tespit atlanıyor")
        last_reject_reason = "GEOFENCE DISI"  # HUD ASCII-only (bkz. ROLL/PITCH notu)
        return None

    return target_lat, target_lon


def calculate_drop_point(target_lat, target_lon, alt, speed_ms, yaw_deg):
    """
    Balistik hesap: hedefe isabet etmek için yükün bırakılması gereken
    GPS noktasını (release point) döner.
    """
    log.debug(f"[GEO][calculate_drop_point] Hedef: ({target_lat:.6f},{target_lon:.6f}) "
          f"alt={alt:.1f}m hız={speed_ms:.1f}m/s yaw={yaw_deg:.1f}°")

    g                = 9.81
    alt              = max(alt, 1.0)   # yerde test: alt=0 sqrt hatası önle
    fall_time        = math.sqrt(2 * alt / g)
    horizontal_dist  = speed_ms * fall_time
    log.debug(f"[GEO][calculate_drop_point] Düşüş süresi={fall_time:.3f}s | "
          f"yatay kayma (ham balistik)={horizontal_dist:.2f}m")

    # Menzil telafisi (2026-09-01, bkz. config.DROP_RANGE_BIAS_M): saf balistik
    # model sürtünme/mekanizma gecikmesini hesaba katmadığı için TUTARLI olarak
    # hedefin gerisine düşürüyordu (saha ölçümü: ~5m kısa) — release noktasını
    # hedefe bu kadar yaklaştırarak telafi ediyoruz. max(0, ...) çok düşük
    # hız/irtifada (horizontal_dist zaten küçükken) negatife düşmesin diye.
    horizontal_dist = max(0.0, horizontal_dist - config.DROP_RANGE_BIAS_M)
    log.debug(f"[GEO][calculate_drop_point] Telafi sonrası yatay kayma="
              f"{horizontal_dist:.2f}m (bias={config.DROP_RANGE_BIAS_M:.1f}m)")

    yaw_rad      = math.radians(yaw_deg)
    delta_north  = -horizontal_dist * math.cos(yaw_rad)
    delta_east   = -horizontal_dist * math.sin(yaw_rad)
    log.debug(f"[GEO][calculate_drop_point] Geri ofset: kuzey={delta_north:.2f}m doğu={delta_east:.2f}m")

    release_lat = target_lat + (delta_north / 111320)
    release_lon = target_lon + (delta_east  / (111320 * math.cos(math.radians(target_lat))))
    log.debug(f"[GEO][calculate_drop_point] Release point: ({release_lat:.6f}, {release_lon:.6f})")

    return release_lat, release_lon

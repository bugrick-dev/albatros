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


def haversine(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki yüzey mesafesini metre cinsinden döner."""
    R    = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


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
    log.info(f"[GEO][pixel_to_gps] Giriş: drone=({drone_lat:.6f},{drone_lon:.6f}) "
          f"alt={alt:.1f}m yaw={yaw_deg:.1f}° roll={roll_deg:.1f}° pitch={pitch_deg:.1f}° "
          f"piksel=({target_cx},{target_cy})")

    if abs(roll_deg) > config.MAX_ROLL_FOR_DETECTION_DEG:
        log.info(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: |roll|={abs(roll_deg):.1f}° > "
                 f"MAX_ROLL_FOR_DETECTION_DEG={config.MAX_ROLL_FOR_DETECTION_DEG}° — tespit atlanıyor")
        return None

    if abs(pitch_deg) > config.MAX_PITCH_FOR_DETECTION_DEG:
        log.info(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: |pitch|={abs(pitch_deg):.1f}° > "
                 f"MAX_PITCH_FOR_DETECTION_DEG={config.MAX_PITCH_FOR_DETECTION_DEG}° — tespit atlanıyor")
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

    log.info(f"[GEO][pixel_to_gps] Açısal ofset: ax={math.degrees(offset_angle_x):.3f}° "
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
        log.info(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: ışın ufka çok yakın/üstünde "
                 f"(dünya-aşağı bileşeni={ray_world[2]:.3f}) — dejenere projeksiyon")
        return None

    t = alt / ray_world[2]
    delta_north = t * ray_world[0]
    delta_east  = t * ray_world[1]
    log.info(f"[GEO][pixel_to_gps] Dünya koordinatları: kuzey={delta_north:.2f}m doğu={delta_east:.2f}m")

    horizontal_dist = math.hypot(delta_north, delta_east)
    if horizontal_dist > config.MAX_DETECTION_DISTANCE_M:
        log.info(f"[GEO][pixel_to_gps] ⚠ REDDEDİLDİ: mesafe={horizontal_dist:.1f}m > "
                 f"MAX_DETECTION_DISTANCE_M={config.MAX_DETECTION_DISTANCE_M}m — sığ açı/dejenere "
                 f"projeksiyon şüphesi, tespit atlanıyor")
        return None

    target_lat = drone_lat + (delta_north / 111320)
    target_lon = drone_lon + (delta_east  / (111320 * math.cos(math.radians(drone_lat))))
    log.info(f"[GEO][pixel_to_gps] Sonuç: hedef GPS=({target_lat:.6f}, {target_lon:.6f})")

    return target_lat, target_lon


def calculate_drop_point(target_lat, target_lon, alt, speed_ms, yaw_deg):
    """
    Balistik hesap: hedefe isabet etmek için yükün bırakılması gereken
    GPS noktasını (release point) döner.
    """
    log.info(f"[GEO][calculate_drop_point] Hedef: ({target_lat:.6f},{target_lon:.6f}) "
          f"alt={alt:.1f}m hız={speed_ms:.1f}m/s yaw={yaw_deg:.1f}°")

    g                = 9.81
    alt              = max(alt, 1.0)   # yerde test: alt=0 sqrt hatası önle
    fall_time        = math.sqrt(2 * alt / g)
    horizontal_dist  = speed_ms * fall_time
    log.info(f"[GEO][calculate_drop_point] Düşüş süresi={fall_time:.3f}s | "
          f"yatay kayma={horizontal_dist:.2f}m")

    yaw_rad      = math.radians(yaw_deg)
    delta_north  = -horizontal_dist * math.cos(yaw_rad)
    delta_east   = -horizontal_dist * math.sin(yaw_rad)
    log.info(f"[GEO][calculate_drop_point] Geri ofset: kuzey={delta_north:.2f}m doğu={delta_east:.2f}m")

    release_lat = target_lat + (delta_north / 111320)
    release_lon = target_lon + (delta_east  / (111320 * math.cos(math.radians(target_lat))))
    log.info(f"[GEO][calculate_drop_point] Release point: ({release_lat:.6f}, {release_lon:.6f})")

    return release_lat, release_lon

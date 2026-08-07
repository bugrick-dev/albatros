"""Coğrafi hesaplamalar: Haversine mesafesi, piksel→GPS dönüşümü, balistik drop point."""
import logging
import math
import config

log = logging.getLogger("geo")


def haversine(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki yüzey mesafesini metre cinsinden döner."""
    R    = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def pixel_to_gps(drone_lat, drone_lon, alt, yaw_deg, target_cx, target_cy):
    """
    Kamera frame piksel koordinatlarını GPS koordinatına dönüştürür.
    Kamera eğimi (CAMERA_PITCH) ve FOV değerleri config'den alınır.
    """
    log.info(f"[GEO][pixel_to_gps] Giriş: drone=({drone_lat:.6f},{drone_lon:.6f}) "
          f"alt={alt:.1f}m yaw={yaw_deg:.1f}° piksel=({target_cx},{target_cy})")

    camera_pitch_rad = math.radians(config.CAMERA_PITCH)
    fov_x_rad        = math.radians(config.CAMERA_FOV_H)
    fov_y_rad        = math.radians(config.CAMERA_FOV_V)

    delta_x_pixel = target_cx - config.WIDTH  / 2
    delta_y_pixel = config.HEIGHT / 2 - target_cy
    log.info(f"[GEO][pixel_to_gps] Merkeze göre piksel ofseti: dx={delta_x_pixel:.1f} dy={delta_y_pixel:.1f}")

    offset_angle_x = delta_x_pixel * (fov_x_rad / config.WIDTH)
    offset_angle_y = delta_y_pixel * (fov_y_rad / config.HEIGHT)
    log.info(f"[GEO][pixel_to_gps] Açısal ofset: ax={math.degrees(offset_angle_x):.3f}° "
          f"ay={math.degrees(offset_angle_y):.3f}°")

    total_pitch = camera_pitch_rad + offset_angle_y
    dist_y      = alt * math.tan(total_pitch)
    # NOT: cos(offset_angle_y) çarpanı olmadan dist_x, hedef dikey merkezden
    # uzaklaştıkça (üstte/altta) 1/cos(offset_angle_y) oranında ŞİŞER — tam
    # pinhole projeksiyonunda dist_x = alt*tan(ox)*cos(oy)/cos(total_pitch)
    # (2026-08-07: sahada ölçülen sistematik hata sonrası bulundu/düzeltildi).
    dist_x      = (alt / math.cos(total_pitch)) * math.tan(offset_angle_x) * math.cos(offset_angle_y)
    log.info(f"[GEO][pixel_to_gps] Zemine projeksiyon: ileri={dist_y:.2f}m sağ={dist_x:.2f}m")

    yaw_rad      = math.radians(yaw_deg)
    delta_north  = (dist_y * math.cos(yaw_rad)) - (dist_x * math.sin(yaw_rad))
    delta_east   = (dist_y * math.sin(yaw_rad)) + (dist_x * math.cos(yaw_rad))
    log.info(f"[GEO][pixel_to_gps] Dünya koordinatları: kuzey={delta_north:.2f}m doğu={delta_east:.2f}m")

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

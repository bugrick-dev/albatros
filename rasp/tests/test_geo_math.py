#!/usr/bin/env python3
"""
GEO MATEMATİK TESTİ — donanım gerektirmez (FC/kamera/Gazebo YOK), saf
geo.py + config.py matematiğini doğrular. `pytest` ile ya da doğrudan
`python3 test_geo_math.py` ile çalıştırılabilir.

Amaç: 2026-08-12'de bulunan "saçma yönlerde 150-160m sapma" hatasını
(sığ açı / grazing-angle büyütmesi — bkz. geo.pixel_to_gps ve config.py
MAX_DETECTION_DISTANCE_M/MAX_PITCH_FOR_DETECTION_DEG notları) somut
sayılarla sabitlemek: reddedilmesi gereken durumların GERÇEKTEN
reddedildiğini, makul durumların GERÇEKTEN kabul edildiğini garantiye
almak. Gazebo/SITL testleri (uçuş kontrolcüsü + gerçek kamera akışı)
bunun YERİNE değil, EK olarak yapılmalı — bu test yalnızca izole
matematiği doğrular, telemetri gecikmesi/gürültüsü gibi sistem
seviyesindeki etkileri kapsamaz.

Çalıştırma:
    cd rasp/tests
    python3 test_geo_math.py
    # veya: pytest test_geo_math.py -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import geo


def _dist_m(lat0, lon0, lat1, lon1):
    return geo.haversine(lat0, lon0, lat1, lon1)


def test_center_pixel_matches_camera_pitch_geometry():
    """
    Boresight piksel = optik merkez. roll=pitch=yaw=0 iken beklenen mesafe
    alt/tan(CAMERA_PITCH) olmalı (CAMERA_PITCH nadir-referanslı: 90°=düz
    aşağı, 0°=ufka paralel).

    NOT (2026-08-19): eskiden "merkez piksel" = geometrik kare merkezi
    (WIDTH/2, HEIGHT/2) sayılıyordu — bu yalnızca kalibrasyonsuz FOV
    yaklaşıklığında doğru. Gerçek kalibrasyon (camera_calib.json) optik
    merkezi cx=309.9/cy=208.3 veriyor, geometrik merkezden (320/240) ~10px
    sol ~32px yukarıda — pixel_to_gps CAMERA_CALIBRATED iken haklı olarak
    CAMERA_CX/CY'yi boresight sayıyor (geo.py offset_angle_x/y hesabı), bu
    yüzden test de aynı referansı kullanmalı; aksi halde kalibrasyonun kendi
    doğru davranışı "hata" gibi görünüyordu (got=64.15m != expected=60.00m).
    """
    alt = 60.0
    if config.CAMERA_CALIBRATED:
        boresight_cx, boresight_cy = config.CAMERA_CX, config.CAMERA_CY
    else:
        boresight_cx, boresight_cy = config.WIDTH / 2, config.HEIGHT / 2
    lat, lon = geo.pixel_to_gps(0.0, 0.0, alt, 0.0, 0.0, 0.0,
                                 boresight_cx, boresight_cy)
    expected_dist = alt / math.tan(math.radians(config.CAMERA_PITCH))
    got_dist = _dist_m(0.0, 0.0, lat, lon)
    assert abs(got_dist - expected_dist) < 0.5, (
        f"Boresight piksel mesafesi beklenenden sapıyor: got={got_dist:.2f}m "
        f"expected={expected_dist:.2f}m"
    )


def test_shallow_angle_pixel_is_rejected():
    """
    Işını gerçekten ufka yaklaştıran kombinasyon (üst kenar pikseli + limit
    içinde burun-yukarı pitch) MAX_DETECTION_DISTANCE_M reddine takılmalı.

    NOT (2026-08-17): test eskiden pitch=0 ile üst kenarı "sığ" sayıyordu —
    bu, kalibrasyonsuz geniş FOV (48.8°) varsayımına göre yazılmıştı. Gerçek
    kalibrasyonla (VFOV ~27.5°) üst kenar 45-14.6≈30° depresyonda kalıyor ve
    60m irtifada ~102m'ye projekte oluyor — bu GEÇERLİ bir tespittir,
    reddedilmemesi doğrudur. Sığ açı ancak attitude ile oluşur: pitch=+20°
    (MAX_PITCH sınırının içinde) üst kenar ışını ~10° depresyona indirir →
    ~330m > MAX_DETECTION_DISTANCE_M → red. Kalibrasyonsuz yedek FOV yolunda
    da aynı kombinasyon ufka çıkar (45-24.4-20≈0.6°) → yine red.
    """
    alt = 60.0
    shallow_cy = config.HEIGHT if config.CAMERA_ROTATION_DEG == 180 else 0
    nose_up = config.MAX_PITCH_FOR_DETECTION_DEG - 10.0  # limit İÇİNDE kalır
    result = geo.pixel_to_gps(0.0, 0.0, alt, 0.0, 0.0, nose_up,
                               config.WIDTH / 2, shallow_cy)
    assert result is None, (
        "Sığ açı kombinasyonu (üst kenar + burun-yukarı pitch) reddedilmedi — "
        "MAX_DETECTION_DISTANCE_M kontrolü beklenmedik şekilde geçti"
    )


def test_pitch_limit_rejects_beyond_threshold():
    over = config.MAX_PITCH_FOR_DETECTION_DEG + 5.0
    result = geo.pixel_to_gps(0.0, 0.0, 60.0, 0.0, 0.0, over,
                               config.WIDTH / 2, config.HEIGHT / 2)
    assert result is None, "MAX_PITCH_FOR_DETECTION_DEG aşımı reddedilmedi"


def test_pitch_within_threshold_is_accepted():
    under = config.MAX_PITCH_FOR_DETECTION_DEG - 5.0
    result = geo.pixel_to_gps(0.0, 0.0, 60.0, 0.0, 0.0, under,
                               config.WIDTH / 2, config.HEIGHT / 2)
    assert result is not None, (
        "MAX_PITCH_FOR_DETECTION_DEG sınırı içindeki pitch yanlışlıkla reddedildi"
    )


def test_roll_limit_still_rejects_beyond_threshold():
    """Mevcut roll reddi (2026-08-08) regresyona uğramamalı."""
    over = config.MAX_ROLL_FOR_DETECTION_DEG + 5.0
    result = geo.pixel_to_gps(0.0, 0.0, 60.0, 0.0, over, 0.0,
                               config.WIDTH / 2, config.HEIGHT / 2)
    assert result is None, "MAX_ROLL_FOR_DETECTION_DEG aşımı reddedilmedi"


def test_computed_distance_never_exceeds_configured_max():
    """
    Kabul edilen HER sonuç için: gerçek mesafe MAX_DETECTION_DISTANCE_M
    içinde kalmalı — geniş bir piksel/attitude taramasıyla kontrol eder.
    """
    alt = 60.0
    checked_any = False
    for cy in range(0, config.HEIGHT, 20):
        for cx in range(0, config.WIDTH, 40):
            for pitch in (-10.0, 0.0, 10.0):
                result = geo.pixel_to_gps(0.0, 0.0, alt, 0.0, 0.0, pitch, cx, cy)
                if result is None:
                    continue
                checked_any = True
                dist = _dist_m(0.0, 0.0, *result)
                assert dist <= config.MAX_DETECTION_DISTANCE_M + 1e-6, (
                    f"Kabul edilen sonuç sınırı aşıyor: piksel=({cx},{cy}) "
                    f"pitch={pitch}° mesafe={dist:.1f}m > "
                    f"{config.MAX_DETECTION_DISTANCE_M}m"
                )
    assert checked_any, "Hiçbir kombinasyon kabul edilmedi — test anlamsız kaldı"


def test_bearing_deg_cardinal_directions():
    """bearing_deg (2026-08-17, drop along/cross ayrışımı için eklendi):
    kuzey=0°, doğu=90°, güney=±180°, batı=-90° civarı dönmeli."""
    lat, lon = 39.0, 32.0
    d = 0.001
    assert abs(geo.bearing_deg(lat, lon, lat + d, lon)) < 0.5, "kuzey ≈ 0° değil"
    assert abs(geo.bearing_deg(lat, lon, lat, lon + d) - 90.0) < 0.5, "doğu ≈ 90° değil"
    assert abs(abs(geo.bearing_deg(lat, lon, lat - d, lon)) - 180.0) < 0.5, "güney ≈ ±180° değil"
    assert abs(geo.bearing_deg(lat, lon, lat, lon - d) + 90.0) < 0.5, "batı ≈ -90° değil"


def test_wrap180_folds_into_range():
    assert geo.wrap180(190.0) == -170.0
    assert geo.wrap180(-190.0) == 170.0
    assert geo.wrap180(0.0) == 0.0
    assert geo.wrap180(180.0) == -180.0  # aralık [-180, 180): 180 → -180 (eşdeğer)
    assert geo.wrap180(540.0) == -180.0


def test_drop_point_is_upcourse_of_target():
    """Release noktası, hedefin rota yönünün GERİSİNDE olmalı (yük ileri
    savrulur) ve geri ofset (ham balistik hız*düşüş süresi EKSİ
    config.DROP_RANGE_BIAS_M menzil telafisi, bkz. geo.calculate_drop_point
    2026-09-01) kadar olmalı."""
    alt, speed, course = 60.0, 10.0, 0.0   # kuzeye uçuş
    tlat, tlon = 39.0, 32.0
    rlat, rlon = geo.calculate_drop_point(tlat, tlon, alt, speed, course)
    assert rlat < tlat, "release noktası hedefin kuzeyinde (ilerisinde) çıktı"
    expected = speed * math.sqrt(2 * alt / 9.81) - config.DROP_RANGE_BIAS_M
    got = geo.haversine(tlat, tlon, rlat, rlon)
    assert abs(got - expected) < 0.5, f"geri ofset {got:.2f}m != beklenen {expected:.2f}m"


def test_drop_point_headwind_increases_range_compensation():
    """Ters rüzgar (headwind), itkisiz düşen yükü yer hızına göre daha erken
    yavaşlatır → release noktası hedefe DAHA da yaklaşmalı (bkz.
    config.DROP_WIND_ALONG_GAIN, 2026-09-02: 17km/h rüzgarda saha ölçümü
    ~6-7m ek kısa düşme göstermişti)."""
    alt, speed, course = 25.0, 13.0, 0.0   # kuzeye uçuş
    tlat, tlon = 39.0, 32.0
    no_wind = geo.calculate_drop_point(tlat, tlon, alt, speed, course)
    fall_time = math.sqrt(2 * alt / 9.81)
    headwind_n = -4.72   # ~17km/h, tam ters yönden (güneyden) esiyor
    with_headwind = geo.calculate_drop_point(tlat, tlon, alt, speed, course,
                                              wind_n=headwind_n, wind_e=0.0)
    d0 = geo.haversine(tlat, tlon, *no_wind)
    d1 = geo.haversine(tlat, tlon, *with_headwind)
    expected_extra = 4.72 * fall_time * config.DROP_WIND_ALONG_GAIN
    got_extra = d0 - d1
    assert got_extra > 0, "ters rüzgarda release noktası hedefe yaklaşmadı"
    assert abs(got_extra - expected_extra) < 0.1, (
        f"ek telafi {got_extra:.2f}m != beklenen {expected_extra:.2f}m")


def test_drop_point_crosswind_shifts_upwind():
    """Yanal rüzgar, düşen yükü rotanın rüzgar yönüne sürükler → release
    noktası bunun TERSİ yöne (rüzgarın estiği tarafa) kaymalı, böylece
    sürüklenme hedefe geri getirsin (2026-09-02, DROP_WIND_CROSS_GAIN)."""
    alt, speed, course = 25.0, 13.0, 0.0   # kuzeye uçuş, sağ=doğu
    tlat, tlon = 39.0, 32.0
    # Doğuya esen rüzgar (wind_e > 0) yükü sağa (doğuya) sürükler.
    rlat, rlon = geo.calculate_drop_point(tlat, tlon, alt, speed, course,
                                           wind_n=0.0, wind_e=4.72)
    assert rlon < tlon, ("doğuya esen rüzgarda release noktası hedefin "
                          "batısına kaymalıydı (sürüklenmeyi telafi etmek için)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test geçti")
    sys.exit(1 if failed else 0)

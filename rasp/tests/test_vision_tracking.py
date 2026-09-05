#!/usr/bin/env python3
"""
HEDEF İZLEME (en-yakın-nokta kilidi) TESTİ — donanım gerektirmez (FC/Gazebo
YOK, gerçek sentetik maskeler üretilir). vision._update_detection'ın
2026-08-12'de eklenen izleme mantığını doğrular: GPS kilidi artık hedefin
İLK görüldüğü karede değil, en yakın (en düşük hatalı) karede yapılıyor —
bkz. config.py DETECTION_CONFIRM_STREAK/DETECTION_LOST_STREAK/
DETECTION_TRACK_MAX_SEC ve MAX_DETECTION_DISTANCE_M notları.

Çalıştırma:
    cd rasp/tests
    python3 test_vision_tracking.py
"""
import sys
from pathlib import Path
from queue import Queue

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import state
import vision

# 2026-09-05 (yarış öncesi inceleme): bu dosya saf, GPS-değerinden BAĞIMSIZ
# izleme MANTIĞINI test ediyor (kilit anı, kayıp/köprüleme) — geofence'i
# DEĞİL. LEVEL_TEL'in sentetik (0,0) konumu, 2026-08-24'te eklenen
# GEOFENCE_POLYGON'un (gerçek, küçük, İstanbul'daki bir alan) ÇOK dışında
# kaldığından _update_detection'ın içindeki pixel_to_gps çağrısı HER karede
# sessizce reddediyordu — tracking[color] hiç dolmuyor, testler yanlış
# sebeple (geofence, izleme mantığı DEĞİL) başarısız oluyordu. Gerçek
# geofence davranışı test_geo_math.py'de kendi (gerçek) poligonuyla ayrıca
# test ediliyor; burada devre dışı bırakmak doğru — bu dosyanın konusu o
# değil.
config.GEOFENCE_POLYGON = []

LEVEL_TEL = {"lat": 0.0, "lon": 0.0, "alt": 60.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0, "speed": 10.0}


def _square_mask(cx, cy, size=20):
    mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), 255, -1)
    return mask


def _frame_from_mask(mask):
    """_update_detection artık MOSSE tracker init/update için gerçek bir BGR
    frame istiyor (2026-08-19) — sentetik testlerde kamera yok, maskenin
    kendisini gri→BGR çevirip 'görüntü' olarak kullanıyoruz. NOT: bu düz
    (dokusuz) içerik tracker'ın PSR güven eşiğini GEÇMEZ (bkz. _textured_frame
    notu) — testlerde bu YETERLİ çünkü bu yardımcı yalnızca köprüleme
    TETİKLENMEYEN testlerde kullanılıyor (tracker.update() zaten ok=False
    dönüp normal miss_streak akışına düşüyor, eski davranışla birebir aynı)."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _pink_noise(h, w, seed):
    """1/f (pembe gürültü) doku — MOSSE'nin PSR güven kontrolü düz/beyaz
    gürültü içerikte HER ZAMAN reddediyor (2026-08-19 saha dışı test
    sırasında bulundu: gerçek fotoğrafta çalışıyor, sentetik düz kare/beyaz
    gürültüde çalışmıyordu) — pembe gürültü doğal görüntülerin spektral
    istatistiğine (1/f) daha yakın, tracker'ı gerçekçi biçimde eğitebiliyor."""
    rng = np.random.RandomState(seed)
    white = rng.randn(h, w)
    f = np.fft.fft2(white)
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    freq = np.sqrt(fy ** 2 + fx ** 2)
    freq[0, 0] = 1e-6
    img = np.real(np.fft.ifft2(f / freq))
    img -= img.min()
    img /= img.max()
    return (img * 255).astype(np.uint8)


def _textured_frame(cx, cy, size, seed=7):
    """Tracker-özel testler için MOSSE'nin GERÇEKTEN kilitlenebildiği bir
    frame üretir: pembe gürültü zemin + kontrastı artırılmış hedef yaması
    (`size`, HSV maskesindeki GERÇEK hedef boyutuyla eşleşir — küçük de
    olabilir, büyük de).

    NOT (2026-08-19, saha dışı ölçüldü): bu cv2.legacy.TrackerMOSSE derlemesi
    ~40px altındaki DÜZ/dokusuz yamalarda güvenilir kilitlenmiyor. Üretim
    kodu bunu vision._pad_bbox_to_min_size ile çözüyor: tracker'a ham
    (küçük olabilecek) tespit bbox'ı değil, hedefin merkezini koruyarak
    config.TRACKER_MIN_PATCH_PX'e (80px) padding'lenmiş bir yama veriliyor
    — bu fonksiyonun ürettiği zemin HER YERDE pembe gürültü olduğundan
    (yalnızca `size`x`size` iç bölge kontrastı artırılmış), küçük bir
    `size` verilse bile üretim kodunun eklediği padding gerçek dokulu bir
    alana denk gelir — tıpkı sahada olacağı gibi (bkz.
    test_tracker_bridges_small_far_target_via_padding).
    """
    frame_gray = _pink_noise(config.HEIGHT, config.WIDTH, seed=seed)
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    patch = frame_gray[y0:y1, x0:x1].astype(np.int16)
    frame_gray[y0:y1, x0:x1] = np.clip(patch + 80, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)


def _new_trackers():
    return {"mavi": None, "kirmizi": None}


def test_lock_happens_at_closest_pass_not_first_sight():
    """
    Hedef frame'i baştan sona (kenardan kenara, merkezden geçerek) kat
    ediyor — kilit, ilk görülen (kenar, en hatalı) karede DEĞİL, merkeze en
    yakın karede olmalı.
    """
    state.detection_active.set()
    queue, queued_colors, tracking, trackers = Queue(), set(), {"mavi": None, "kirmizi": None}, _new_trackers()

    for x in range(50, config.WIDTH - 50, 15):
        mask = _square_mask(x, config.HEIGHT // 2)
        vision._update_detection(mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking,
                                  _frame_from_mask(mask), trackers)
        if "mavi" in queued_colors:
            break

    assert "mavi" in queued_colors, "Hedef hiç kilitlenmedi"
    assert queue.qsize() == 1
    locked = queue.get()
    # Beklenen "en yakın" mesafe: alt/tan(CAMERA_PITCH) formülü DEĞİL —
    # bu satır y=HEIGHT//2'de sabit taranıyor, gerçek kalibrasyon boresight'ı
    # (config.CAMERA_CY=208.3) bu satırda değil (2026-08-19, bkz. test_geo_math.py
    # test_center_pixel_matches_camera_pitch_geometry notu). Bu yüzden beklenen
    # değer, aynı taranan x'ler üzerinde geo.pixel_to_gps'in ürettiği GERÇEK
    # minimum mesafeden hesaplanıyor — kalibrasyon merkezi nerede olursa olsun
    # test kendi kendine tutarlı kalır, ilk görülen kenar pikselinden ÇOK daha
    # yakın olmalı iddiası hâlâ doğrulanır.
    import geo
    candidate_dists = []
    for x in range(50, config.WIDTH - 50, 15):
        result = geo.pixel_to_gps(0.0, 0.0, LEVEL_TEL["alt"], 0.0, 0.0, 0.0, x, config.HEIGHT // 2)
        if result is not None:
            candidate_dists.append(geo.haversine(0.0, 0.0, *result))
    expected_center_dist = min(candidate_dists)
    got_dist = geo.haversine(0.0, 0.0, locked["lat"], locked["lon"])
    assert abs(got_dist - expected_center_dist) < 1.0, (
        f"Kilit merkez geçişine değil, başka bir kareye denk geldi: "
        f"got={got_dist:.2f}m expected≈{expected_center_dist:.2f}m"
    )


def test_shallow_only_target_never_locks():
    """
    Hedef YALNIZCA sığ açıya denk gelen kenarda görünüyor, hiç merkeze
    yaklaşmıyorsa — MAX_DETECTION_DISTANCE_M reddi sürekli tetiklenmeli,
    hedef hiçbir zaman (yanlış bir koordinatla) kilitlenmemeli.
    """
    state.detection_active.set()
    queue, queued_colors, tracking, trackers = Queue(), set(), {"mavi": None, "kirmizi": None}, _new_trackers()
    shallow_cy = config.HEIGHT if config.CAMERA_ROTATION_DEG == 180 else 0

    for _ in range(30):
        mask = _square_mask(config.WIDTH // 2, shallow_cy)
        vision._update_detection(mask, "kirmizi", queue, queued_colors, LEVEL_TEL, tracking,
                                  _frame_from_mask(mask), trackers)

    assert "kirmizi" not in queued_colors, "Sığ açı hedefi yanlışlıkla kilitlendi"
    assert queue.empty()


def test_target_lost_from_frame_locks_best_seen_so_far():
    """
    Hedef bir süre izlendikten sonra kadraj dışına çıkarsa (art arda
    DETECTION_LOST_STREAK kare boyunca hiç görünmezse), o ana kadarki en
    iyi örnekle hemen kilitlenmeli — sonsuza kadar beklenmemeli.
    """
    state.detection_active.set()
    queue, queued_colors, tracking, trackers = Queue(), set(), {"mavi": None, "kirmizi": None}, _new_trackers()

    for x in (300, 310, 320):
        mask = _square_mask(x, config.HEIGHT // 2)
        vision._update_detection(mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking,
                                  _frame_from_mask(mask), trackers)
    assert "mavi" not in queued_colors, "Henüz kadraj dışına çıkmadan kilitlendi"

    # NOT (2026-08-19): boş kare gerçekten hedefsiz (MOSSE şablonuyla hiç
    # örtüşmüyor) olduğu için tracker.update() hemen ok=False döner (ölçüldü)
    # — köprüleme bütçesini TÜKETMEZ, miss_streak eskisi gibi DETECTION_LOST_STREAK
    # karede dolar. Gerçek kamerada da aynı: hedef TAMAMEN kadraj dışına
    # çıktığında MOSSE'nin PSR/güven eşiği düşer, sahte köprüleme yapmaz.
    empty_mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    empty_frame = _frame_from_mask(empty_mask)
    for _ in range(config.DETECTION_LOST_STREAK):
        vision._update_detection(empty_mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking,
                                  empty_frame, trackers)

    assert "mavi" in queued_colors, "Kadraj dışına çıkınca kilitlenmedi"
    assert queue.qsize() == 1


def test_tracker_bridges_single_frame_color_dropout():
    """
    MOSSE köprüleme (2026-08-19, bkz. config.py TRACKER_MAX_BRIDGE_FRAMES):
    hedef fiziksel olarak kadrajda kalmaya devam ederken (frame/BGR görüntü
    AYNI) HSV/kontur tespiti TEK bir karede (renk eşiği kaçırması, motion
    blur vb.) başarısız olursa miss_streak ARTMAMALI ve HUD/overlay o karede
    de hedefi göstermeye devam etmeli.
    """
    state.detection_active.set()
    queue, queued_colors, tracking, trackers = Queue(), set(), {"mavi": None, "kirmizi": None}, _new_trackers()

    # Büyük (80px) yama KASITLI — bkz. _textured_frame notu: MOSSE küçük/uzak
    # tespitlerde güvenilir kilitlenmiyor, bu test yalnızca köprüleme
    # MEKANİZMASININ kendisini doğruluyor.
    mask = _square_mask(200, config.HEIGHT // 2, size=80)
    frame = _textured_frame(200, config.HEIGHT // 2, 80)
    vision._update_detection(mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)
    assert state.detected_targets["mavi"] is not None
    miss_before = tracking["mavi"]["miss_streak"]

    # Aynı frame (hedef hâlâ görüntüde) ama HSV maskesi bu sefer boş —
    # renk tespitinin TEK karede kaçırması simülasyonu.
    empty_mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    vision._update_detection(empty_mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)

    assert state.detected_targets["mavi"] is not None, (
        "Tracker köprülemedi — tek karelik HSV kaçırmasında HUD hedefi kaybetti"
    )
    assert tracking["mavi"]["miss_streak"] == miss_before, (
        "Köprülenen kare miss_streak'i artırdı — erken 'kadraj dışı' riskini büyütür"
    )
    assert "mavi" not in queued_colors


def test_tracker_bridge_budget_caps_then_resumes_normal_miss_tracking():
    """
    Tracker sürekli (ok=True) eşleşse bile TRACKER_MAX_BRIDGE_FRAMES'i aşan
    art arda HSV kaçırmalarında normal miss_streak sayımına düşülmeli —
    aksi halde MOSSE sürüklenirse hedef asla 'kadraj dışı' finalize
    edilemez (bkz. config.py TRACKER_MAX_BRIDGE_FRAMES notu).
    """
    state.detection_active.set()
    queue, queued_colors, tracking, trackers = Queue(), set(), {"mavi": None, "kirmizi": None}, _new_trackers()

    mask = _square_mask(200, config.HEIGHT // 2, size=80)
    frame = _textured_frame(200, config.HEIGHT // 2, 80)
    vision._update_detection(mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)

    empty_mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    # AYNI frame'i vermeye devam et — MOSSE her seferinde eşleşecek (ok=True),
    # yani köprüleme bütçesi tükenene kadar miss_streak ARTMAMALI.
    for i in range(config.TRACKER_MAX_BRIDGE_FRAMES):
        vision._update_detection(empty_mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)
        assert tracking["mavi"]["miss_streak"] == 0, f"{i}. köprü karesinde miss_streak beklenmedik arttı"

    # Bütçe tükendi — bundan sonraki kaçırmalar gerçek miss_streak'e düşmeli.
    for _ in range(config.DETECTION_LOST_STREAK):
        vision._update_detection(empty_mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)

    assert "mavi" in queued_colors, "Bütçe tükendikten sonra kadraj dışı finalize edilmedi"


def test_tracker_bridges_small_far_target_via_padding():
    """
    İrtifa değişimlerine dayanıklılık (2026-08-19): hedefin GÖRÜNÜR boyutu
    yüksek irtifada/uzak tespitte KÜÇÜK olabilir (~20px, config.MIN_AREA'ya
    yakın) — MOSSE bu boyutta düz/dokusuz bir yamada güvenilir kilitlenmiyor
    ama vision._pad_bbox_to_min_size ham bbox'ı TRACKER_MIN_PATCH_PX'e
    padding'leyip çevresindeki gerçek dokuyu da tracker'a veriyor. Bu test,
    yalnızca büyük/yakın tespitlerde değil KÜÇÜK tespitte de köprülemenin
    çalıştığını doğruluyor.
    """
    state.detection_active.set()
    queue, queued_colors, tracking, trackers = Queue(), set(), {"mavi": None, "kirmizi": None}, _new_trackers()

    small_size = 20  # config.MIN_AREA (~307px²) civarı — sqrt(307)≈18px
    mask = _square_mask(200, config.HEIGHT // 2, size=small_size)
    frame = _textured_frame(200, config.HEIGHT // 2, small_size)
    vision._update_detection(mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)
    assert state.detected_targets["mavi"] is not None
    miss_before = tracking["mavi"]["miss_streak"]

    empty_mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    vision._update_detection(empty_mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking, frame, trackers)

    assert state.detected_targets["mavi"] is not None, (
        "Küçük (uzak/yüksek irtifa) hedefte padding'siz köprüleme başarısız oldu"
    )
    assert tracking["mavi"]["miss_streak"] == miss_before, (
        "Küçük hedefte köprülenen kare miss_streak'i artırdı"
    )
    assert "mavi" not in queued_colors


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        # Her testten önce paylaşılan state.detection_active'i sıfırla/kur.
        state.detection_active.clear()
        try:
            t()
            print(f"✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test geçti")
    sys.exit(1 if failed else 0)

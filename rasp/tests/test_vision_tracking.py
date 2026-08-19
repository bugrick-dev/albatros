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

LEVEL_TEL = {"lat": 0.0, "lon": 0.0, "alt": 60.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0, "speed": 10.0}


def _square_mask(cx, cy, size=20):
    mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), 255, -1)
    return mask


def test_lock_happens_at_closest_pass_not_first_sight():
    """
    Hedef frame'i baştan sona (kenardan kenara, merkezden geçerek) kat
    ediyor — kilit, ilk görülen (kenar, en hatalı) karede DEĞİL, merkeze en
    yakın karede olmalı.
    """
    state.detection_active.set()
    queue, queued_colors, tracking = Queue(), set(), {"mavi": None, "kirmizi": None}

    for x in range(50, config.WIDTH - 50, 15):
        vision._update_detection(_square_mask(x, config.HEIGHT // 2), "mavi",
                                  queue, queued_colors, LEVEL_TEL, tracking)
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
    queue, queued_colors, tracking = Queue(), set(), {"mavi": None, "kirmizi": None}
    shallow_cy = config.HEIGHT if config.CAMERA_ROTATION_DEG == 180 else 0

    for _ in range(30):
        vision._update_detection(_square_mask(config.WIDTH // 2, shallow_cy), "kirmizi",
                                  queue, queued_colors, LEVEL_TEL, tracking)

    assert "kirmizi" not in queued_colors, "Sığ açı hedefi yanlışlıkla kilitlendi"
    assert queue.empty()


def test_target_lost_from_frame_locks_best_seen_so_far():
    """
    Hedef bir süre izlendikten sonra kadraj dışına çıkarsa (art arda
    DETECTION_LOST_STREAK kare boyunca hiç görünmezse), o ana kadarki en
    iyi örnekle hemen kilitlenmeli — sonsuza kadar beklenmemeli.
    """
    state.detection_active.set()
    queue, queued_colors, tracking = Queue(), set(), {"mavi": None, "kirmizi": None}

    for x in (300, 310, 320):
        vision._update_detection(_square_mask(x, config.HEIGHT // 2), "mavi",
                                  queue, queued_colors, LEVEL_TEL, tracking)
    assert "mavi" not in queued_colors, "Henüz kadraj dışına çıkmadan kilitlendi"

    empty_mask = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.uint8)
    for _ in range(config.DETECTION_LOST_STREAK):
        vision._update_detection(empty_mask, "mavi", queue, queued_colors, LEVEL_TEL, tracking)

    assert "mavi" in queued_colors, "Kadraj dışına çıkınca kilitlenmedi"
    assert queue.qsize() == 1


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

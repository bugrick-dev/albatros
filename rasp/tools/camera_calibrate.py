#!/usr/bin/env python3
"""
Kamera kalibrasyonu — tek seferlik çalıştırılır.

Kullanım (Pi üzerinde):
    sudo /home/albatros/venv/bin/python3 rasp/tools/camera_calibrate.py

Yapman gereken TEK şey: script başladıktan sonra satranç tahtası desenini
(otomatik oluşturulup rasp/tools/chessboard_pattern.png olarak kaydedilir —
A4 kağıda düz/kırışıksız bas, SERT bir yüzeye (karton/tahta) yapıştır)
kameranın önünde YAVAŞÇA hareket ettirmek: farklı açılardan, farklı
mesafelerden, kadrajın farklı köşelerinden göster. Script yeterli sayıda
iyi görüntü (varsayılan 20) toplayınca otomatik durur ve hesaplamayı yapar.

Ne yapar:
  1. albatros.service çalışıyorsa geçici olarak durdurur (kamerayı tek bir
     süreç kullanabilir — rpicam-vid ile çakışmasın diye), script bitince
     (başarılı/başarısız/Ctrl+C fark etmez) eski haline geri döndürür.
  2. Satranç tahtası deseni yoksa üretir (chessboard_pattern.png).
  3. Picamera2 ile canlı kare akışından satranç tahtası köşelerini otomatik
     tespit eder, yeterince FARKLI pozlardan CHESSBOARD_COUNT adet iyi
     görüntü toplayana kadar devam eder (ekran/önizleme gerekmez — headless
     çalışır, terminale ilerleme yazar).
  4. cv2.calibrateCamera() ile gerçek fx,fy,cx,cy ve distorsiyon
     katsayılarını hesaplar, reprojection RMS hatasını raporlar.
  5. Sonucu rasp/camera_calib.json'a kaydeder ve config.py'ye eklenecek
     hazır Python satırlarını ekrana basar.

NOT: config.WIDTH x config.HEIGHT (640x480) çözünürlüğünde kalibre eder —
üretimde (vision.py) kullanılan çözünürlükle AYNI olmalı, aksi halde
fx/fy/cx/cy geçersiz olur.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# --- Ayarlar ---
BOARD_COLS      = 9    # satranç tahtası İÇ köşe sayısı (yatay) — kare sayısı değil, köşe sayısı
BOARD_ROWS      = 6    # satranç tahtası İÇ köşe sayısı (dikey)
CHESSBOARD_COUNT = 20   # toplanacak geçerli görüntü sayısı
MIN_CAPTURE_INTERVAL_SEC = 1.2   # ardışık kabul arasında minimum süre (poz çeşitliliği için)
MIN_CORNER_SHIFT_PX = 25.0       # önceki kabul edilenden en az bu kadar farklı olmalı

TOOLS_DIR   = Path(__file__).resolve().parent
PATTERN_PNG = TOOLS_DIR / "chessboard_pattern.png"
CALIB_JSON  = TOOLS_DIR.parent / "camera_calib.json"

SERVICE_NAME = "albatros.service"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 1) SATRANÇ TAHTASI DESENİ ====================

def ensure_pattern_png():
    if PATTERN_PNG.exists():
        return
    log(f"Satranç tahtası deseni oluşturuluyor → {PATTERN_PNG}")
    # (BOARD_COLS+1) x (BOARD_ROWS+1) kare, her kare 120px — A4'te net basılabilir çözünürlük
    square_px = 120
    cols_sq, rows_sq = BOARD_COLS + 1, BOARD_ROWS + 1
    img = np.zeros((rows_sq * square_px, cols_sq * square_px), dtype=np.uint8)
    for r in range(rows_sq):
        for c in range(cols_sq):
            if (r + c) % 2 == 0:
                img[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 255
    # kenara beyaz kenar boşluğu (kağıt kenarından taşmasın, tespih kolaylaşsın)
    margin = 80
    padded = np.full((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin), 255, dtype=np.uint8)
    padded[margin:margin + img.shape[0], margin:margin + img.shape[1]] = img
    cv2.imwrite(str(PATTERN_PNG), padded)
    log(f"✓ Desen kaydedildi ({cols_sq}x{rows_sq} kare = {BOARD_COLS}x{BOARD_ROWS} iç köşe). "
        f"A4 kağıda 1:1 ölçekte (küçültmeden) yazdır, düz/sert bir yüzeye yapıştır.")


# ==================== 2) SERVİS DURDUR/BAŞLAT ====================

def service_is_active():
    r = subprocess.run(["systemctl", "is-active", "--quiet", SERVICE_NAME])
    return r.returncode == 0


def stop_service_if_running():
    if service_is_active():
        log(f"{SERVICE_NAME} çalışıyor — kamera çakışmasın diye geçici durduruluyor...")
        subprocess.run(["sudo", "systemctl", "stop", SERVICE_NAME], check=False)
        time.sleep(2)
        return True
    return False


def restart_service():
    log(f"{SERVICE_NAME} yeniden başlatılıyor...")
    subprocess.run(["sudo", "systemctl", "start", SERVICE_NAME], check=False)


# ==================== 3) GÖRÜNTÜ TOPLAMA ====================

def capture_calibration_images():
    from picamera2 import Picamera2

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (config.WIDTH, config.HEIGHT), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1.5)  # sensör oturması için

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    # NOT: gerçek kare boyutunu (mm) ölçmene gerek YOK — sadece fx/fy/cx/cy ve
    # distorsiyon katsayıları hesaplanıyor, bunlar kare boyutundan etkilenmez
    # (yalnızca konum/rvec-tvec ölçeklenir, biz onu kullanmıyoruz).

    objpoints, imgpoints = [], []
    last_corners = None
    last_capture_time = 0.0

    log(f"Canlı tarama başladı — hedef: {CHESSBOARD_COUNT} geçerli görüntü. "
        f"Satranç tahtasını farklı açı/mesafe/konumlardan yavaşça göster...")
    log("(Ctrl+C ile istediğin an durdurabilirsin, o ana kadar toplananla kalibrasyon dener)")

    try:
        while len(objpoints) < CHESSBOARD_COUNT:
            frame = picam2.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            found, corners = cv2.findChessboardCorners(
                gray, (BOARD_COLS, BOARD_ROWS),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            )
            if not found:
                continue

            now = time.time()
            if now - last_capture_time < MIN_CAPTURE_INTERVAL_SEC:
                continue

            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if last_corners is not None:
                shift = float(np.mean(np.linalg.norm(
                    corners_refined.reshape(-1, 2) - last_corners.reshape(-1, 2), axis=1
                )))
                if shift < MIN_CORNER_SHIFT_PX:
                    continue  # aynı pozdan tekrar — atla, tahtayı hareket ettir

            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            last_corners = corners_refined
            last_capture_time = now

            preview = frame.copy()
            cv2.drawChessboardCorners(preview, (BOARD_COLS, BOARD_ROWS), corners_refined, found)
            # NOT: Picamera2 "RGB888" istese de veri fiilen BGR sırasında geliyor
            # (bilinen bir picamera2 tuhaflığı) — burada TEKRAR RGB2BGR uygulamak
            # kanalları bir daha ters çevirip kırmızı/maviyi karıştırıyordu (yalnızca
            # bu önizleme JPG'sini etkiler; gri tonlamalı köşe tespiti — dolayısıyla
            # kalibrasyon sonuçları — satranç tahtası siyah-beyaz olduğu için bundan
            # etkilenmez, 2026-08-07 sahada fark edildi/düzeltildi).
            cv2.imwrite(str(TOOLS_DIR / "last_capture.jpg"), preview)

            log(f"✓ Görüntü {len(objpoints)}/{CHESSBOARD_COUNT} kabul edildi "
                f"— tahtayı FARKLI bir açı/konuma hareket ettir")

    except KeyboardInterrupt:
        log(f"Kullanıcı durdurdu — {len(objpoints)} görüntüyle devam ediliyor")
    finally:
        picam2.stop()

    return objpoints, imgpoints, (config.WIDTH, config.HEIGHT)


# ==================== 4) KALİBRASYON HESABI ====================

def run_calibration(objpoints, imgpoints, image_size):
    if len(objpoints) < 8:
        log(f"HATA: Yalnızca {len(objpoints)} geçerli görüntü toplandı — güvenilir kalibrasyon "
            f"için en az 8-10 gerekir. Daha fazla poz göstererek tekrar dene.")
        return None

    log(f"{len(objpoints)} görüntüyle cv2.calibrateCamera() çalıştırılıyor...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    fov_x = np.degrees(2 * np.arctan(image_size[0] / (2 * fx)))
    fov_y = np.degrees(2 * np.arctan(image_size[1] / (2 * fy)))

    log("=" * 60)
    log(f"✓ Kalibrasyon tamamlandı — reprojection RMS hatası: {rms:.4f} piksel")
    if rms > 1.0:
        log("  ⚠ UYARI: RMS > 1.0 piksel — kalibrasyon zayıf olabilir. "
            "Daha fazla/çeşitli poz ile tekrar denemen önerilir.")
    log(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    log(f"  Gerçek ölçülen FOV: yatay={fov_x:.2f}°  dikey={fov_y:.2f}°  "
        f"(config.py'deki mevcut değer: {config.CAMERA_FOV_H}°/{config.CAMERA_FOV_V}°)")
    log(f"  Distorsiyon katsayıları (k1,k2,p1,p2,k3): {dist_coeffs.ravel().tolist()}")
    log("=" * 60)

    result = {
        "timestamp":     datetime.now().isoformat(),
        "image_width":   image_size[0],
        "image_height":  image_size[1],
        "rms_error_px":  rms,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs":   dist_coeffs.ravel().tolist(),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "measured_fov_h_deg": fov_x,
        "measured_fov_v_deg": fov_y,
        "num_images": len(objpoints),
    }
    CALIB_JSON.write_text(json.dumps(result, indent=2))
    log(f"✓ Sonuç kaydedildi → {CALIB_JSON}")

    log("\nconfig.py için hazır satırlar (CAMERA_FOV_H/V yerine — geo.py'nin de "
        "güncellenmesi gerekecek, bu script sadece ölçüyor, entegrasyonu ayrı yapılacak):")
    log(f'    CAMERA_FX = {fx:.4f}')
    log(f'    CAMERA_FY = {fy:.4f}')
    log(f'    CAMERA_CX = {cx:.4f}')
    log(f'    CAMERA_CY = {cy:.4f}')
    log(f'    CAMERA_DIST_COEFFS = {dist_coeffs.ravel().tolist()}')

    return result


# ==================== ANA AKIŞ ====================

def main():
    log("=== Kamera Kalibrasyonu Başlıyor ===")
    ensure_pattern_png()

    was_running = stop_service_if_running()
    try:
        objpoints, imgpoints, image_size = capture_calibration_images()
        if objpoints:
            run_calibration(objpoints, imgpoints, image_size)
        else:
            log("HATA: Hiç geçerli görüntü toplanamadı.")
    finally:
        if was_running:
            restart_service()
    log("=== Bitti ===")


if __name__ == "__main__":
    main()

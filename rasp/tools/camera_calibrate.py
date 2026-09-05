#!/usr/bin/env python3
"""
Kamera kalibrasyonu — tek seferlik çalıştırılır.

Kullanım (Pi üzerinde, monitör bağlıyken — bkz. CANLI ÖNİZLEME aşağıda):
    /home/albatros/venv/bin/python3 rasp/tools/camera_calibrate.py

NOT: BAŞINDA sudo KULLANMA — kullanıcı zaten video/gpio gruplarında
(Picamera2 için sudo gerekmiyor) ve script'in kendi içindeki tek ayrıcalıklı
adım (albatros.service durdur/başlat) her hâlükârda kendi sudo'sunu
subprocess ile çağırıyor (parola sorulmaz, passwordless sudo tanımlı).
sudo ile ÇALIŞTIRIRSAN kök kullanıcının farklı bir Wayland/masaüstü oturum
dizini olduğundan (root'un XDG_RUNTIME_DIR'ı senin oturumunun soketini
göremez) CANLI ÖNİZLEME PENCERESİ açılmaz, script sessizce headless moda
düşer.

Yapman gereken TEK şey: script başladıktan sonra satranç tahtası desenini
(otomatik oluşturulup rasp/tools/chessboard_pattern.png olarak kaydedilir —
A4 kağıda düz/kırışıksız bas, SERT bir yüzeye (karton/tahta) yapıştır)
kameranın önünde YAVAŞÇA hareket ettirmek: farklı açılardan, farklı
mesafelerden, kadrajın farklı köşelerinden göster. Script yeterli sayıda
iyi görüntü (varsayılan 20) toplayınca otomatik durur ve hesaplamayı yapar.

CANLI ÖNİZLEME (2026-09-05 eklendi): monitör bağlıysa (bu Pi'de labwc/
Wayland masaüstü) canlı kamera görüntüsü, tespit edilen satranç tahtası
köşeleri ve ilerleme durumu ekranda ayrı bir pencerede gösterilir — artık
kör kör hareket ettirmek yerine tahtanın GERÇEKTEN görülüp görülmediğini
anlık izleyebilirsin. Mouse GEREKMEZ: pencere odaktayken 'q' ya da ESC
tuşu, terminaldeki Ctrl+C ile AYNI şekilde o ana kadar toplananla devam
eder. Monitör/görüntü sunucusu yoksa (ör. saf SSH, X/Wayland soketi
erişilemez) pencere açma denemesi sessizce başarısız olur ve script eski
(2026-09-05 öncesi) headless davranışına düşer — terminale ilerleme
yazmaya devam eder, ÇÖKMEZ.

Ne yapar:
  1. albatros.service çalışıyorsa geçici olarak durdurur (kamerayı tek bir
     süreç kullanabilir — rpicam-vid ile çakışmasın diye), script bitince
     (başarılı/başarısız/Ctrl+C fark etmez) eski haline geri döndürür.
  2. Satranç tahtası deseni yoksa üretir (chessboard_pattern.png).
  3. Picamera2 ile canlı kare akışından satranç tahtası köşelerini otomatik
     tespit eder, yeterince FARKLI pozlardan CHESSBOARD_COUNT adet iyi
     görüntü toplayana kadar devam eder — HER karede (kabul edilsin/
     edilmesin) canlı önizleme penceresine çizilir, terminale de ilerleme
     yazılır.
  4. cv2.calibrateCamera() ile gerçek fx,fy,cx,cy ve distorsiyon
     katsayılarını hesaplar, reprojection RMS hatasını raporlar.
  5. Sonucu rasp/camera_calib.json'a kaydeder ve config.py'ye eklenecek
     hazır Python satırlarını ekrana basar.

NOT: config.WIDTH x config.HEIGHT (640x480) çözünürlüğünde kalibre eder —
üretimde (vision.py) kullanılan çözünürlükle AYNI olmalı, aksi halde
fx/fy/cx/cy geçersiz olur.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Canlı önizleme (2026-09-05): OpenCV'nin Qt GUI arka ucu bu Pi'deki saf
# Wayland masaüstünde (labwc, XWayland YOK) varsayılan xcb eklentisiyle
# pencere açamıyor — QT_QPA_PLATFORM=wayland AÇIKÇA istenmeli. cv2 IMPORT
# EDİLMEDEN ÖNCE ayarlanmalı (Qt eklentisi ilk kullanımda seçiliyor).
# Kullanıcı zaten kendi ortamında farklı bir değer ayarlamışsa (ör. gerçek
# X11/XWayland oturumu) DOKUNULMUYOR — yalnızca WAYLAND_DISPLAY VARSA ve
# QT_QPA_PLATFORM hiç ayarlanmamışsa varsayılan veriliyor.
if os.environ.get("WAYLAND_DISPLAY") and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "wayland"

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

_PREVIEW_WINDOW = "Kamera Kalibrasyonu — q/ESC: bitir"


def _open_preview_window():
    """Canlı önizleme penceresini açmayı DENER (2026-09-05) — monitör/görüntü
    sunucusu yoksa (ör. saf SSH, X/Wayland soketi erişilemez) cv2 burada
    istisna fırlatır; bu fonksiyon False döner, çağıran taraf sessizce eski
    (2026-09-05 öncesi) headless davranışına düşer. Script'in geri kalanı
    ÇÖKMEZ — yalnızca görsel geri bildirim kaybolur."""
    try:
        cv2.namedWindow(_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_PREVIEW_WINDOW, config.WIDTH * 2, config.HEIGHT * 2)
        return True
    except Exception as e:
        log(f"⚠ Canlı önizleme penceresi açılamadı ({e}) — headless devam ediliyor "
            f"(terminale ilerleme yazılmaya devam eder)")
        return False


def _draw_preview(frame, found, corners, status_text, status_color, progress_text):
    """HER karede (kabul edilsin/edilmesin) çağrılır — tespit edilen köşeleri
    ve ilerleme durumunu canlı pencereye çizer."""
    preview = frame.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(preview, (BOARD_COLS, BOARD_ROWS), corners, found)
    # NOT: Picamera2 "RGB888" istese de veri fiilen BGR sırasında geliyor (bkz.
    # aşağıdaki last_capture.jpg notu) — bu yüzden burada TEKRAR RGB2BGR
    # UYGULANMIYOR, önizleme frame'i olduğu gibi kullanılıyor.
    cv2.putText(preview, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, status_color, 2, cv2.LINE_AA)
    cv2.imshow(_PREVIEW_WINDOW, preview)


def _preview_quit_requested():
    """cv2.waitKey ile pencere olaylarını pompalar VE 'q'/ESC'e basılıp
    basılmadığını kontrol eder — mouse gerekmez, terminaldeki Ctrl+C ile
    AYNI etkiyi (o ana kadar toplananla devam) verir."""
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27)  # 27 = ESC


def capture_calibration_images():
    from picamera2 import Picamera2

    # NOT: rpicam-vid (üretim) de kasıtlı olarak rotasyonsuz (native) çekiyor
    # — kamera ters monte edilse bile hem burada hem üretimde native yön
    # kullanılıyor, bu yüzden kalibrasyon rotasyon dursun/değişsin farketmez,
    # her zaman geçerli kalır (bkz. pipeline.py, geo.py CAMERA_ROTATION_DEG).
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
    accepted_flash_until = 0.0  # bu zamana kadar "KABUL EDİLDİ ✓" gösterilir

    preview_on = _open_preview_window()

    log(f"Canlı tarama başladı — hedef: {CHESSBOARD_COUNT} geçerli görüntü. "
        f"Satranç tahtasını farklı açı/mesafe/konumlardan yavaşça göster...")
    if preview_on:
        log("(Ctrl+C VEYA önizleme penceresindeyken 'q'/ESC ile istediğin an "
            "durdurabilirsin, o ana kadar toplananla kalibrasyon dener)")
    else:
        log("(Ctrl+C ile istediğin an durdurabilirsin, o ana kadar toplananla kalibrasyon dener)")

    try:
        while len(objpoints) < CHESSBOARD_COUNT:
            frame = picam2.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            progress_text = f"Kalibrasyon: {len(objpoints)}/{CHESSBOARD_COUNT}"

            found, corners = cv2.findChessboardCorners(
                gray, (BOARD_COLS, BOARD_ROWS),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            )
            if not found:
                if preview_on:
                    # NOT: cv2.putText (Hershey font) UTF-8 Türkçe karakterleri (İ/Ğ/Ş/ı
                    # vb.) ve ✓/— gibi özel sembolleri render edemiyor ("??" basıyor,
                    # bkz. vision.py/geo.py'deki aynı not) — bu yüzden pencere
                    # metinleri KASITLI olarak ASCII-only.
                    status = ("KABUL EDILDI +" if time.time() < accepted_flash_until
                               else "ARANIYOR - tahtayi kadraja al")
                    color = (0, 255, 0) if time.time() < accepted_flash_until else (0, 165, 255)
                    _draw_preview(frame, found, corners, status, color, progress_text)
                    if _preview_quit_requested():
                        log("Kullanıcı 'q'/ESC ile durdurdu")
                        break
                continue

            now = time.time()
            if now - last_capture_time < MIN_CAPTURE_INTERVAL_SEC:
                if preview_on:
                    _draw_preview(frame, found, corners, "TESPIT EDILDI - bekleniyor (cooldown)",
                                  (0, 255, 255), progress_text)
                    if _preview_quit_requested():
                        log("Kullanıcı 'q'/ESC ile durdurdu")
                        break
                continue

            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if last_corners is not None:
                shift = float(np.mean(np.linalg.norm(
                    corners_refined.reshape(-1, 2) - last_corners.reshape(-1, 2), axis=1
                )))
                if shift < MIN_CORNER_SHIFT_PX:
                    if preview_on:
                        _draw_preview(frame, found, corners_refined,
                                      "TESPIT EDILDI - ayni poz, HAREKET ETTIR",
                                      (0, 165, 255), progress_text)
                        if _preview_quit_requested():
                            log("Kullanıcı 'q'/ESC ile durdurdu")
                            break
                    continue  # aynı pozdan tekrar — atla, tahtayı hareket ettir

            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            last_corners = corners_refined
            last_capture_time = now
            accepted_flash_until = now + 0.6

            if preview_on:
                _draw_preview(frame, found, corners_refined,
                              f"KABUL EDILDI + ({len(objpoints)}/{CHESSBOARD_COUNT})",
                              (0, 255, 0), progress_text)
                if _preview_quit_requested():
                    log("Kullanıcı 'q'/ESC ile durdurdu")
                    break

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
        if preview_on:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # bazı Qt sürümlerinde pencerenin fiilen kapanması için gerekli

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

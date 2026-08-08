"""
OpenCV: HSV renk tespiti, kare doğrulama, overlay ve video pipeline thread'i.
FFmpeg decode/encode ve GStreamer süreçleri bu modülde başlatılır.
"""
import logging
import queue as _queue
import subprocess
import threading
import time
import cv2
import numpy as np
import config
import state
import geo

log = logging.getLogger("vision")


# ==================== YARDIMCI FONKSİYONLAR ====================

def is_square(contour):
    """
    Konturun kareye yakın bir şekil olup olmadığını kontrol eder.
    4 köşe + en-boy oranı tek başına altıgen/üçgeni elemeye yetmiyor (ikisinin de
    bounding box'ı kareye yakın çıkabiliyor) — bu yüzden doluluk oranı (kontur
    alanı / bounding box alanı) da kontrol ediliyor: kare ~0.9-1.0, altıgen
    ~0.65-0.87, üçgen ~0.5 (2026-07-26 sahada altıgen/üçgen yanlış tespiti sonrası
    eklendi).
    """
    perimeter = cv2.arcLength(contour, True)
    approx    = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) != config.SQUARE_CORNER_TOLERANCE:
        return False
    if not cv2.isContourConvex(approx):
        return False
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w <= 0 or h <= 0:
        return False
    ratio = max(w, h) / min(w, h)
    if not (config.ASPECT_RATIO_MIN < ratio < config.ASPECT_RATIO_MAX):
        return False
    extent = cv2.contourArea(contour) / (w * h)
    return extent > config.SQUARE_MIN_EXTENT


def _apply_morph(mask):
    """Kapat + Aç morfoloji uygular — gürültüyü azaltır."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, state.kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN,  state.kernel)


def _detect_square_in_mask(mask):
    """
    Maskede geçerli ilk kareyi arar.
    Dönüş: (cx, cy, contour) veya None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (config.MIN_AREA < area < config.MAX_AREA):
            continue
        if not is_square(cnt):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy, cnt
    return None


def _update_detection(mask, color_key, queue, queued_colors):
    """
    Maskeden kare tespit eder.
    state.detected_targets güncellenir; detection_active ise kuyruk da beslenir.
    """
    result = _detect_square_in_mask(mask)
    if result:
        cx, cy, cnt = result
        state.detected_targets[color_key] = {"cx": cx, "cy": cy, "contour": cnt}
        if state.detection_active.is_set() and color_key not in queued_colors:
            queue.put({"color": color_key, "cx": cx, "cy": cy})
            queued_colors.add(color_key)
            log.info(f"[VISION] *** {color_key.upper()} KARE TESPİT EDİLDİ *** "
                  f"piksel=({cx},{cy}) — mission kuyruğuna eklendi")
    else:
        state.detected_targets[color_key] = None


# ==================== ANA THREAD ====================


def _wait_for_tcp_listen(port, timeout=15, interval=0.2):
    """rpicam-vid TCP portu dinlemeye baslayana kadar bekler (baglanti KURMADAN kontrol eder,
    aksi halde --listen tek seferlik accept slotunu tuketebilir)."""
    start = time.time()
    while time.time() - start < timeout:
        if state.shutdown_requested.is_set():
            log.info("[VISION] Kapanış sinyali alındı — port bekleme iptal edildi")
            return False
        try:
            out = subprocess.run(["ss", "-tln"], capture_output=True, text=True, timeout=1)
            if f":{port} " in out.stdout:
                log.info(f"[VISION] ✓ Port {port} dinlemede — devam ediliyor")
                return True
        except Exception:
            pass
        time.sleep(interval)
    log.info(f"[VISION] ⚠ Port {port} {timeout}s icinde dinlemeye gecmedi, yine de devam ediliyor")
    return False

def opencv_processing_thread(queue):
    log.info("[VISION] Thread başlatıldı — rpicam-vid TCP portu bekleniyor...")
    _wait_for_tcp_listen(config.RPICAM_TCP_PORT, timeout=25)
    if state.shutdown_requested.is_set():
        log.info("[VISION] Kapanış sinyali alındı — pipeline başlatılmadan çıkılıyor")
        return

    # --- FFmpeg decode: TCP H264 → raw BGR ---
    # -r (giriş framerate ipucu) -i'den ONCE sart: aksi halde probesize/analyzeduration
    # cok kucuk oldugu icin ffmpeg gercek framerate'i tahmin edemiyor ve neredeyse
    # tum kareleri "drop" ediyor (2026-07-26 sahada bulunup dogrulandi).
    ffmpeg_decode_cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-r", str(config.FPS),
        "-i", f"tcp://127.0.0.1:{config.RPICAM_TCP_PORT}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.WIDTH}x{config.HEIGHT}",
        "-r", str(config.FPS),
        "-",
    ]
    log.info(f"[VISION] FFmpeg decode başlatılıyor: {' '.join(ffmpeg_decode_cmd)}")
    ffmpeg_decode_stderr = open("/home/albatros/logs/ffmpeg_decode.log", "wb")
    state.ffmpeg_decode_process = subprocess.Popen(
        ffmpeg_decode_cmd, stdout=subprocess.PIPE, stderr=ffmpeg_decode_stderr
    )
    log.info(f"[VISION] FFmpeg decode başladı → PID={state.ffmpeg_decode_process.pid}")

    # --- FFmpeg encode: raw BGR → H264 (yazilimsal) ---
    # Pi 5'te bcm2835-codec gibi ayri bir H264 donanim encode blogu yok (video19
    # sadece HEVC decode) - h264_v4l2m2m "Could not find a valid device" hatasi
    # veriyordu (2026-07-26 sahada bulunup dogrulandi). libx264 yerine kullanildi.
    ffmpeg_encode_cmd = [
        "ffmpeg",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.WIDTH}x{config.HEIGHT}",
        "-r", str(config.FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-b:v", "2000k", "-g", "15",
        "-flush_packets", "1",
        "-f", "h264", "-",
    ]
    log.info(f"[VISION] FFmpeg encode başlatılıyor: {' '.join(ffmpeg_encode_cmd)}")
    ffmpeg_encode_stderr = open("/home/albatros/logs/ffmpeg_encode.log", "wb")
    state.ffmpeg_encode_process = subprocess.Popen(
        ffmpeg_encode_cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=ffmpeg_encode_stderr
    )
    log.info(f"[VISION] FFmpeg encode başladı → PID={state.ffmpeg_encode_process.pid} (libx264)")

    # Encode yazımını ayrı thread'de yap — encoder meşgulse frame düş, OpenCV'yi bloklama
    _encode_queue = _queue.Queue(maxsize=2)

    def _encode_writer():
        while True:
            frame_bytes = _encode_queue.get()
            if frame_bytes is None:
                break
            try:
                state.ffmpeg_encode_process.stdin.write(frame_bytes)
            except Exception:
                pass

    threading.Thread(target=_encode_writer, daemon=True).start()

    # --- GStreamer: H264 → RTP → UDP → WFB-ng ---
    def ffmpeg_to_gstreamer():
        gst_cmd = [
            "gst-launch-1.0", "fdsrc", "!",
            "h264parse", "!",
            "rtph264pay", "config-interval=1", "pt=96", "!",
            "udpsink", "host=127.0.0.1", f"port={config.WFB_UDP_PORT}",
        ]
        log.info(f"[VISION] GStreamer başlatılıyor: {' '.join(gst_cmd)}")
        state.gst_process = subprocess.Popen(
            gst_cmd, stdin=state.ffmpeg_encode_process.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        log.info(f"[VISION] GStreamer başladı → PID={state.gst_process.pid}")
        for line in state.gst_process.stdout:
            log.info(f"[GST] {line.rstrip()}")
            if "New clock" in line:
                log.info("[VISION] ✓ Video/WFB pipeline PLAYING — aktarım fiilen başladı")
        state.gst_process.wait()
        log.info(f"[VISION] GStreamer süreci sonlandı (PID={state.gst_process.pid})")

    threading.Thread(target=ffmpeg_to_gstreamer, daemon=True).start()

    frame_size    = config.WIDTH * config.HEIGHT * 3
    gps_data      = {}      # {color: (lat, lon)} — her renk için yalnızca bir kez hesaplanır
    queued_colors = set()   # her renk yalnızca bir kez kuyruğa girer
    frame_count   = 0
    fps_time      = time.time()
    current_fps   = 0
    debug_timer   = time.time()

    # Kalibrasyon varsa distorsiyon düzeltme haritası BİR KEZ hesaplanır (her
    # frame'de cv2.undistort çağırmak yerine cv2.remap kullanmak için — çok
    # daha ucuz). camera_matrix hem eski hem yeni matris olarak verildiği için
    # fx/fy/cx/cy (dolayısıyla geo.pixel_to_gps'teki açı hesabı) düzeltilmiş
    # frame'de de GEÇERLİ kalır (bkz. config.py CAMERA_CALIBRATED).
    undistort_map1 = undistort_map2 = None
    if config.CAMERA_CALIBRATED:
        camera_matrix = np.array([
            [config.CAMERA_FX, 0, config.CAMERA_CX],
            [0, config.CAMERA_FY, config.CAMERA_CY],
            [0, 0, 1],
        ])
        dist_coeffs = np.array(config.CAMERA_DIST_COEFFS)
        undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix,
            (config.WIDTH, config.HEIGHT), cv2.CV_16SC2,
        )
        log.info("[VISION] ✓ Kamera kalibrasyonu aktif — frame'ler distorsiyon düzeltmesinden geçirilecek")
    else:
        log.info("[VISION] ⚠ Kamera kalibrasyonu YOK — ham (distorsiyonlu) frame kullanılıyor")

    log.info(f"[VISION] Frame okuma döngüsü başladı — frame_size={frame_size} bytes")

    while True:
        try:
            raw_frame = state.ffmpeg_decode_process.stdout.read(frame_size)

            if len(raw_frame) != frame_size:
                log.info(f"[VISION] Eksik frame: {len(raw_frame)}/{frame_size} byte — atlanıyor")
                time.sleep(0.01)
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (config.HEIGHT, config.WIDTH, 3)
            ).copy()

            if undistort_map1 is not None:
                frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)

            # FPS hesabı
            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_time    = now

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Mavi kare tespiti
            mask_blue = _apply_morph(cv2.inRange(hsv, config.BLUE_HSV_LOWER, config.BLUE_HSV_UPPER))
            _update_detection(mask_blue, "mavi", queue, queued_colors)

            # Kırmızı kare tespiti (iki HSV aralığı OR'lanır)
            mask_red = _apply_morph(cv2.bitwise_or(
                cv2.inRange(hsv, config.RED1_HSV_LOWER, config.RED1_HSV_UPPER),
                cv2.inRange(hsv, config.RED2_HSV_LOWER, config.RED2_HSV_UPPER),
            ))
            _update_detection(mask_red, "kirmizi", queue, queued_colors)

            # === OVERLAY ===
            for color, data in state.detected_targets.items():
                if data:
                    box_color = (255, 100, 0) if color == "mavi" else (0, 0, 255)
                    cv2.drawContours(frame, [data["contour"]], -1, box_color, 3)
                    if color in gps_data:
                        lat, lon = gps_data[color]
                        cv2.putText(
                            frame, f"{color.upper()}: {lat:.6f}, {lon:.6f}",
                            (data["cx"] - 100, data["cy"] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                        )

            cv2.putText(frame, f"FPS: {current_fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            with state.telemetry_lock:
                tel = state.current_telemetry.copy()

            if tel["alt"]:
                cv2.putText(frame, f"ALT: {tel['alt']:.1f}m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            wp = state.current_wp
            if wp["index"] is not None:
                cv2.putText(frame, f"WP: {wp['index']}/{wp['total']}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            fc_ok = state.fc_connected
            cv2.putText(
                frame,
                f"FC: {'BAGLI' if fc_ok else 'BAGLI DEGIL'} ({config.FC_BAUDRATE})",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if fc_ok else (0, 0, 255), 2,
            )

            rf_ok = state.wfb_process is not None and state.wfb_process.poll() is None
            cv2.putText(
                frame,
                f"RF: {'OK' if rf_ok else 'KOPTU'}",
                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if rf_ok else (0, 0, 255), 2,
            )

            for color, label, pos, col_ok, col_no in [
                ("mavi",    "MAVI: OK",    (config.WIDTH - 160, 30), (255, 100, 0), (128, 128, 128)),
                ("kirmizi", "KIRMIZI: OK", (config.WIDTH - 160, 60), (0, 0, 255),   (128, 128, 128)),
            ]:
                detected = state.detected_targets[color]
                cv2.putText(
                    frame,
                    label if detected else label.replace("OK", "--"),
                    pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    col_ok if detected else col_no, 2,
                )

            try:
                _encode_queue.put_nowait(frame.tobytes())
            except _queue.Full:
                pass  # encoder meşgul, bu frame'i düş

            # GPS koordinatı hesapla — her renk için yalnızca bir kez
            if None not in tel.values():
                for color, data in state.detected_targets.items():
                    if data and color not in gps_data:
                        lat, lon = geo.pixel_to_gps(
                            tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
                            data["cx"], data["cy"],
                        )
                        gps_data[color] = (lat, lon)
                        log.info(f"[VISION] {color.upper()} GPS koordinatı hesaplandı: "
                              f"({lat:.6f}, {lon:.6f})")

            # Periyodik durum özeti (her 5 saniyede bir)
            if now - debug_timer >= 5.0:
                try:
                    cv2.imwrite("/home/albatros/logs/hud_preview.jpg", frame)
                except Exception:
                    pass
                mavi_str = (
                    f"({state.detected_targets['mavi']['cx']},{state.detected_targets['mavi']['cy']})"
                    if state.detected_targets["mavi"] else "—"
                )
                kirmizi_str = (
                    f"({state.detected_targets['kirmizi']['cx']},{state.detected_targets['kirmizi']['cy']})"
                    if state.detected_targets["kirmizi"] else "—"
                )
                detection_str = "AKTİF" if state.detection_active.is_set() else "BEKLİYOR"
                log.info(f"[VISION][STATUS] FPS={current_fps} | "
                      f"mavi={mavi_str} kirmizi={kirmizi_str} | "
                      f"tespit={detection_str} | kuyruk={queue.qsize()}")
                if tel["lat"] is not None:
                    log.info(f"[VISION][STATUS] Telemetri: "
                          f"lat={tel['lat']:.6f} lon={tel['lon']:.6f} "
                          f"alt={tel['alt']:.1f}m yaw={tel['yaw']:.1f}°")
                debug_timer = now

        except Exception as e:
            log.info(f"[VISION] HATA: {e}")
            time.sleep(0.1)

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
import pipeline

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


# ==================== YAYIN (stream_frame) KOORDİNAT DÖNÜŞÜMÜ ====================
# Tespit native (ham) frame'de yapılıyor (kalibrasyon + geo.py matematiği için,
# bkz. config.py CAMERA_ROTATION_DEG notu), ama overlay YAYINA giden çevrilmiş
# kareye çiziliyor — HUD metinleri de doğru yönde görünsün diye. Bu yüzden
# native koordinatlardan (kontur, cx/cy) gelen her şey önce stream uzayına
# çevrilmeli; sabit pozisyonlu HUD öğeleri (FPS, ALT, vb.) zaten stream_frame
# üzerine çizildiği için dönüşüme ihtiyaç duymaz.

def _to_stream_xy(x, y):
    if config.CAMERA_ROTATION_DEG == 180:
        return config.WIDTH - 1 - x, config.HEIGHT - 1 - y
    return x, y


def _to_stream_contour(contour):
    if config.CAMERA_ROTATION_DEG == 180:
        c = contour.copy()
        c[:, :, 0] = config.WIDTH - 1 - c[:, :, 0]
        c[:, :, 1] = config.HEIGHT - 1 - c[:, :, 1]
        return c
    return contour


def _mask_debug_info(mask):
    """
    Tanı amaçlı (2026-08-17): mavi/kırmızı hiç tespit edilmiyorken sorunun RENK
    (HSV aralığı hiçbir pikseli yakalamıyor) mi yoksa ŞEKİL (renk tutuyor ama
    is_square/alan filtresi reddediyor) mi olduğunu ayırt etmek için — 13-17
    Ağustos loglarında tek bir başarılı tespit yok, körlemesine HSV ayarlamak
    yerine bir sonraki testte gerçek sayı görmek gerekiyor.
    """
    pixel_count = int(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"pixels": pixel_count, "candidate": None}
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
    rect = cv2.minAreaRect(largest)
    w, h = rect[1]
    ratio = round(max(w, h) / min(w, h), 2) if w > 0 and h > 0 else None
    extent = round(area / (w * h), 2) if w > 0 and h > 0 else None
    return {
        "pixels": pixel_count,
        "candidate": {
            "area": int(area),
            "in_area_range": config.MIN_AREA < area < config.MAX_AREA,
            "corners": len(approx),
            "convex": bool(cv2.isContourConvex(approx)),
            "ratio": ratio,
            "extent": extent,
        },
    }


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


def _finalize_track(color_key, tracking, queue, queued_colors, reason):
    """En iyi (yer mesafesi en küçük) örneği kuyruğa GERÇEK GPS koordinatıyla gönderir."""
    track = tracking[color_key]
    queue.put({"color": color_key, "lat": track["lat"], "lon": track["lon"], "alt": track["alt"]})
    queued_colors.add(color_key)
    log.info(f"[VISION] *** {color_key.upper()} KİLİTLENDİ ({reason}) *** "
          f"({track['lat']:.6f},{track['lon']:.6f}) en-iyi-mesafe={track['dist']:.1f}m "
          f"— mission kuyruğuna eklendi")


def _bbox_to_contour(x, y, w, h):
    """MOSSE bbox'ını cv2.drawContours/_to_stream_contour ile uyumlu 4 köşeli
    kontur dizisine çevirir (tracker köprüleme karelerinde overlay için)."""
    x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
    return np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)


def _pad_bbox_to_min_size(x, y, w, h, min_size, frame_w, frame_h):
    """Bbox'ı merkezini koruyarak en az min_size x min_size olacak şekilde
    büyütür (çerçeve dışına taşarsa kelepçelenir, merkez kayabilir).

    Neden: MOSSE'nin PSR güven eşiği küçük yamalarda güvenilir geçmiyor
    (bkz. config.TRACKER_MIN_PATCH_PX notu) — hedefin GÖRÜNÜR boyutu irtifaya
    göre büyük ölçüde değiştiğinden, tracker'a ham (küçük olabilecek) tespit
    bbox'ı değil, çevresindeki gerçek dokuyu da içeren bu büyütülmüş yama
    veriliyor. Böylece köprüleme yalnızca yakın/büyük tespitlerle sınırlı
    kalmaz, irtifa değişimlerine dayanıklı olur."""
    cx, cy = x + w / 2.0, y + h / 2.0
    pw, ph = max(w, min_size), max(h, min_size)
    px = int(round(max(0, min(cx - pw / 2.0, frame_w - pw))))
    py = int(round(max(0, min(cy - ph / 2.0, frame_h - ph))))
    pw = int(round(min(pw, frame_w - px)))
    ph = int(round(min(ph, frame_h - py)))
    return px, py, pw, ph


def _update_detection(mask, color_key, queue, queued_colors, tel, tracking, frame, trackers):
    """
    Maskeden kare tespit eder. state.detected_targets HUD/overlay için her
    zaman güncellenir.

    GPS kilidi: hedefin İLK görüldüğü karede DEĞİL (o genelde kenarda/uzak
    ve en hatalı örnektir — bkz. config.py MAX_DETECTION_DISTANCE_M notu,
    2026-08-12), uçak yaklaşırken görülen EN İYİ (pixel_to_gps'in ürettiği
    yer mesafesi en küçük) karede yapılır. Kilit üç durumdan biriyle tetiklenir
    (bkz. config.py DETECTION_CONFIRM_STREAK/DETECTION_LOST_STREAK/
    DETECTION_TRACK_MAX_SEC): art arda kötüleşme (en yakın noktayı geçtik),
    art arda kayıp (kadraj dışına çıktı) ya da süre aşımı (güvenlik ağı).

    MOSSE köprüleme (2026-08-19, bkz. config.py TRACKER_MAX_BRIDGE_FRAMES):
    HSV/kontur tespiti bu ARADA tek-iki kare kaçırırsa (motion blur, H264
    blok gürültüsü, anlık parlama) `trackers[color_key]` son gerçek tespitten
    beslenen bir MOSSE korelasyon tracker'ı tutar ve bu boşluğu tahmin edilen
    konumla köprüler — miss_streak erken tetiklenip kısa geçiş penceresinde
    hedef gereksiz yere kaybedilmesin diye. Tracker tahmini SADECE HUD/overlay
    ve miss_streak köprülemesi için kullanılır; GPS kilidi (pixel_to_gps) her
    zaman gerçek renk-doğrulanmış bir karede hesaplanır, sürüklenme riski
    konum doğruluğuna asla karışmaz.
    """
    result = _detect_square_in_mask(mask)
    track = tracking[color_key]
    locked = color_key in queued_colors
    tstate = trackers[color_key]

    if result:
        cx, cy, cnt = result
        state.detected_targets[color_key] = {"cx": cx, "cy": cy, "contour": cnt}

        # Gerçek (renk-doğrulanmış) tespit — MOSSE'yi bu karede YENİDEN
        # başlat. Sık sık resetlemek KASITLI: tracker'ı hep taze bir gerçek
        # tespite dayandırıp sürüklenmenin birikmesini önler. Bbox, hedef
        # küçük (yüksek irtifa) olsa bile TRACKER_MIN_PATCH_PX'e padding'lenir
        # — bkz. _pad_bbox_to_min_size notu.
        x, y, w, h = cv2.boundingRect(cnt)
        tx, ty, tw, th = _pad_bbox_to_min_size(
            x, y, w, h, config.TRACKER_MIN_PATCH_PX, config.WIDTH, config.HEIGHT)
        try:
            cv_tracker = cv2.legacy.TrackerMOSSE_create()
            cv_tracker.init(frame, (tx, ty, tw, th))
        except Exception as e:
            cv_tracker = None
            log.debug(f"[VISION] {color_key} MOSSE init hatası: {e}")
        trackers[color_key] = {"cv": cv_tracker, "bridged": 0}

        # Bayat telemetri reddi (2026-08-17, "277m belirsiz yönde sapma" analizi):
        # frame anına en yakın telemetri örneği bile TELEMETRY_MATCH_MAX_AGE_S'den
        # eskiyse bu kareden GPS HESABI YAPILMAZ — eskiden yalnızca uyarı
        # loglanıp hesap yine de yapılıyordu; dönüş sırasında yüz ms'lerce eski
        # yaw/roll ile projeksiyon, işareti keyfi yönde onlarca-yüzlerce metre
        # kaydırabiliyor. EKF reddi de aynı mantık: FC "global konumum sağlıklı
        # değil" diyorsa (is False; None=henüz veri yok, engelleme) o anki
        # drone konumu güvenilmez, tespit örneği alınmaz.
        tel_fresh = max(tel.get("pos_age_s") or 0.0,
                        tel.get("att_age_s") or 0.0) <= config.TELEMETRY_MATCH_MAX_AGE_S
        ekf_ok = state.ekf_health["global_position_ok"] is not False

        if (state.detection_active.is_set() and not locked
                and None not in tel.values() and tel_fresh and ekf_ok):
            geo_result = geo.pixel_to_gps(
                tel["lat"], tel["lon"], tel["alt"], tel["yaw"],
                tel["roll"], tel["pitch"], cx, cy,
            )
            if geo_result is not None:
                lat, lon = geo_result
                dist = geo.haversine(tel["lat"], tel["lon"], lat, lon)
                if track is None or dist < track["dist"]:
                    tracking[color_key] = track = {
                        "lat": lat, "lon": lon, "alt": tel["alt"], "dist": dist,
                        "worse_streak": 0, "miss_streak": 0,
                        "first_seen": track["first_seen"] if track else time.time(),
                    }
                    log.info(f"[VISION] {color_key.upper()} yeni en-iyi örnek: "
                          f"mesafe={dist:.1f}m piksel=({cx},{cy})")
                else:
                    track["worse_streak"] += 1
                    track["miss_streak"] = 0
    else:
        bridged = False
        if (tstate is not None and tstate["cv"] is not None and not locked
                and tstate["bridged"] < config.TRACKER_MAX_BRIDGE_FRAMES):
            ok, bbox = tstate["cv"].update(frame)
            if ok:
                bx, by, bw, bh = bbox
                cx, cy = int(bx + bw / 2), int(by + bh / 2)
                state.detected_targets[color_key] = {
                    "cx": cx, "cy": cy, "contour": _bbox_to_contour(bx, by, bw, bh),
                }
                tstate["bridged"] += 1
                bridged = True

        if not bridged:
            state.detected_targets[color_key] = None
            if track is not None and not locked:
                track["miss_streak"] += 1
                if track["miss_streak"] >= config.DETECTION_LOST_STREAK:
                    _finalize_track(color_key, tracking, queue, queued_colors, "kadraj dışı")
                    return

    track = tracking[color_key]
    if track is not None and not locked:
        if track["worse_streak"] >= config.DETECTION_CONFIRM_STREAK:
            _finalize_track(color_key, tracking, queue, queued_colors, "en yakın nokta geçildi")
        elif time.time() - track["first_seen"] > config.DETECTION_TRACK_MAX_SEC:
            _finalize_track(color_key, tracking, queue, queued_colors, "süre aşımı")


# ==================== ANA THREAD ====================


def opencv_processing_thread(queue):
    log.info("[VISION] Thread başlatıldı — rpicam-vid raw stdout bekleniyor...")
    if state.shutdown_requested.is_set():
        log.info("[VISION] Kapanış sinyali alındı — pipeline başlatılmadan çıkılıyor")
        return

    # NOT (2026-08-16): Eskiden rpicam-vid H264 üretip TCP üzerinden ffmpeg'e
    # gönderiyordu, ffmpeg de bunu raw BGR'a DECODE ediyordu — sonra OpenCV
    # işleyip aşağıdaki ffmpeg_encode ile TEKRAR H264'e encode ediliyordu.
    # Çift encode/decode gereksiz CPU yükü + gecikme demekti (ffmpeg decode
    # süreci komple kaldırıldı, bkz. pipeline.py rpicam_cmd — artık --codec
    # yuv420 -o - ile raw I420 doğrudan bu sürecin stdout'undan okunuyor).

    # --- FFmpeg encode + GStreamer zinciri ---
    # Pi 5'te bcm2835-codec gibi ayri bir H264 donanim encode blogu yok (video19
    # sadece HEVC decode) - h264_v4l2m2m "Could not find a valid device" hatasi
    # veriyordu (2026-07-26 sahada bulunup dogrulandi). libx264 yerine kullanildi.
    #
    # Bitrate/GOP artık config'ten (2026-08-17): eskiden -b:v 2000k hardcode'du,
    # config.BITRATE (WFB FEC bütçesine göre 1M'e düşürülen değer) fiilen
    # KULLANILMIYORDU — MCS0/20MHz/longGI hava hızı ~6.5Mbps x FEC 4/12 ≈
    # ~2.1Mbps kullanılabilir bant; 2000k video + RTP/FEC yükü bunu aşıp menzil
    # sınırında paket kaybı/donma üretiyordu. maxrate/bufsize de eklendi ki
    # libx264 anlık patlamaları (sahne değişimi) bandın üstüne taşırmasın.
    def _start_encode_chain():
        ffmpeg_encode_cmd = [
            "ffmpeg",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{config.WIDTH}x{config.HEIGHT}",
            "-r", str(config.FPS),
            "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-profile:v", "baseline",
            "-b:v", str(config.BITRATE), "-maxrate", str(config.BITRATE),
            "-bufsize", str(config.BITRATE // 2), "-g", str(config.INTRA),
            "-flush_packets", "1",
            "-f", "h264", "-",
        ]
        log.info(f"[VISION] FFmpeg encode başlatılıyor: {' '.join(ffmpeg_encode_cmd)}")
        ffmpeg_encode_stderr = open("/home/albatros/logs/ffmpeg_encode.log", "ab")
        state.ffmpeg_encode_process = subprocess.Popen(
            ffmpeg_encode_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=ffmpeg_encode_stderr
        )
        log.info(f"[VISION] FFmpeg encode başladı → PID={state.ffmpeg_encode_process.pid} (libx264)")

        def _gst_pump(ffmpeg_proc):
            gst_cmd = [
                "gst-launch-1.0", "fdsrc", "!",
                "h264parse", "!",
                "rtph264pay", "config-interval=1", "pt=96", "!",
                "udpsink", "host=127.0.0.1", f"port={config.WFB_UDP_PORT}",
            ]
            log.info(f"[VISION] GStreamer başlatılıyor: {' '.join(gst_cmd)}")
            state.gst_process = subprocess.Popen(
                gst_cmd, stdin=ffmpeg_proc.stdout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            log.info(f"[VISION] GStreamer başladı → PID={state.gst_process.pid}")
            for line in state.gst_process.stdout:
                log.info(f"[GST] {line.rstrip()}")
                if "New clock" in line:
                    log.info("[VISION] ✓ Video/WFB pipeline PLAYING — aktarım fiilen başladı")
            state.gst_process.wait()
            log.info(f"[VISION] GStreamer süreci sonlandı (PID={state.gst_process.pid})")

        threading.Thread(target=_gst_pump, args=(state.ffmpeg_encode_process,),
                         daemon=True).start()

    _start_encode_chain()

    # Encode yazımını ayrı thread'de yap — encoder meşgulse frame düş, OpenCV'yi
    # bloklama. Encoder ÖLÜRSE (2026-08-17): eskiden exception sessizce yutuluyor,
    # yayın (hakeme anlık kanıt — şartname zorunluluğu!) fark edilmeden ölü
    # kalıyordu. Artık loglanır ve 5 sn'lik geri çekilmeyle zincir yeniden kurulur.
    _encode_queue = _queue.Queue(maxsize=2)

    def _encode_writer():
        last_restart = 0.0
        while not state.shutdown_requested.is_set():
            frame_bytes = _encode_queue.get()
            if frame_bytes is None:
                break
            try:
                state.ffmpeg_encode_process.stdin.write(frame_bytes)
            except Exception as e:
                proc = state.ffmpeg_encode_process
                dead = proc is None or proc.poll() is not None
                now_m = time.monotonic()
                if dead and now_m - last_restart > 5.0:
                    log.info(f"[VISION] ⚠ FFmpeg encode ÖLDÜ ({e}) — zincir yeniden kuruluyor")
                    last_restart = now_m
                    try:
                        _start_encode_chain()
                    except Exception as e2:
                        log.info(f"[VISION] Encode zinciri yeniden kurulamadı: {e2}")

    threading.Thread(target=_encode_writer, daemon=True).start()

    # Raw YUV420 (I420 planar): width*height*1.5 byte/kare — eski BGR decode
    # çıktısının (width*height*3) yarısı, ayrıca kaynakta bir codec turu eksik.
    frame_size    = config.WIDTH * config.HEIGHT * 3 // 2

    # --- Taze-kare okuyucu thread (2026-08-17, "277m sapma" analizi) ---
    # ESKİDEN işleme döngüsü pipe'tan doğrudan okuyordu: OpenCV 30fps'e
    # yetişemediği an pipe dolar, rpicam bloklanır ve işlenen kare fiilen
    # SANİYELER önce çekilmiş olabilirdi — ama frame_time "şimdi" kabul
    # edildiği için telemetri eşleşmesi (nearest_telemetry_at) dönüş/manevra
    # sırasında tamamen yanlış yaw/roll seçip GPS işaretini keyfi yönde
    # yüzlerce metre kaydırabiliyordu. Artık bu thread pipe'ı SÜREKLİ boşaltır
    # (backpressure yok), işleme döngüsü her turda yalnızca EN SON kareyi alır
    # (aradakiler bilinçli düşürülür) ve frame_time gerçek okuma anına yakındır.
    # rpicam ÖLÜRSE (EOF): eskiden sonsuz "Eksik frame: 0/..." busy-loop'una
    # giriliyordu; artık loglanıp pipeline.restart_rpicam() ile süreç yeniden
    # başlatılır.
    _latest_frame = {"buf": None, "ts": 0.0, "seq": 0}
    _frame_cond   = threading.Condition()

    def _frame_reader():
        while not state.shutdown_requested.is_set():
            proc = state.rpicam_process
            if proc is None or proc.stdout is None:
                time.sleep(0.2)
                continue
            try:
                buf = proc.stdout.read(frame_size)
            except Exception as e:
                log.info(f"[VISION] Frame okuma hatası: {e}")
                buf = b""
            ts = time.monotonic()
            if buf is None or len(buf) != frame_size:
                if state.shutdown_requested.is_set():
                    return
                log.info(f"[VISION] ⚠ rpicam akışı kesildi "
                         f"({0 if not buf else len(buf)}/{frame_size} byte) — yeniden başlatılıyor")
                pipeline.restart_rpicam()
                time.sleep(1.0)
                continue
            with _frame_cond:
                _latest_frame["buf"] = buf
                _latest_frame["ts"]  = ts
                _latest_frame["seq"] += 1
                _frame_cond.notify_all()

    threading.Thread(target=_frame_reader, daemon=True).start()
    _last_seq = 0
    gps_data      = {}      # {color: (lat, lon)} — HUD için tracking'teki en-iyi örneği yansıtır
    queued_colors = set()   # her renk yalnızca bir kez kuyruğa girer (kilitlendi)
    tracking      = {"mavi": None, "kirmizi": None}  # {color: {"lat","lon","alt","dist","worse_streak","miss_streak","first_seen"} | None}
    trackers      = {"mavi": None, "kirmizi": None}  # {color: {"cv": MOSSE tracker, "bridged": int} | None} — bkz. _update_detection
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

    last_staleness_warn = 0.0

    while not state.shutdown_requested.is_set():
        try:
            # En son kareyi bekle/al — okuyucu thread pipe'ı sürekli boşaltıyor,
            # burada yalnızca en taze kare işlenir (geride kalanlar düşürülür).
            # frame_time = karenin pipe'tan OKUNDUĞU an (sensör anına, ISP/pipe
            # gecikmesi hariç, pratikte en yakın ölçülebilir an) — telemetri
            # eşlemesi bu ana göre (bkz. state.nearest_telemetry_at).
            with _frame_cond:
                if _latest_frame["seq"] == _last_seq:
                    _frame_cond.wait(timeout=0.5)
                if _latest_frame["seq"] == _last_seq:
                    continue  # hâlâ yeni kare yok (rpicam restart olabilir)
                raw_frame  = _latest_frame["buf"]
                frame_time = _latest_frame["ts"]
                _last_seq  = _latest_frame["seq"]

            # I420 (YUV420 planar) → BGR. rpicam-vid --codec yuv420 çıktısı bu
            # düzende: Y düzlemi (H×W) + U/V düzlemleri (H/2×W/2 her biri),
            # toplam H*1.5 satır × W sütun olarak yorumlanabilir (OpenCV'nin
            # standart I420 kabulü budur).
            yuv = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (config.HEIGHT * 3 // 2, config.WIDTH)
            )
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            if undistort_map1 is not None:
                frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)

            # FPS hesabı
            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_time    = now

            # Tespit-anı telemetrisi: _update_detection'ın anlık pixel_to_gps hesabı
            # için burada okunuyor. ESKİDEN "şu an ne varsa o" (state.current_telemetry)
            # okunuyordu — bu, frame'in gerçek yakalama anıyla telemetri okuma anı
            # arasındaki pipeline gecikmesi kadar YANLIŞ (stale) veri demekti,
            # özellikle hızlı dönüş/manevra sırasında onlarca-yüzlerce metrelik
            # hataya yol açabiliyordu (checklist "Öncelik 3"). Artık frame_time'a
            # EN YAKIN konum/duruş örnekleri ayrı ayrı seçiliyor (bkz. state.py
            # nearest_telemetry_at). Henüz hiç telemetri gelmediyse (açılış anı)
            # eski None-doldurma davranışına düşülüyor (2026-08-16).
            tel = state.nearest_telemetry_at(frame_time)
            if tel is None:
                tel = {"lat": None, "lon": None, "alt": None,
                       "yaw": None, "roll": None, "pitch": None}
            else:
                staleness = max(tel["pos_age_s"], tel["att_age_s"])
                if staleness > 0.15 and now - last_staleness_warn > 5.0:
                    log.info(f"[VISION] ⚠ telemetri eşleşmesi {staleness*1000:.0f}ms eski "
                             f"(pos={tel['pos_age_s']*1000:.0f}ms att={tel['att_age_s']*1000:.0f}ms)")
                    last_staleness_warn = now

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Mavi kare tespiti
            mask_blue = _apply_morph(cv2.inRange(hsv, config.BLUE_HSV_LOWER, config.BLUE_HSV_UPPER))
            _update_detection(mask_blue, "mavi", queue, queued_colors, tel, tracking, frame, trackers)

            # Kırmızı kare tespiti (iki HSV aralığı OR'lanır)
            mask_red = _apply_morph(cv2.bitwise_or(
                cv2.inRange(hsv, config.RED1_HSV_LOWER, config.RED1_HSV_UPPER),
                cv2.inRange(hsv, config.RED2_HSV_LOWER, config.RED2_HSV_UPPER),
            ))
            _update_detection(mask_red, "kirmizi", queue, queued_colors, tel, tracking, frame, trackers)

            for _color in ("mavi", "kirmizi"):
                _t = tracking[_color]
                if _t is not None:
                    gps_data[_color] = (_t["lat"], _t["lon"])

            # Tespit (yukarıda) hep native frame'de yapıldı — kalibrasyon ve
            # geo.py matematiği için gerekli. Overlay'in TAMAMI ise (kontur +
            # tüm HUD metinleri) buradan itibaren stream_frame'e (yayına giden,
            # gerekirse çevrilmiş kare) çiziliyor — böylece HUD metinleri de
            # video içeriğiyle AYNI, doğru yönde görünür.
            stream_frame = (
                cv2.rotate(frame, cv2.ROTATE_180)
                if config.CAMERA_ROTATION_DEG == 180 else frame
            )

            # === OVERLAY (stream_frame üzerine, native koordinatlar dönüştürülerek) ===
            for color, data in state.detected_targets.items():
                if data:
                    box_color = (255, 100, 0) if color == "mavi" else (0, 0, 255)
                    cv2.drawContours(stream_frame, [_to_stream_contour(data["contour"])], -1, box_color, 3)
                    if color in gps_data:
                        lat, lon = gps_data[color]
                        sx, sy = _to_stream_xy(data["cx"], data["cy"])
                        cv2.putText(
                            stream_frame, f"{color.upper()}: {lat:.6f}, {lon:.6f}",
                            (sx - 100, sy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                        )

            cv2.putText(stream_frame, f"FPS: {current_fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if tel["alt"] is not None:
                cv2.putText(stream_frame, f"ALT: {tel['alt']:.1f}m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Anlık uçak GPS konumu (2026-08-20) — log'a bakmaya gerek kalmadan
            # HUD'dan takip edilebilsin diye. tel zaten frame anına en yakın
            # örneği taşıyor (bkz. state.nearest_telemetry_at).
            if tel["lat"] is not None and tel["lon"] is not None:
                cv2.putText(stream_frame, f"POS: {tel['lat']:.6f}, {tel['lon']:.6f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            wp = state.current_wp
            if wp["index"] is not None:
                cv2.putText(stream_frame, f"WP: {wp['index']}/{wp['total']}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Bu WP'ye GEÇİLDİĞİ ANDAKİ konum (2026-08-20) — POS'tan farkı:
                # "şu an" değil, "bu WP'ye geçerken neredeydik" (bkz.
                # mission.waypoint_tracking_task, state.current_wp).
                if wp["lat"] is not None and wp["lon"] is not None:
                    cv2.putText(
                        stream_frame, f"WP{wp['index']} GPS: {wp['lat']:.6f}, {wp['lon']:.6f}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    )

            fc_ok = state.fc_connected
            cv2.putText(
                stream_frame,
                f"FC: {'BAGLI' if fc_ok else 'BAGLI DEGIL'} ({config.FC_BAUDRATE})",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if fc_ok else (0, 0, 255), 2,
            )

            rf_ok = state.wfb_process is not None and state.wfb_process.poll() is None
            cv2.putText(
                stream_frame,
                f"RF: {'OK' if rf_ok else 'KOPTU'}",
                (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if rf_ok else (0, 0, 255), 2,
            )

            for color, label, pos, col_ok, col_no in [
                ("mavi",    "MAVI: OK",    (config.WIDTH - 160, 30), (255, 100, 0), (128, 128, 128)),
                ("kirmizi", "KIRMIZI: OK", (config.WIDTH - 160, 60), (0, 0, 255),   (128, 128, 128)),
            ]:
                detected = state.detected_targets[color]
                cv2.putText(
                    stream_frame,
                    label if detected else label.replace("OK", "--"),
                    pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    col_ok if detected else col_no, 2,
                )

            # Kilitlenen hedefler (KALICI GPS, 2026-08-20) + gönderilen servo
            # komutları ("SERVO AÇILDI") — ekranın ALT-SOL köşesinde, SABİT
            # pozisyonlarda tek blok halinde (üstteki FPS/ALT/POS/WP/FC/RF
            # bloğuyla ve sağ üstteki MAVI/KIRMIZI OK göstergesiyle çakışmasın
            # diye; ayrıca satırlar renk bazında SABİT yerde kalsın ki bir
            # olay gelip geçtikçe diğer satırlar yer değiştirmesin).
            # detected_targets'tan farkı: locked/servo canlı tespitten
            # bağımsız — hedef kadrajdan çıksa/tracker kaybolsa da kalır
            # (bkz. state.locked_targets/servo_events, mission.py).
            _hud_slots = (
                ("mavi",    state.locked_targets["mavi"],
                 lambda lat, lon: f"MAVI KILIT: {lat:.6f},{lon:.6f}",
                 (255, 100, 0), config.HEIGHT - 140),
                ("kirmizi", state.locked_targets["kirmizi"],
                 lambda lat, lon: f"KIRMIZI KILIT: {lat:.6f},{lon:.6f}",
                 (0, 0, 255), config.HEIGHT - 108),
            )
            for color, locked, fmt, box_color, y in _hud_slots:
                if locked is not None:
                    lat, lon = locked
                    cv2.putText(stream_frame, fmt(lat, lon), (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            for color, y in (("mavi", config.HEIGHT - 70), ("kirmizi", config.HEIGHT - 45)):
                ev = state.servo_events[color]
                if ev is not None:
                    lat_str = f"{ev['lat']:.6f}" if ev["lat"] is not None else "?"
                    lon_str = f"{ev['lon']:.6f}" if ev["lon"] is not None else "?"
                    cv2.putText(
                        stream_frame,
                        f"SERVO ACILDI: {color.upper()} kanal={ev['channel']} @ {lat_str},{lon_str}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2,
                    )

            try:
                _encode_queue.put_nowait(stream_frame.tobytes())
            except _queue.Full:
                pass  # encoder meşgul, bu frame'i düş

            # Periyodik durum özeti (her 5 saniyede bir)
            if now - debug_timer >= 5.0:
                try:
                    cv2.imwrite("/home/albatros/logs/hud_preview.jpg", stream_frame)
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

                # Tanı: tespit edilemeyen renkler için RENK mi ŞEKİL mi sorunu ayır
                for _color, _mask in (("mavi", mask_blue), ("kirmizi", mask_red)):
                    if state.detected_targets[_color]:
                        continue
                    info = _mask_debug_info(_mask)
                    if info["candidate"] is None:
                        log.info(f"[VISION][DEBUG] {_color}: maskede HİÇ piksel/kontur yok "
                              f"(pixel={info['pixels']}) — HSV aralığı bu kareyi hiç yakalamıyor")
                    else:
                        c = info["candidate"]
                        log.info(f"[VISION][DEBUG] {_color}: en büyük aday alan={c['area']} "
                              f"(aralıkta={c['in_area_range']}, MIN={config.MIN_AREA} MAX={config.MAX_AREA}) "
                              f"köşe={c['corners']} dışbükey={c['convex']} "
                              f"oran={c['ratio']} doluluk={c['extent']} "
                              f"(pixel_toplam={info['pixels']})")
                if tel["lat"] is not None:
                    log.info(f"[VISION][STATUS] Telemetri: "
                          f"lat={tel['lat']:.6f} lon={tel['lon']:.6f} "
                          f"alt={tel['alt']:.1f}m yaw={tel['yaw']:.1f}°")
                # GPS/EKF durumu (checklist "GPS ve heading" — 2026-08-16): tespit
                # anında bunlar şüpheliyse (fix_type düşük, global_position_ok=False)
                # o tespite güvenilmemeli, offline analizde bu satırlar aranmalı.
                gh, eh = state.gps_health, state.ekf_health
                if gh["fix_type"] is not None:
                    log.info(f"[VISION][STATUS] GPS/EKF: uydu={gh['num_satellites']} "
                          f"fix={gh['fix_type']} global_ok={eh['global_position_ok']} "
                          f"local_ok={eh['local_position_ok']} home_ok={eh['home_position_ok']}")
                debug_timer = now

        except Exception as e:
            log.info(f"[VISION] HATA: {e}")
            time.sleep(0.1)

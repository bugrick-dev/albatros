"""Video pipeline: WFB-ng, rpicam-vid başlatma/durdurma."""
import logging
import subprocess
import shlex
import threading
import time
import config
import state

log = logging.getLogger("pipeline")
WFB_TX_LOG_PATH = "/home/albatros/logs/wfb_tx.log"
RPICAM_LOG_PATH = "/home/albatros/logs/rpicam.log"


def _find_iface_by_mac():
    """Tek seferlik 'ip link' taraması; bulamazsa None döner."""
    result = subprocess.run(["ip", "link"], capture_output=True, text=True)
    log.info(f"[PIPELINE] 'ip link' çıktısı ({len(result.stdout)} karakter)")
    lines = result.stdout.splitlines()

    for i, line in enumerate(lines):
        if config.WFB_MAC.lower() in line.lower():
            log.info(f"[PIPELINE] MAC satırı bulundu (indeks {i}): {line.strip()}")
            prev  = lines[i - 1] if i > 0 else ""
            parts = prev.split(": ")
            if len(parts) >= 2:
                iface = parts[1].split("@")[0].strip()
                log.info(f"[PIPELINE] Önceki satır: {prev.strip()} → arayüz={iface}")
                return iface
    return None


def setup_monitor_mode(retry_timeout=20, retry_interval=1.0):
    """WiFi arayüzünü MAC adresine göre bulur ve monitor mode'a alır.

    Batarya/BEC beslemesinde açılışta gerilim toparlanana kadar USB WiFi
    adaptörü geç enumerate olabilir — bu yuzden tek seferlik kontrol yerine
    sinirli sure tekrar denenir.
    """
    log.info(f"[PIPELINE] WiFi arayüzü aranıyor (MAC={config.WFB_MAC})...")

    try:
        iface = None
        start = time.time()
        attempt = 0
        while time.time() - start < retry_timeout:
            attempt += 1
            iface = _find_iface_by_mac()
            if iface:
                break
            log.info(f"[PIPELINE] MAC {config.WFB_MAC} bulunamadı (deneme {attempt}), "
                  f"{retry_interval}s sonra tekrar denenecek...")
            time.sleep(retry_interval)

        if not iface:
            log.info(f"[PIPELINE] HATA: MAC {config.WFB_MAC} {retry_timeout}s içinde hiçbir satırda bulunamadı!")
            return None

        log.info(f"[PIPELINE] ✓ Arayüz: {iface}")

    except Exception as e:
        log.info(f"[PIPELINE] HATA: ip link çalıştırılamadı: {e}")
        return None

    log.info("[PIPELINE] rfkill unblock + NetworkManager durduruluyor...")
    subprocess.run(["sudo", "rfkill", "unblock", "all"], check=False)
    subprocess.run(["sudo", "systemctl", "stop", "NetworkManager"], check=False)
    time.sleep(1)

    cmds = [
        ["sudo", "ip",  "link", "set", iface, "down"],
        ["sudo", "iw",  "dev",  iface, "set", "monitor", "none"],
        ["sudo", "ip",  "link", "set", iface, "up"],
        ["sudo", "iw",  "dev",  iface, "set", "channel", str(config.WFB_CHANNEL), "HT20"],
        ["sudo", "iw",  "dev",  iface, "set", "txpower", "fixed", str(config.WFB_TXPOWER_MBM)],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        status = "✓" if r.returncode == 0 else f"✗ (kod={r.returncode})"
        log.info(f"[PIPELINE] {status}  {' '.join(cmd[2:])}")
        if r.returncode != 0 and r.stderr:
            log.info(f"[PIPELINE]   stderr: {r.stderr.strip()}")

    log.info(f"[PIPELINE] ✓ Monitor mode aktif: {iface} | kanal={config.WFB_CHANNEL}")
    return iface


def _wfb_cmd_str(iface):
    return (
        f"sudo wfb_tx -K {config.WFB_KEY_PATH} "
        f"-i {config.WFB_LINK_ID} -p 0 -u {config.WFB_UDP_PORT} "
        f"-M {config.WFB_MCS} -B {config.WFB_BANDWIDTH} -G {config.WFB_GUARD_INTERVAL} "
        f"-L {config.WFB_LDPC} -k {config.WFB_FEC_K} -n {config.WFB_FEC_N} {iface}"
    )


def _iface_exists(iface):
    r = subprocess.run(["ip", "link", "show", iface], capture_output=True, text=True)
    return r.returncode == 0


def _rearm_monitor(iface):
    cmds = [
        ["sudo", "ip", "link", "set", iface, "down"],
        ["sudo", "iw", "dev", iface, "set", "monitor", "none"],
        ["sudo", "ip", "link", "set", iface, "up"],
        ["sudo", "iw", "dev", iface, "set", "channel", str(config.WFB_CHANNEL), "HT20"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=False, capture_output=True, text=True)


def _restart_wfb(iface):
    if state.wfb_process:
        try:
            state.wfb_process.kill()
        except Exception:
            pass
    try:
        wfb_out = open(WFB_TX_LOG_PATH, "ab")
        state.wfb_process = subprocess.Popen(
            shlex.split(_wfb_cmd_str(iface)), stdout=wfb_out, stderr=subprocess.STDOUT
        )
        log.info(f"[WATCHDOG] wfb_tx yeniden başlatıldı → PID={state.wfb_process.pid}")
    except Exception as e:
        log.info(f"[WATCHDOG] wfb_tx yeniden başlatma HATA: {e}")


def _wfb_watchdog(iface):
    """Arayuz (USB PM buglu surucu nedeniyle) kaybolursa ya da wfb_tx olurse
    otomatik olarak monitor mode + wfb_tx'i yeniden kurar."""
    log.info("[WATCHDOG] wfb_tx/arayüz izleyici başladı")
    missing_ticks = 0
    while not state.shutdown_requested.is_set():
        time.sleep(0.5)
        iface_ok = _iface_exists(iface)
        proc_ok = state.wfb_process is not None and state.wfb_process.poll() is None
        if iface_ok and proc_ok:
            missing_ticks = 0
            continue
        missing_ticks += 1
        if missing_ticks < 2:
            continue
        if not iface_ok:
            for _ in range(20):
                if state.shutdown_requested.is_set():
                    return
                if _iface_exists(iface):
                    break
                time.sleep(0.3)
            else:
                continue
        log.info("[WATCHDOG] Arayüz/wfb_tx sorunlu — yeniden kuruluyor...")
        _rearm_monitor(iface)
        _restart_wfb(iface)
        missing_ticks = 0
    log.info("[WATCHDOG] Kapanış sinyali alındı, izleyici durdu")


def _spawn_rpicam():
    rpicam_cmd = (
        f"rpicam-vid -t 0 --codec yuv420 "
        f"--width {config.WIDTH} --height {config.HEIGHT} --framerate {config.FPS} "
        f"-o -"
    )
    log.info(f"[PIPELINE] rpicam-vid komutu: {rpicam_cmd}")
    rpicam_stderr = open(RPICAM_LOG_PATH, "ab")
    state.rpicam_process = subprocess.Popen(
        shlex.split(rpicam_cmd), stdout=subprocess.PIPE, stderr=rpicam_stderr
    )
    log.info(f"[PIPELINE] rpicam-vid başladı → PID={state.rpicam_process.pid} (raw stdout)")


def restart_rpicam():
    """rpicam-vid öldüğünde (EOF — bkz. vision._frame_reader) süreci yeniden
    başlatır. Eski davranış: EOF sonsuz 'Eksik frame' busy-loop'una dönüyordu
    ve video bir daha gelmiyordu (watchdog yalnızca wfb_tx'i izliyordu)."""
    old = state.rpicam_process
    if old is not None:
        try:
            if old.poll() is None:
                old.kill()
            old.wait(timeout=2)
        except Exception as e:
            log.info(f"[PIPELINE] Eski rpicam süreci kapatılamadı: {e}")
    try:
        _spawn_rpicam()
    except Exception as e:
        log.info(f"[PIPELINE] rpicam yeniden başlatma HATA: {e}")


def start_pipeline(iface):
    """WFB-ng transmitter ve rpicam-vid süreçlerini başlatır."""
    log.info("\n[PIPELINE] Pipeline başlatılıyor...")

    # WFB-ng
    wfb_cmd = (
        f"sudo wfb_tx -K {config.WFB_KEY_PATH} "
        f"-i {config.WFB_LINK_ID} -p 0 -u {config.WFB_UDP_PORT} "
        f"-M {config.WFB_MCS} -B {config.WFB_BANDWIDTH} -G {config.WFB_GUARD_INTERVAL} "
        f"-L {config.WFB_LDPC} -k {config.WFB_FEC_K} -n {config.WFB_FEC_N} {iface}"
    )
    log.info(f"[PIPELINE] wfb_tx komutu: {wfb_cmd}")
    wfb_out = open(WFB_TX_LOG_PATH, "wb")
    state.wfb_process = subprocess.Popen(shlex.split(wfb_cmd), stdout=wfb_out, stderr=subprocess.STDOUT)
    log.info(f"[PIPELINE] wfb_tx başladı → PID={state.wfb_process.pid} | log={WFB_TX_LOG_PATH}")
    time.sleep(2)

    # rpicam-vid — NOT: --rotation KASITLI olarak kullanılmıyor. Kamera fiziksel
    # olarak ters monte edilse bile capture native (ham) yönde kalır; tespit/GPS
    # matematiği native frame üzerinde çalışır (geo.py CAMERA_ROTATION_DEG ile
    # telafi eder), yalnızca vision.py yayına giden kareyi çeviriyor (bkz.
    # vision.py _encode_queue). Böylece kalibrasyon da native yönde kalıp
    # değişmeden geçerliliğini korur (2026-08-09).
    #
    # RAW YUV420 çıktı (stdout, -o -) — ESKİDEN --codec h264 + --listen TCP ile
    # H264 üretilip vision.py'de ffmpeg ile tekrar raw'a DECODE ediliyordu, sonra
    # OpenCV işleyip TEKRAR H264'e encode ediyordu (çift encode/decode, gereksiz
    # CPU + gecikme). Artık rpicam-vid doğrudan raw veriyor, vision.py TEK bir
    # ffmpeg (encode, transmisyon için) kullanıyor (2026-08-16, çift codec turu
    # kaldırıldı — bkz. vision.py cv2.cvtColor I420->BGR notu).
    _spawn_rpicam()
    time.sleep(5)

    watchdog_thread = threading.Thread(target=_wfb_watchdog, args=(iface,), daemon=True)
    watchdog_thread.start()

    log.info("[PIPELINE] ✓ Pipeline hazır (FFmpeg + GStreamer vision thread'inde başlayacak)\n")
    return True


def stop_pipeline():
    """Tüm video pipeline süreçlerini düzgün şekilde sonlandırır."""
    log.info("[PIPELINE] Pipeline durduruluyor...")
    procs = [
        (state.gst_process,           "GStreamer"),
        (state.ffmpeg_encode_process, "FFmpeg-enc"),
        (state.rpicam_process,        "rpicam-vid"),
        (state.wfb_process,           "wfb_tx"),
    ]
    for proc, name in procs:
        if proc:
            try:
                proc.terminate()
                log.info(f"[PIPELINE] {name} → terminate() gönderildi (PID={proc.pid})")
            except Exception as e:
                log.info(f"[PIPELINE] {name} → terminate() HATA: {e}")
        else:
            log.info(f"[PIPELINE] {name} → zaten None, atlandı")
    log.info("[PIPELINE] Süreçlerin gerçekten kapanması bekleniyor...")
    for proc, name in procs:
        if proc:
            try:
                proc.wait(timeout=3)
                log.info(f"[PIPELINE] {name} → kapandı")
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                    log.info(f"[PIPELINE] {name} → terminate yetersizdi, kill() ile zorlandı")
                except Exception as e:
                    log.info(f"[PIPELINE] {name} → kill() HATA: {e}")
            except Exception:
                pass
    # Guvenlik agi: isme gore de zorla temizle (shell/sudo nedeniyle PID kaybi olabilir)
    for pname in ("rpicam-vid", "wfb_tx"):
        try:
            subprocess.run(["sudo", "pkill", "-9", "-f", pname],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    log.info("[PIPELINE] ✓ Tüm süreçlere terminate gönderildi")

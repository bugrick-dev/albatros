"""Video pipeline: WFB-ng, rpicam-vid başlatma/durdurma."""
import subprocess
import time
import config
import state


def setup_monitor_mode():
    """WiFi arayüzünü MAC adresine göre bulur ve monitor mode'a alır."""
    print(f"[PIPELINE] WiFi arayüzü aranıyor (MAC={config.WFB_MAC})...")

    try:
        result = subprocess.run(["ip", "link"], capture_output=True, text=True)
        print(f"[PIPELINE] 'ip link' çıktısı ({len(result.stdout)} karakter)")
        iface = None
        lines = result.stdout.splitlines()

        for i, line in enumerate(lines):
            if config.WFB_MAC.lower() in line.lower():
                print(f"[PIPELINE] MAC satırı bulundu (indeks {i}): {line.strip()}")
                prev  = lines[i - 1] if i > 0 else ""
                parts = prev.split(": ")
                if len(parts) >= 2:
                    iface = parts[1].split("@")[0].strip()
                    print(f"[PIPELINE] Önceki satır: {prev.strip()} → arayüz={iface}")
                break

        if not iface:
            print(f"[PIPELINE] HATA: MAC {config.WFB_MAC} hiçbir satırda bulunamadı!")
            return None

        print(f"[PIPELINE] ✓ Arayüz: {iface}")

    except Exception as e:
        print(f"[PIPELINE] HATA: ip link çalıştırılamadı: {e}")
        return None

    print("[PIPELINE] rfkill unblock + NetworkManager durduruluyor...")
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
        print(f"[PIPELINE] {status}  {' '.join(cmd[2:])}")
        if r.returncode != 0 and r.stderr:
            print(f"[PIPELINE]   stderr: {r.stderr.strip()}")

    print(f"[PIPELINE] ✓ Monitor mode aktif: {iface} | kanal={config.WFB_CHANNEL}")
    return iface


def start_pipeline(iface):
    """WFB-ng transmitter ve rpicam-vid süreçlerini başlatır."""
    print("\n[PIPELINE] Pipeline başlatılıyor...")

    # WFB-ng
    wfb_cmd = (
        f"sudo wfb_tx -K {config.WFB_KEY_PATH} "
        f"-i {config.WFB_LINK_ID} -p 0 -u {config.WFB_UDP_PORT} "
        f"-M {config.WFB_MCS} -B {config.WFB_BANDWIDTH} -G {config.WFB_GUARD_INTERVAL} "
        f"-L {config.WFB_LDPC} -k {config.WFB_FEC_K} -n {config.WFB_FEC_N} {iface}"
    )
    print(f"[PIPELINE] wfb_tx komutu: {wfb_cmd}")
    state.wfb_process = subprocess.Popen(wfb_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[PIPELINE] wfb_tx başladı → PID={state.wfb_process.pid}")
    time.sleep(2)

    # rpicam-vid
    rpicam_cmd = (
        f"rpicam-vid -t 0 --inline --codec h264 "
        f"--width {config.WIDTH} --height {config.HEIGHT} --framerate {config.FPS} "
        f"--bitrate {config.BITRATE} --intra {config.INTRA} "
        f"--listen -o tcp://127.0.0.1:{config.RPICAM_TCP_PORT}"
    )
    print(f"[PIPELINE] rpicam-vid komutu: {rpicam_cmd}")
    state.rpicam_process = subprocess.Popen(rpicam_cmd, shell=True, stderr=subprocess.DEVNULL)
    print(f"[PIPELINE] rpicam-vid başladı → PID={state.rpicam_process.pid} | TCP:{config.RPICAM_TCP_PORT}")
    time.sleep(5)

    print("[PIPELINE] ✓ Pipeline hazır (FFmpeg + GStreamer vision thread'inde başlayacak)\n")
    return True


def stop_pipeline():
    """Tüm video pipeline süreçlerini düzgün şekilde sonlandırır."""
    print("[PIPELINE] Pipeline durduruluyor...")
    procs = [
        (state.gst_process,           "GStreamer"),
        (state.ffmpeg_encode_process, "FFmpeg-enc"),
        (state.ffmpeg_decode_process, "FFmpeg-dec"),
        (state.rpicam_process,        "rpicam-vid"),
        (state.wfb_process,           "wfb_tx"),
    ]
    for proc, name in procs:
        if proc:
            try:
                proc.terminate()
                print(f"[PIPELINE] {name} → terminate() gönderildi (PID={proc.pid})")
            except Exception as e:
                print(f"[PIPELINE] {name} → terminate() HATA: {e}")
        else:
            print(f"[PIPELINE] {name} → zaten None, atlandı")
    print("[PIPELINE] ✓ Tüm süreçlere terminate gönderildi")

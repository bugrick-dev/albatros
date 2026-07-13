#!/usr/bin/env python3
"""
SERVO TETİKLEME TESTİ — uçuş gerektirmez, elle çalıştırılır.
main.py ile AYNI video pipeline'ını kullanır:
  rpicam-vid (TCP) → FFmpeg decode → OpenCV (tespit+overlay) → FFmpeg encode → GStreamer → WFB-ng
asıl koddan TEK FARK: FC/mission YOK, GPS mesafe kontrolü YOK.
  Kare kamerada görülür görülmez ilgili servo tetiklenir
  (şartname: mavi hedefe kırmızı boyalı yük, kırmızı hedefe mavi boyalı yük).

Ne yapar:
  1. Servo GPIO başlatır — pigpio ile (asıl koddaki servo.py'nin RPi.GPIO
     yazılımsal PWM'i DEĞİL, bkz. NOT aşağıda).
  2. WiFi monitor mode + WFB-ng + rpicam-vid başlatır (pipeline.start_pipeline).
  3. vision.opencv_processing_thread'i başlatır — HSV kare tespiti + WFB üzerinden
     görüntü yayını (bounding box dahil, telemetri overlay'i FC olmadığı için boş kalır).
  4. state.detected_targets sürekli izlenir (her karede güncellenir): kare
     ekranda GÖRÜNDÜĞÜ SÜRECE ilgili servo AÇIK kalır, kare kaybolur kaybolmaz
     servo NÖTR konuma döner. Tek seferlik "aç-kapa" darbesi DEĞİL — durum
     kareye bağlı sürekli takip edilir.
  5. Ctrl+C ile çıkılana kadar çalışır.

NOT — neden pigpio, neden servo.py (ana kod) DEĞİL:
  RPi.GPIO'nun PWM'i yazılımsaldır (CPU zamanlamasıyla darbe üretir). Bu script
  aynı anda rpicam-vid + 2x ffmpeg + gstreamer + wfb_tx + OpenCV thread'ini
  çalıştırdığı için CPU yükü altında darbe genişliği düzensizleşiyor, bu da
  servonun kare görülmeden ÖNCE bile seğirmesine ve ısınmasına yol açıyordu.
  pigpio darbeleri DMA üzerinden üretir — CPU meşgul olsa da jitter'a girmez.
  Bu düzeltme kasıtlı olarak yalnızca bu test dosyasında; servo.py/main.py'ye
  DOKUNULMADI, orada hâlâ RPi.GPIO kullanılıyor (test sonucu doğrulanmadan
  ana kod değiştirilmeyecek).

Gereksinim (bu Pi'de KURULU ve pigpiod servisi ETKİN — bkz. not):
  Debian Trixie'nin apt deposu artık pigpiod daemon'unu paketlemiyor (yalnızca
  istemci araçlarını içeriyor), bu yüzden pigpio kaynak koddan derlenip
  kuruldu ve /etc/systemd/system/pigpiod.service ile `systemctl enable --now`
  yapıldı (boot'ta otomatik başlar). Sıfırdan bir Pi'de tekrar kurmak gerekirse:
    git clone https://github.com/joan2937/pigpio.git && cd pigpio
    make -j$(nproc) && sudo make install
    sudo cp util/pigpiod.service /etc/systemd/system/  # ExecStart'ı
      /usr/local/bin/pigpiod olacak şekilde düzenleyin
    sudo systemctl daemon-reload && sudo systemctl enable --now pigpiod

  ÖNEMLİ — PORT ÇAKIŞMASI: pigpiod varsayılan olarak TCP 8888'i dinler, ki bu
  config.RPICAM_TCP_PORT (rpicam-vid'in --listen ile bağladığı port) ile
  BİREBİR AYNI. pigpiod önce 8888'de ayağa kalkarsa rpicam-vid portu
  bağlayamaz ve WFB akışı hiç başlamaz ("failed to bind listen socket").
  Bu yüzden servis dosyasında pigpiod `-p 8779` ile başlatılıyor
  (bkz. _PIGPIOD_PORT aşağıda) — ASLA 8888'e geri almayın.

Çalıştırma:
    cd rasp/tests
    python3 test_servo_color_trigger.py

Notlar:
  - Gerçek Pi üzerinde çalıştırılmalı (kamera + pigpiod + WiFi adaptörü gerekiyor).
  - FC bağlantısı YOK — sadece kamera tespiti servo tetikler, GPS/mesafe şartı aranmaz.
"""
import asyncio
import sys
import threading

sys.stdout.reconfigure(line_buffering=True)

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pigpio

import config
import state
import pipeline
import vision


# ==================== TEST-LOKAL SERVO (pigpio, servo.py'nin yerine) ====================

_PW_NEUTRAL   = 1500   # us — servo.py'deki %7.5 duty (nötr) karşılığı
_PW_RELEASE   = 500   # us — servo.py'deki %12.5 duty (bırak) karşılığı
_PIGPIOD_PORT = 8779   # DİKKAT: varsayılan 8888, config.RPICAM_TCP_PORT ile ÇAKIŞIR!
                        # pigpiod bu Pi'de -p 8779 ile çalışacak şekilde kuruldu
                        # (/etc/systemd/system/pigpiod.service).

_pi = None


def init_servo():
    """pigpio ile servo başlatır — jitter-free, CPU yükünden etkilenmez."""
    global _pi
    print("[SERVO] pigpio ile servo başlatılıyor...")
    _pi = pigpio.pi("localhost", _PIGPIOD_PORT)
    if not _pi.connected:
        print("[SERVO] HATA: pigpiod'ye bağlanılamadı — "
              "'sudo systemctl start pigpiod' çalıştırıldı mı?")
        sys.exit(1)

    _pi.set_servo_pulsewidth(config.SERVO_KIRMIZI_YUK_PIN, _PW_NEUTRAL)
    print(f"[SERVO] Kırmızı yük → GPIO {config.SERVO_KIRMIZI_YUK_PIN} | nötr ({_PW_NEUTRAL}us)")

    _pi.set_servo_pulsewidth(config.SERVO_MAVI_YUK_PIN, _PW_NEUTRAL)
    print(f"[SERVO] Mavi yük    → GPIO {config.SERVO_MAVI_YUK_PIN} | nötr ({_PW_NEUTRAL}us)")

    print("[SERVO] ✓ Tüm servolar nötr konumda hazır (pigpio)")


def _servo_pin_for_target(hedef_rengi):
    """Şartname: mavi hedefe kırmızı boyalı yük, kırmızı hedefe mavi boyalı yük."""
    if hedef_rengi == "mavi":
        return config.SERVO_KIRMIZI_YUK_PIN, "KIRMIZI"
    return config.SERVO_MAVI_YUK_PIN, "MAVİ"


def open_servo(hedef_rengi):
    pin, yuk = _servo_pin_for_target(hedef_rengi)
    _pi.set_servo_pulsewidth(pin, _PW_RELEASE)
    print(f"[SERVO] {hedef_rengi.upper()} kare GÖRÜNDÜ → {yuk} servo AÇIK")


def close_servo(hedef_rengi):
    pin, yuk = _servo_pin_for_target(hedef_rengi)
    _pi.set_servo_pulsewidth(pin, _PW_NEUTRAL)
    print(f"[SERVO] {hedef_rengi.upper()} kare KAYBOLDU → {yuk} servo NÖTR (kapandı)")


def cleanup_servo():
    print("[SERVO] Temizleniyor (pigpio)...")
    if _pi:
        _pi.set_servo_pulsewidth(config.SERVO_KIRMIZI_YUK_PIN, 0)
        _pi.set_servo_pulsewidth(config.SERVO_MAVI_YUK_PIN, 0)
        _pi.stop()
    print("[SERVO] ✓ pigpio temizlendi")


async def detection_watch_task():
    """
    state.detected_targets'i sürekli izler (her karede vision thread tarafından
    güncellenir). Kare göründüğü sürece servo AÇIK, kaybolunca NÖTR — durum
    değişmediği sürece tekrar komut göndermez (gereksiz pigpio çağrısı yok).
    """
    print("[TEST] Tespit izleme başladı — mavi/kırmızı kare bekleniyor")
    is_open = {"mavi": False, "kirmizi": False}
    while True:
        for color in ("mavi", "kirmizi"):
            visible = state.detected_targets[color] is not None
            if visible and not is_open[color]:
                open_servo(color)
                is_open[color] = True
            elif not visible and is_open[color]:
                close_servo(color)
                is_open[color] = False
        await asyncio.sleep(0.05)


def main():
    print("=" * 60)
    print("SERVO TETİKLEME TESTİ — yalnızca kare tespitiyle servo açma")
    print(f"  Video   : {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
    print(f"  WFB     : MAC={config.WFB_MAC}  kanal={config.WFB_CHANNEL}")
    print("=" * 60)

    # 1. GPIO servo başlat (pigpio ile — bkz. dosya başındaki NOT)
    init_servo()

    # 2. WiFi monitor mode
    iface = pipeline.setup_monitor_mode()
    if not iface:
        print("[TEST] Monitor mode kurulamadı — çıkılıyor")
        cleanup_servo()
        sys.exit(1)

    # 3. Video pipeline (WFB-ng + rpicam) — asıl koddaki gibi
    if not pipeline.start_pipeline(iface):
        print("[TEST] Pipeline başlatılamadı — çıkılıyor")
        cleanup_servo()
        sys.exit(1)

    # 4. OpenCV thread — tespit + bounding box + WFB yayını (asıl koddaki gibi)
    #    state.detected_targets her karede güncellenir; detection_watch_task
    #    bunu doğrudan okuyacağı için detection_active/target_queue GEREKMİYOR.
    print("[TEST] OpenCV thread başlatılıyor...")
    opencv_thread = threading.Thread(
        target=vision.opencv_processing_thread,
        args=(state.target_queue,),
        daemon=True,
    )
    opencv_thread.start()
    print(f"[TEST] OpenCV thread başladı (TID={opencv_thread.ident})")

    print("\n" + "=" * 60)
    print("✓ HAZIR — kare göründüğü sürece servo açık, kaybolunca nötre dönecek")
    print("  Ctrl+C ile çıkın")
    print("=" * 60 + "\n")

    try:
        asyncio.run(detection_watch_task())
    except KeyboardInterrupt:
        print("\n[TEST] Ctrl+C — durduruldu")
    finally:
        print("[TEST] Temizleniyor...")
        pipeline.stop_pipeline()
        cleanup_servo()
        print("[TEST] ✓ Temizlik tamamlandı")


if __name__ == "__main__":
    main()

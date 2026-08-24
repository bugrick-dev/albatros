"""
Donanım heartbeat LED'i (2026-08-23).

Uçağın içine fiziksel erişim kısıtlı olduğu için gövdeyi açmadan "sistem
donmuş mu" sorusuna dıştan görülebilir bir cevap vermek için eklendi.
Kernel'in kendi ACT LED/heartbeat trigger'ından FARKI: bu, Python tarafının
(vision döngüsünün) GERÇEKTEN ilerlediğini kanıtlıyor — kernel çalışıyor ama
uygulama (ör. bir deadlock/askıda kalma) donmuşsa kernel heartbeat'i yine de
yanıp sönmeye devam eder, bizimki ise DURUR.

Nasıl çalışır:
  - vision.py ana işleme döngüsü HER turda state.last_vision_tick'i
    time.monotonic() ile günceller (bkz. state.py notu).
  - Buradaki gözcü thread bu damganın config.HEARTBEAT_STALE_SEC'ten daha
    bayat olup olmadığını periyodik kontrol eder: taze ise LED yanıp söner,
    bayatsa (vision donmuş sayılır) LED SÖNER — yanıp sönmenin durması
    dışarıdan bakan biri için fark edilir bir anormallik sinyali.

GPIO donanımı yoksa/hatalıysa (pin takılı değil, lgpio erişilemiyor vb.)
start() SESSİZCE (tek log satırı hariç) devre dışı kalır — bu tamamen teşhis
amaçlı bir katman, uçuş yazılımının kendisini ASLA etkilememeli.
"""
import logging
import threading
import time

import config
import state

log = logging.getLogger("heartbeat")

_led = None


def _watchdog_loop():
    blinking = True
    while not state.shutdown_requested.is_set():
        tick = state.last_vision_tick
        alive = tick is not None and (time.monotonic() - tick) < config.HEARTBEAT_STALE_SEC

        if alive and not blinking:
            _led.blink(on_time=0.5, off_time=0.5)
            blinking = True
            log.info("[HEARTBEAT] ✓ vision döngüsü toparlandı — LED tekrar yanıp sönüyor")
        elif not alive and blinking:
            _led.off()
            blinking = False
            log.info("[HEARTBEAT] ⚠ vision döngüsü donmuş görünüyor "
                     f"(last_vision_tick {config.HEARTBEAT_STALE_SEC}s'den bayat) — LED söndürüldü")

        time.sleep(1.0)

    try:
        _led.off()
    except Exception:
        pass
    log.info("[HEARTBEAT] Kapanış sinyali alındı, gözcü durdu")


def start():
    """Heartbeat LED'ini başlatır. Hata durumunda uçuş yazılımını hiç
    etkilemeden sessizce devre dışı kalır (bkz. modül docstring)."""
    global _led
    try:
        from gpiozero import LED
        _led = LED(config.HEARTBEAT_LED_GPIO)
        _led.blink(on_time=0.5, off_time=0.5)
    except Exception as e:
        log.info(f"[HEARTBEAT] ⚠ LED başlatılamadı (GPIO{config.HEARTBEAT_LED_GPIO}) "
                 f"— heartbeat devre dışı, uçuş yazılımı ETKİLENMEDİ: {e}")
        return

    threading.Thread(target=_watchdog_loop, daemon=True).start()
    log.info(f"[HEARTBEAT] ✓ LED başlatıldı (GPIO{config.HEARTBEAT_LED_GPIO}, "
             f"donma eşiği={config.HEARTBEAT_STALE_SEC}s)")

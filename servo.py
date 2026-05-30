"""RPi GPIO servo kontrolü — yük bırakma mekanizması."""
import asyncio
import time
import RPi.GPIO as GPIO
import config
import state


def init_servo():
    print("[SERVO] GPIO servo başlatılıyor...")
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(config.SERVO_KIRMIZI_YUK_PIN, GPIO.OUT)
    state.servo_kirmizi_pwm = GPIO.PWM(config.SERVO_KIRMIZI_YUK_PIN, 50)
    state.servo_kirmizi_pwm.start(7.5)
    print(f"[SERVO] Kırmızı yük → GPIO {config.SERVO_KIRMIZI_YUK_PIN} | 50Hz PWM başladı (duty=7.5%)")

    GPIO.setup(config.SERVO_MAVI_YUK_PIN, GPIO.OUT)
    state.servo_mavi_pwm = GPIO.PWM(config.SERVO_MAVI_YUK_PIN, 50)
    state.servo_mavi_pwm.start(7.5)
    print(f"[SERVO] Mavi yük    → GPIO {config.SERVO_MAVI_YUK_PIN} | 50Hz PWM başladı (duty=7.5%)")

    print("[SERVO] ✓ Tüm servolar nötr konumda hazır")


def _drop_payload_sync(hedef_rengi):
    """
    Şartname: mavi hedefe kırmızı boyalı yük, kırmızı hedefe mavi boyalı yük.
    Sadece ayrı thread'den çağrılır (sync sleep içerir).
    """
    if hedef_rengi == "mavi":
        pwm = state.servo_kirmizi_pwm
        yuk = "KIRMIZI"
    else:
        pwm = state.servo_mavi_pwm
        yuk = "MAVİ"

    print(f"[SERVO] {yuk} boyalı yük bırakma başladı (hedef={hedef_rengi.upper()})")
    if not pwm:
        print(f"[SERVO] HATA: {yuk} PWM nesnesi None — init_servo() çağrıldı mı?")
        return

    print(f"[SERVO] duty → 12.5% (bırak konumu)")
    pwm.ChangeDutyCycle(12.5)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(7.5)
    print(f"[SERVO] duty → 7.5% (nötr) | {yuk} boyalı yük bırakıldı ✓")


async def drop_payload(hedef_rengi):
    """Async wrapper — event loop'u bloklamaz."""
    print(f"[SERVO] drop_payload() async çağrıldı: hedef={hedef_rengi}")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _drop_payload_sync, hedef_rengi)
    print(f"[SERVO] drop_payload() tamamlandı: hedef={hedef_rengi}")


def cleanup_servo():
    print("[SERVO] Temizleniyor...")
    if state.servo_kirmizi_pwm:
        state.servo_kirmizi_pwm.stop()
        print(f"[SERVO] GPIO {config.SERVO_KIRMIZI_YUK_PIN} durduruldu")
    if state.servo_mavi_pwm:
        state.servo_mavi_pwm.stop()
        print(f"[SERVO] GPIO {config.SERVO_MAVI_YUK_PIN} durduruldu")
    GPIO.cleanup()
    print("[SERVO] ✓ GPIO temizlendi")

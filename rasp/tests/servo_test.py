#!/usr/bin/env python3
"""
Pixhawk AUX1 / AUX2 servo ac-kapa testi (ArduPilot)

Kanal haritalama (ArduPilot):
  AUX OUT 1 -> MAVLink servo kanali 9   (SERVO9_*)
  AUX OUT 2 -> MAVLink servo kanali 10  (SERVO10_*)

ON KOSUL:
  - SERVO9_FUNCTION ve SERVO10_FUNCTION parametreleri "0 (Disabled)" OLMAMALI.
    Manuel test icin genelde "RCPassThru" (RC kanaline gore) ya da ilgili
    fonksiyona ayarlanmis olmali. Mission Planner / QGroundControl'den
    kontrol edilebilir.
  - Pixhawk'ta fiziksel safety switch varsa basili (aktif) olmali, aksi
    halde cikislar guc almaz.
  - Pervane/motor bagli DEGIL (bu test icin teyit edildi).
"""

import time
from pymavlink import mavutil

# --- Ayarlar ---
PORT = '/dev/ttyACM0'
BAUD = 115200

SERVO_AUX1 = 9
SERVO_AUX2 = 10

PWM_CLOSED = 1000   # "kapali" pozisyon (ihtiyaca gore degistir)
PWM_OPEN = 2000     # "acik" pozisyon (ihtiyaca gore degistir)

CYCLE_DELAY = 2      # saniye, acik/kapali arasi bekleme
CYCLES = 5           # kac kere ac-kapa yapilsin


def set_servo(master, channel, pwm):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,          # confirmation
        channel,    # param1: servo kanal numarasi
        pwm,        # param2: PWM degeri (us)
        0, 0, 0, 0, 0
    )


def main():
    print(f"Pixhawk'a baglaniliyor: {PORT} @ {BAUD} baud ...")
    master = mavutil.mavlink_connection(PORT, baud=BAUD)

    print("Heartbeat bekleniyor...")
    master.wait_heartbeat()
    print(f"Baglandi: system {master.target_system}, component {master.target_component}")

    print(f"{CYCLES} dongu boyunca AUX1 (servo{SERVO_AUX1}) ve AUX2 (servo{SERVO_AUX2}) test edilecek.")
    print("Durdurmak icin Ctrl+C.")

    try:
        for i in range(CYCLES):
            print(f"\n--- Dongu {i + 1}/{CYCLES} ---")

            print(f"AUX1 -> ACIK  ({PWM_OPEN})")
            set_servo(master, SERVO_AUX1, PWM_OPEN)
            print(f"AUX2 -> ACIK  ({PWM_OPEN})")
            set_servo(master, SERVO_AUX2, PWM_OPEN)
            time.sleep(CYCLE_DELAY)

            print(f"AUX1 -> KAPALI ({PWM_CLOSED})")
            set_servo(master, SERVO_AUX1, PWM_CLOSED)
            print(f"AUX2 -> KAPALI ({PWM_CLOSED})")
            set_servo(master, SERVO_AUX2, PWM_CLOSED)
            time.sleep(CYCLE_DELAY)

    except KeyboardInterrupt:
        print("\nTest durduruldu.")

    print("Test tamamlandi.")


if __name__ == '__main__':
    main()

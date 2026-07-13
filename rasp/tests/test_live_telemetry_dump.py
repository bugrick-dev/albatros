#!/usr/bin/env python3
"""
CANLI TELEMETRİ DÖKÜMÜ — uçuş gerektirmez, elle çalıştırılır.
MAVSDK/mission_raw YOK — pymavlink ile doğrudan bağlanıp yalnızca FC'nin
kendiliğinden yayınladığı telemetriyi (GLOBAL_POSITION_INT, ATTITUDE,
GPS_RAW_INT, VFR_HUD, SYS_STATUS, MISSION_CURRENT) okuyup ekrana basar.
Hiçbir komut/istek göndermez — salt okunur bir izleme aracıdır, bu yüzden
FC'ye giden yönün (parametre/misyon isteklerinin) çalışıp çalışmadığından
bağımsız olarak, yalnızca gelen telemetrinin canlı ve sağlıklı olup
olmadığını doğrulamak için kullanılır.

Çalıştırma:
    cd rasp/tests
    python3 test_live_telemetry_dump.py [--port /dev/ttyAMA3] [--baud 115200]

Notlar:
  - Ctrl+C ile çıkın.
  - --baud, config.FC_BAUDRATE'ten farklı denemek isterseniz (ör. FC'nin
    hangi baud'da olduğunu doğrularken) override edilebilir.
"""
import argparse
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymavlink import mavutil

import config


def main(port, baud):
    print(f"[TELEMETRY] Bağlanılıyor: {port} @ {baud}")
    m = mavutil.mavlink_connection(port, baud=baud)

    print("[TELEMETRY] HEARTBEAT bekleniyor (10s)...")
    hb = m.wait_heartbeat(timeout=10)
    if not hb:
        print("[TELEMETRY] ✗ HEARTBEAT gelmedi — bağlantı/baud kontrol edin")
        sys.exit(1)
    print(f"[TELEMETRY] ✓ sys={m.target_system} comp={m.target_component} "
          f"type={hb.type} autopilot={hb.autopilot} base_mode={hb.base_mode} "
          f"system_status={hb.system_status}")
    print("[TELEMETRY] Canlı veri basılıyor (Ctrl+C ile çıkın)...\n")

    state = {}
    last_print = 0.0

    while True:
        msg = m.recv_match(blocking=True, timeout=2)
        if msg is None:
            print("[TELEMETRY] ... 2s içinde mesaj gelmedi ...")
            continue

        t = msg.get_type()

        if t == "GLOBAL_POSITION_INT":
            state["lat"] = msg.lat / 1e7
            state["lon"] = msg.lon / 1e7
            state["alt"] = msg.relative_alt / 1000.0
            state["hdg"] = msg.hdg / 100.0
        elif t == "ATTITUDE":
            state["roll"] = msg.roll * 57.2958
            state["pitch"] = msg.pitch * 57.2958
            state["yaw"] = msg.yaw * 57.2958
        elif t == "VFR_HUD":
            state["airspeed"] = msg.airspeed
            state["groundspeed"] = msg.groundspeed
            state["throttle"] = msg.throttle
        elif t == "GPS_RAW_INT":
            state["fix"] = msg.fix_type
            state["sats"] = msg.satellites_visible
        elif t == "SYS_STATUS":
            state["batt_v"] = msg.voltage_battery / 1000.0
            state["batt_pct"] = msg.battery_remaining
        elif t == "MISSION_CURRENT":
            state["wp"] = f"{msg.seq}/{msg.total}"

        now = time.time()
        if now - last_print >= 1.0 and state:
            print(
                f"lat={state.get('lat', '?')} lon={state.get('lon', '?')} alt={state.get('alt', '?')}m "
                f"hdg={state.get('hdg', '?')}° | "
                f"roll={state.get('roll', 0):.1f}° pitch={state.get('pitch', 0):.1f}° yaw={state.get('yaw', 0):.1f}° | "
                f"gs={state.get('groundspeed', '?')}m/s as={state.get('airspeed', '?')}m/s thr={state.get('throttle', '?')}% | "
                f"fix={state.get('fix', '?')} sat={state.get('sats', '?')} | "
                f"batt={state.get('batt_v', '?')}V {state.get('batt_pct', '?')}% | "
                f"WP={state.get('wp', '?')}"
            )
            last_print = now


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=config.FC_PORT)
    parser.add_argument("--baud", type=int, default=config.FC_BAUDRATE)
    args = parser.parse_args()

    try:
        main(args.port, args.baud)
    except KeyboardInterrupt:
        print("\n[TELEMETRY] Ctrl+C — durduruldu")

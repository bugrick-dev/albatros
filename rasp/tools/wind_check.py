#!/usr/bin/env python3
"""
Yerde/bench testi: FC'ye bağlanıp 30s boyunca ArduPilot WIND mesajını dinler
ve mission._wind_from_to_ned ile NED'e çevirir. albatros.service KAPALIYKEN
çalıştırın (seri port tek kullanıcı):

  sudo systemctl stop albatros && sudo /home/albatros/venv/bin/python3 /home/albatros/albatros/rasp/tools/wind_check.py

Beklenen: "WIND #1: ... m/s ...°'DEN" satırları. Yerde EKF rüzgar tahmini
0 civarı olabilir — önemli olan mesajın GELMESİ (telemetry.wind()'in aksine).
"""
import asyncio, json, math, sys, time
sys.path.insert(0, "/home/albatros/albatros/rasp")
import config
from mavsdk import System
from mavsdk.mavlink_direct import MavlinkMessage

DURATION_S = 30

async def main():
    drone = System()
    addr = f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}"
    print(f"bağlanıyor: {addr}")
    await drone.connect(system_address=addr)
    async for st in drone.core.connection_state():
        if st.is_connected:
            print("✓ FC bağlı"); break

    await drone.mavlink_direct.send_message(MavlinkMessage(
        message_name="COMMAND_LONG", system_id=255, component_id=190,
        target_system_id=1, target_component_id=1,
        fields_json=json.dumps({"target_system": 1, "target_component": 1,
            "command": config.CMD_SET_MESSAGE_INTERVAL, "confirmation": 0,
            "param1": float(config.MAVLINK_MSG_ID_WIND),
            "param2": float(int(1_000_000 / config.WIND_STREAM_HZ)),
            "param3": 0.0, "param4": 0.0, "param5": 0.0, "param6": 0.0, "param7": 0.0})))
    print(f"SET_MESSAGE_INTERVAL(WIND, {config.WIND_STREAM_HZ}Hz) gönderildi; {DURATION_S}s dinleniyor...")

    count = 0
    t0 = time.monotonic()
    async def listen():
        nonlocal count
        async for msg in drone.mavlink_direct.message("WIND"):
            f = json.loads(msg.fields_json)
            d, s = float(f["direction"]), float(f["speed"])
            r = math.radians(d)
            count += 1
            print(f"WIND #{count} (+{time.monotonic()-t0:.1f}s): {s:.2f} m/s {d:.0f}°'DEN "
                  f"→ wind_n={-s*math.cos(r):+.2f} wind_e={-s*math.sin(r):+.2f}  raw={f}")
    try:
        await asyncio.wait_for(listen(), timeout=DURATION_S)
    except asyncio.TimeoutError:
        pass
    print(f"\nSONUÇ: {DURATION_S}s içinde {count} WIND mesajı geldi "
          f"({'✓ kaynak çalışıyor' if count else '✗ HİÇ GELMEDİ — SRx_EXTRA3 / dialect kontrol edin'})")

asyncio.run(main())

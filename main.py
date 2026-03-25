import asyncio
from mavsdk import System

async def run():
    drone = System()
    
    print("Cube Orange Plus'a ttyAMA3 üzerinden bağlanılıyor...")
    await drone.connect(system_address="serial:///dev/ttyAMA3:57600")
    
    print("Bağlantı bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone bağlandı!")
            break
    
    print("Veri akışı başlıyor... Çıkmak için Ctrl+C'ye basın.\n")
    
    try:
        async for attitude in drone.telemetry.attitude_euler():
            roll = attitude.roll_deg
            pitch = attitude.pitch_deg
            yaw = attitude.yaw_deg
            print(f"Roll: {roll:7.2f}° | Pitch: {pitch:7.2f}° | Yaw: {yaw:7.2f}°", end="\r")
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\nProgram kullanıcı tarafından sonlandırıldı.")
import asyncio
from mavsdk import System

async def run():
    # Drone objesini oluştur
    drone = System()
    
    # RPi seri portu üzerinden bağlan
    # Not: Port ismi donanımına göre /dev/ttyS0 veya /dev/serial0 olabilir.
    print("Cube Orange Plus'a bağlanılıyor...")
    await drone.connect(system_address="serial:///dev/serial0:921600")

    # Bağlantının kurulmasını bekle
    print("Bağlantı bekleniyor...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone bağlandı!")
            break

    print("Veri akışı başlıyor... Çıkmak için Ctrl+C'ye basın.\n")
        # Roll, Pitch ve Yaw verilerini asenkron olarak çek ve terminale yazdır
    try:
        async for attitude in drone.telemetry.attitude_euler():
            roll = attitude.roll_deg
            pitch = attitude.pitch_deg
            yaw = attitude.yaw_deg
            
            # \r (carriage return) ile satır başına dönüp eski verinin üzerine yazarız
            # :7.2f formatı, değerlerin hizalı durmasını sağlar
            print(f"Roll: {roll:7.2f}° | Pitch: {pitch:7.2f}° | Yaw: {yaw:7.2f}°", end="\r")
            
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    # Programın Ctrl+C ile temiz bir şekilde kapatılmasını sağlarız
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından sonlandırıldı.")
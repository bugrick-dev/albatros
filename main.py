import serial
import time

PORT = "/dev/serial0" # Hata almadığın portu buraya yaz
BAUD_RATE = 921600

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print("Loopback testi başlıyor... RPi TX ve RX pinleri BİRBİRİNE bağlı olmalı.")
    
    # Pi kendi kendine mesaj gönderiyor
    mesaj = b"TEKNOFEST_TEST\n"
    ser.write(mesaj)
    time.sleep(0.1)
    
    # Pi kendi gönderdiğini okuyabiliyor mu?
    if ser.in_waiting > 0:
        gelen_veri = ser.read(ser.in_waiting)
        print(f"\n[SONUÇ]: BAŞARILI! Gelen veri: {gelen_veri}")
        print("Raspberry Pi'nin UART donanımı ve Linux ayarları SAPASAĞLAM çalışıyor.")
    else:
        print("\n[SONUÇ]: BAŞARISIZ!")
        print("Pi kendi gönderdiği mesajı bile duyamadı. Pinler ölü veya config hala bozuk.")

except Exception as e:
    print(f"Hata: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
import serial
import time

# Eğer serial0 çalışmazsa burayı "/dev/ttyS0" veya "/dev/ttyAMA0" yap
PORT = "/dev/serial0" 
BAUD_RATE = 921600

try:
    print(f"{PORT} portu {BAUD_RATE} baud hızında açılıyor...")
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print("Bağlantı başarılı! Veri dinleniyor...\n")
    print("NOT: Ekrana garip/anlamsız karakterler veya hex kodları akarsa sistem ÇALIŞIYOR demektir.")
    print("Çıkmak için Ctrl+C'ye basın.\n" + "-"*50)
    
    while True:
        if ser.in_waiting > 0:
            # Porttaki ham veriyi oku
            raw_data = ser.read(ser.in_waiting)
            # Veriyi ekrana bas
            print(raw_data)
        time.sleep(0.05)

except serial.SerialException as e:
    print(f"\n[HATA] Port açılamadı: {e}")
    print("Muhtemel sebep: 'sudo usermod -a -G dialout $USER' izni verilmemiş veya port adı yanlış.")
except KeyboardInterrupt:
    print("\nTest sonlandırıldı.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
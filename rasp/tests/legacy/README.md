# Legacy testler

Bu dizindeki testler **artık var olmayan** altyapıya (silinen `servo.py`,
`SERVO_*_YUK_PIN` GPIO pinleri, TCP tabanlı rpicam→ffmpeg decode pipeline'ı,
`RPICAM_TCP_PORT`) referans verir ve bugünkü kodla ÇALIŞMAZ (ImportError/
AttributeError). Tarihsel kayıt için tutuluyorlar (2026-08-17).

Not: pigpiod servisi hâlâ `-p 8779` portunda kalmalıdır (8888 çakışması
notu için `test_servo_color_trigger.py` başlığına bakın).

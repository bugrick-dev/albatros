"""
TEKNOFEST 2026 - Sabit Kanat - Görev 2
Tüm sabitler ve konfigürasyon parametreleri.
GCS planı kesinleşince SEARCH_START_WP ve SEARCH_LOOP_EXIT_WP güncellenmelidir.
"""
import json
from pathlib import Path

import numpy as np

# --- Video ---
WIDTH  = 640
HEIGHT = 480
FPS    = 30
BITRATE = 1000000  # WFB FEC yedekliliğiyle (bkz. WFB_FEC_*) uyumlu düşürüldü
INTRA   = 15        # paket kaybı sonrası hızlı toparlanma için sık keyframe

# Kamera fiziksel olarak ters monte (kablo yukarı çıkacak şekilde) — rpicam-vid
# seviyesinde düzeltiliyor (yazılımsal cv2.rotate yerine): tespit algoritması
# (renk/şekil) ve CAMERA_PITCH rotasyondan etkilenmiyor, ham görüntü zaten
# doğru yönde gelir. DEĞİŞTİRİLİRSE kamera kalibrasyonu YENİDEN yapılmalı —
# calibrate script'i o anki yönelimi ölçüyor (2026-08-09).
CAMERA_ROTATION_DEG = 180

# --- Tespit alanı sınırları ---
MIN_AREA = int(0.001 * WIDTH * HEIGHT)
MAX_AREA = int(0.3   * WIDTH * HEIGHT)

# --- HSV renk aralıkları (bireysel kanallar) ---
# Gerçek yarışma malzemesinin (branda vb.) tam tonu henüz bilinmediği için
# geniş aralıkta bırakıldı (2026-07-26) — malzeme netleşince gerçek örnekle
# (HSV tuner widget'ıyla) daraltılacak.
BLUE_H_MIN, BLUE_H_MAX = 100, 130
BLUE_S_MIN, BLUE_S_MAX =  80, 255
BLUE_V_MIN, BLUE_V_MAX =  50, 255

RED1_H_MIN, RED1_H_MAX =   0,  10
RED2_H_MIN, RED2_H_MAX = 170, 180
RED_S_MIN,  RED_S_MAX  =  80, 255
RED_V_MIN,  RED_V_MAX  =  50, 255

# --- HSV eşik dizileri (cv2.inRange'de doğrudan kullanılır) ---
BLUE_HSV_LOWER = np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN])
BLUE_HSV_UPPER = np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX])
RED1_HSV_LOWER = np.array([RED1_H_MIN, RED_S_MIN,  RED_V_MIN])
RED1_HSV_UPPER = np.array([RED1_H_MAX, RED_S_MAX,  RED_V_MAX])
RED2_HSV_LOWER = np.array([RED2_H_MIN, RED_S_MIN,  RED_V_MIN])
RED2_HSV_UPPER = np.array([RED2_H_MAX, RED_S_MAX,  RED_V_MAX])

# --- Kare tespiti ---
SQUARE_CORNER_TOLERANCE = 4
ASPECT_RATIO_MIN = 0.7
ASPECT_RATIO_MAX = 1.3
# Doluluk oranı (kontur alanı / bounding box alanı): kare ~0.9-1.0, altıgen ~0.65-0.87,
# üçgen ~0.5 — yarışma alanındaki altıgen/üçgen şekilleri kesin olarak elemek için
# (4 köşeye yaklaşma + en-boy oranı tek başına bunları ayırt edemiyor, 2026-07-26).
SQUARE_MIN_EXTENT = 0.85

# --- WFB-ng ---
WFB_MAC      = "6c:4c:bc:0a:62:a0"
WFB_KEY_PATH = "/home/albatros/gs.key"
WFB_LINK_ID  = "7669206"
WFB_CHANNEL  = 157   # UNII-3, daha az yoğun — kart 88x2bu.conf'ta rtw_regd_src=1/rtw_country_code=US
                      # gerektiriyor, aksi halde sürücü bu kanalı kilitler (2026-07-26 doğrulandı)
WFB_MCS            = 0      # en dayanıklı modülasyon (BPSK, HT MCS0)
WFB_BANDWIDTH      = 20     # MHz — dar bant, daha uzun menzil
WFB_GUARD_INTERVAL = "long"
WFB_LDPC           = 1
WFB_FEC_K          = 4
WFB_FEC_N          = 12     # 3x FEC yedekliliği — sahada ölçülüp doğrulandı
# T3U Plus (RTL8812BU, 88x2bu) donanım tavanı: 3100 mBm kabul ediliyor,
# 3200 mBm reddediliyor (region/reg-domain bu kartta gerçek gücü sınırlamıyor).
WFB_TXPOWER_MBM    = 3100

# --- Port numaraları ---
RPICAM_TCP_PORT = 8888
FFMPEG_UDP_PORT = 9000
WFB_UDP_PORT    = 5600

# --- Uçuş bilgisayarı ---
# TELEM2 jumper kablolaması havada GND temassızlığına bağlı bağlantı kopmalarına
# yol açtı → Pixhawk'ın kendi USB portu Pi 5'in USB-A soketine doğrudan bağlanıyor.
# Bu, TELEM2/SERIAL2'den bağımsız ayrı bir MAVLink kanalı — FC tarafında parametre
# değişikliği gerekmez. FC güç modülünden beslenmeye devam eder, USB yalnızca veri hattı.
FC_BAUDRATE = 115200  # USB CDC-ACM gerçek baud'u yok sayar, mavsdk bağlantı string'i için gerekli


def _resolve_fc_port():
    """FC'nin USB-seri cihaz yolunu döndürür.

    /dev/ttyACM0 gibi sabit bir isim yerine /dev/serial/by-id/ kullanılır: başka bir
    USB-seri cihaz (GPS, adaptör vb.) daha önce takılırsa ACM numarası kayabilir,
    by-id ise cihazın USB seri numarasına bağlı olduğu için sabit kalır.
    """
    by_id_dir = Path("/dev/serial/by-id")
    if by_id_dir.is_dir():
        candidates = sorted(by_id_dir.iterdir())
        if candidates:
            return str(candidates[0])
    return "/dev/ttyACM0"


FC_PORT = _resolve_fc_port()

# --- Kamera ---
# CAMERA_FOV_H/V: yalnızca camera_calib.json YOKSA (kalibrasyon yapılmamış
# cihaz) kullanılan yedek/varsayılan değerler — Pi Camera V2 spek değerleri,
# bu sistemdeki imx477 (HQ Camera, değiştirilebilir lens) için GERÇEĞİ
# yansıtmaz. Her cihaz rasp/tools/camera_calibrate.py ile kalibre edilip
# kendi camera_calib.json'unu üretmeli (bkz. aşağıdaki yükleme bloğu) —
# ölçülen gerçek FOV bu iki kamerada ~36°/27.5° çıktı, 62.2°/48.8° değil
# (2026-08-07 sahada bulundu, bkz. geo.py pixel_to_gps).
CAMERA_FOV_H = 62.2
CAMERA_FOV_V = 48.8
CAMERA_PITCH = 45.0

# --- Roll/pitch toleransı (bkz. geo.pixel_to_gps) ---
# Bu açıyı aşan roll'da tespit REDDEDİLİR (GPS hesabı yapılmaz, kuyruğa
# eklenmez) — yüksek bank açısında piksel->GPS projeksiyonu ufka yaklaşıp
# aşırı büyük/dejenere mesafeler üretebiliyor (simülasyonla doğrulandı,
# 2026-08-08: 30°+ roll'da köşe pikselleri 100-1000+ m hataya çıkabiliyor).
MAX_ROLL_FOR_DETECTION_DEG = 30.0

# --- Kamera kalibrasyonu (cihaza özel, git'e girmez — bkz. tools/camera_calibrate.py) ---
CAMERA_CALIB_PATH = Path(__file__).parent / "camera_calib.json"
CAMERA_CALIBRATED = False
CAMERA_FX = CAMERA_FY = CAMERA_CX = CAMERA_CY = None
CAMERA_DIST_COEFFS = None
if CAMERA_CALIB_PATH.is_file():
    try:
        _calib = json.loads(CAMERA_CALIB_PATH.read_text())
        CAMERA_FX          = _calib["fx"]
        CAMERA_FY          = _calib["fy"]
        CAMERA_CX          = _calib["cx"]
        CAMERA_CY          = _calib["cy"]
        CAMERA_DIST_COEFFS = _calib["dist_coeffs"]
        CAMERA_CALIBRATED  = True
    except Exception:
        pass  # bozuk/eksik dosya — FOV tabanlı yedek hesaba düş

# --- Servo (RPi GPIO) ---
# Şartname: mavi hedefe kırmızı boyalı yük, kırmızı hedefe mavi boyalı yük
SERVO_KIRMIZI_YUK_PIN = 18   # kırmızı boyalı yük → mavi hedefe
SERVO_MAVI_YUK_PIN    = 23   # mavi boyalı yük    → kırmızı hedefe

# --- Uçuş parametreleri ---
DRONE_SPEED_MS             = 15.0
DROP_TRIGGER_RADIUS_M      = 20
SCAN_EXIT_DELAY_SEC        = 15
SINGLE_TARGET_TIMEOUT_SEC  = 30
FC_CONNECT_TIMEOUT_SEC     = 10   # FC bağlantısı kurulamazsa video-only moda geç
FC_RECONNECT_INTERVAL_SEC  = 3    # Bağlantı koptuğunda (ör. kalibrasyon sonrası FC reboot)
                                   # yeniden deneme aralığı — bkz. mission.fc_connection_task

# --- Tarama / drop hız yönetimi ---
SEARCH_START_WP     = 4      # Bu WP'ye gelince tarama hızına geç (GCS planına göre ayarla)
DETECTION_ACTIVE_WP = 4      # Bu WP'ye gelince tespit aktif olur (GCS planına göre ayarla)
SEARCH_SPEED_MS     = 10.0   # Tarama hızı (m/s)
DROP_SPEED_MS       = 10.0   # Yük bırakmadan önce hız (m/s)
SEARCH_LOOP_EXIT_WP = 7      # DO_JUMP'tan SONRAKİ WP index. Hedefler bulunduktan sonra bu index'ten
                              # itibaren olan öğeler "iniş sekansı" sayılır ve yeni drop misyonuna
                              # olduğu gibi eklenir (bkz. mission.build_and_start_drop_mission)

# --- FC Servo (USE_FC_SERVO=True ise geçerli) ---
USE_FC_SERVO        = True  # True → DO_SET_SERVO (FC çıkışı), False → RPi GPIO
SERVO_KIRMIZI_FC_NO = 9      # FC servo kanalı — kırmızı yük (AUX1)
SERVO_MAVI_FC_NO    = 10     # FC servo kanalı — mavi yük (AUX2)

# Her servo kendi güvenli aralığıyla ayrı tanımlanıyor — sahada doğrulandı
# (test_wp_mark_servo_release ile, 2026-07-15). İkisi de aynı yönde: PWM
# azalınca açılıyor, artınca kapanıyor.
PWM_KIRMIZI_RELEASE = 900    # kırmızı servo (AUX1) — açma
PWM_KIRMIZI_NEUTRAL = 1750   # kırmızı servo (AUX1) — kapalı/nötr
PWM_MAVI_RELEASE    = 900    # mavi servo (AUX2) — açma
PWM_MAVI_NEUTRAL    = 1600   # mavi servo (AUX2) — kapalı/nötr

# --- MAVLink komut kodları (MAV_CMD_*) ---
CMD_NAV_WAYPOINT     = 16
CMD_NAV_LOITER_TURNS = 18
CMD_CONDITION_DIST   = 114
CMD_DO_CHANGE_SPEED  = 178
CMD_DO_SET_SERVO     = 183
CMD_RTL              = 20

# --- Orbit test parametreleri (bkz. tests/test_mission_orbit_swap.py) ---
# Yük bırakma yerine hedef üstünde dönüş test edilirken kullanılır.
ORBIT_RADIUS_M   = 30.0   # hedef üstünde dönüş yarıçapı (m)
ORBIT_TURNS      = 3      # hedef başına tur sayısı
ORBIT_ALTITUDE_M = 60.0   # yaklaşma + loiter irtifası (relative alt, sabit)

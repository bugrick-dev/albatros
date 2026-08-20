"""
Paylaşılan değişken global durum.
Tüm modüller bu dosyadan okur/yazar — dairesel import yoktur.
"""
import threading
from collections import deque
import numpy as np
from queue import Queue

# --- Morfoloji çekirdeği ---
kernel = np.ones((5, 5), np.uint8)

# --- Hedef kuyruğu (OpenCV thread → mission_task) ---
target_queue = Queue()

# --- Telemetri ---
telemetry_lock    = threading.Lock()
# vel_n/vel_e: NED yer hızı bileşenleri — drop_trigger_task'ın gerçek yer izi
# rotasını (course) hesaplaması için (yaw değil: sabit kanat rüzgarda crab
# yapar, yük burnun değil hız vektörünün yönünde savrulur — 2026-08-17).
current_telemetry = {"lat": None, "lon": None, "alt": None, "yaw": None, "roll": None, "pitch": None,
                     "speed": None, "vel_n": None, "vel_e": None}

# --- Zaman damgalı telemetri geçmişi (pixel_to_gps için "en yakın örnek" eşlemesi) ---
# position_task ve attitude_task FARKLI MAVSDK akışları, farklı hızlarda güncelleniyor
# — bu yüzden ikisi ayrı tamponlanıyor, tüketici (vision.py) her birinden bağımsız
# olarak frame anına en yakın örneği seçiyor. time.monotonic() kullanılıyor (duvar
# saati NTP sıçramalarından etkilenmesin diye, bkz. main.py/Pi RTC'siz boot notu).
# maxlen=200: MAVSDK attitude_euler tipik olarak 10-50Hz akıyor, bu en kötü ihtimalle
# ~4 saniyelik geçmiş tutar — bol marj, ihmal edilebilir bellek (2026-08-16).
TELEMETRY_HISTORY_MAXLEN = 200
position_history  = deque(maxlen=TELEMETRY_HISTORY_MAXLEN)  # (ts, lat, lon, alt)
attitude_history   = deque(maxlen=TELEMETRY_HISTORY_MAXLEN)  # (ts, yaw, roll, pitch)
speed_history      = deque(maxlen=TELEMETRY_HISTORY_MAXLEN)  # (ts, speed, vel_north_m_s, vel_east_m_s)

# --- GPS/EKF sağlık durumu (checklist "GPS ve heading" — 2026-08-16) ---
# MAVSDK'nın Health akışı en yakın "EKF durumu" eşdeğeri (ham EKF flag'leri
# değil ama is_global_position_ok/is_local_position_ok fiilen aynı amaca
# hizmet ediyor — çoğu GCS de bunu böyle gösterir).
gps_health = {"num_satellites": None, "fix_type": None}
ekf_health = {"global_position_ok": None, "local_position_ok": None, "home_position_ok": None}


def nearest_telemetry_at(ts):
    """
    Verilen zaman damgasına (time.monotonic()) konum/duruş açısından en yakın
    örnekleri AYRI AYRI bulup birleştirir (position_task ve attitude_task
    farklı hızlarda akıyor, ikisini karıştırmadan en-yakın eşleme yapılmalı).
    Henüz hiç veri gelmemişse None döner — çağıran taraf eski current_telemetry
    davranışına (None kontrolü) geri düşebilir.
    pos_age_s/att_age_s: eşleşen örneğin frame anına MUTLAK zaman uzaklığı
    (saniye) — hem loglama hem de TELEMETRY_MATCH_MAX_AGE_S reddi için
    (bkz. vision.py). abs() şart: eşleşen örnek frame'den SONRA da gelmiş
    olabilir, işaretli fark max() içinde yanlış küçük görünürdü (2026-08-17).
    """
    if not position_history or not attitude_history:
        return None
    pos = min(position_history, key=lambda e: abs(e[0] - ts))
    att = min(attitude_history, key=lambda e: abs(e[0] - ts))
    return {
        "lat": pos[1], "lon": pos[2], "alt": pos[3],
        "yaw": att[1], "roll": att[2], "pitch": att[3],
        "pos_age_s": abs(ts - pos[0]),
        "att_age_s": abs(ts - att[0]),
    }

# --- FC bağlantı durumu (HUD icin) ---
fc_connected = False

# --- FC yeniden bağlanma döngüsü zaten çalışıyor mu (çakışan görev başlatmayı önler) ---
fc_reconnecting = False

# --- Tespit edilen hedefler ---
detected_targets = {"mavi": None, "kirmizi": None}

# --- Kilitlenen (kuyruğa alınan/mission'a kaydedilen) hedeflerin GPS konumu ---
# detected_targets'tan farkı: bu SABİT — hedef kadrajdan çıksa/tracker kaybolsa
# bile burada kalır, HUD'da "kilitlendi" bilgisini kalıcı göstermek için
# (2026-08-20, bkz. mission.py mission_task — release_points.append noktası).
locked_targets = {"mavi": None, "kirmizi": None}  # {color: (lat, lon)} | None

# --- Gönderilen servo (bırakma) komutları — HUD'da "SERVO AÇILDI" bilgisi için ---
# (2026-08-20, bkz. mission.py _trigger_release).
servo_events = {"mavi": None, "kirmizi": None}  # {color: {"channel","lat","lon","ts"}} | None

# --- Tespit etkinleştirme (Direk 2 geçildikten sonra set edilir) ---
detection_active = threading.Event()

# --- Aktif WP takibi ---
# lat/lon: bu WP'YE GEÇİLDİĞİ ANDAKİ uçak konumu (2026-08-20, bkz.
# mission.waypoint_tracking_task) — "şu an neredeyiz" (current_telemetry)
# değil, "bu WP'ye geçerken neredeydik" bilgisi, HUD'da kalıcı gösterilir.
current_wp       = {"index": None, "total": None, "lat": None, "lon": None}

# --- Alt süreç tutamaçları ---
# (ffmpeg_decode_process kaldırıldı — decode aşaması 2026-08-16'da öldü)
wfb_process           = None
rpicam_process        = None
gst_process           = None
ffmpeg_encode_process = None

# --- Kapanis sinyali (thread'lerin duzgun sonlanmasi icin) ---
shutdown_requested = threading.Event()

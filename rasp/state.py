"""
Paylaşılan değişken global durum.
Tüm modüller bu dosyadan okur/yazar — dairesel import yoktur.
"""
import threading
import numpy as np
from queue import Queue

# --- Morfoloji çekirdeği ---
kernel = np.ones((5, 5), np.uint8)

# --- Hedef kuyruğu (OpenCV thread → mission_task) ---
target_queue = Queue()

# --- Telemetri ---
telemetry_lock    = threading.Lock()
current_telemetry = {"lat": None, "lon": None, "alt": None, "yaw": None, "roll": None, "pitch": None, "speed": None}

# --- FC bağlantı durumu (HUD icin) ---
fc_connected = False

# --- FC yeniden bağlanma döngüsü zaten çalışıyor mu (çakışan görev başlatmayı önler) ---
fc_reconnecting = False

# --- Tespit edilen hedefler ---
detected_targets = {"mavi": None, "kirmizi": None}

# --- Tespit etkinleştirme (Direk 2 geçildikten sonra set edilir) ---
detection_active = threading.Event()

# --- Aktif WP takibi ---
current_wp       = {"index": None, "total": None}

# --- Alt süreç tutamaçları ---
wfb_process           = None
rpicam_process        = None
gst_process           = None
ffmpeg_decode_process = None
ffmpeg_encode_process = None

# --- Kapanis sinyali (thread'lerin duzgun sonlanmasi icin) ---
shutdown_requested = threading.Event()

# --- Servo PWM nesneleri ---
servo_kirmizi_pwm = None
servo_mavi_pwm    = None

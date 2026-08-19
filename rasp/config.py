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

# Kamera fiziksel olarak ters monte (kablo yukarı çıkacak şekilde). NOT:
# capture (rpicam-vid) ve kalibrasyon KASITLI olarak native/rotasyonsuz kalır
# (kalibrasyon değişmeden geçerli olsun diye) — bu sabit yalnızca:
#   1) geo.py'de piksel açı ofsetlerinin işaretini telafi etmek (kamera kendi
#      bakış ekseni etrafında 180° dönükse sağ/sol, yukarı/aşağı ters okunur)
#   2) vision.py'de yalnızca YAYINA giden (encode edilen) kareyi çevirmek
#      (yer istasyonundaki insan operatör düz görsün diye — tespit/GPS
#      matematiğini etkilemez)
# için kullanılıyor. Şu an 0 veya 180 destekleniyor.
CAMERA_ROTATION_DEG = 180

# --- Tespit alanı sınırları ---
MIN_AREA = int(0.001 * WIDTH * HEIGHT)
MAX_AREA = int(0.3   * WIDTH * HEIGHT)

# --- HSV renk aralıkları (bireysel kanallar) ---
# Gerçek yarışma malzemesinin (branda vb.) tam tonu henüz bilinmediği için
# geniş aralıkta bırakıldı (2026-07-26) — malzeme netleşince gerçek örnekle
# (HSV tuner widget'ıyla) daraltılacak.
#
# 2026-08-12: sahada MAVİ branda hiç tespit edilmedi — aralık körlemesine
# (elde referans görüntü/HSV örneği olmadan) genişletildi:
#   H  100-130 -> 95-135  (güneş ışığında/parlamada gerçek malzeme camgöbeğine
#                 (cyan) ya da mora doğru kaymış olabilir)
#   S   80-255 -> 60-255  (soluk/tozlu branda ya da güneşte "yıkanmış" görünüm
#                 düşük doygunluk üretebilir)
#   V   50-255 -> 35-255  (gölgede kalan kısımları da yakalasın)
# Bu GEÇİCİ bir düzeltme — hâlâ gözle tahmin. Bir sonraki saha testinde
# gerçek bir kare/fotoğraf alınırsa (ya da bir HSV tuner aracıyla canlı
# ayarlanırsa) bu aralık kesinleştirilmeli; şu an fazla geniş olursa yanlış
# pozitif riski artar (bkz. MAX_DETECTION_DISTANCE_M/roll/pitch reddi zaten
# uzak/şüpheli tespitleri eliyor ama sahte YAKIN tespitlere karşı koruma
# değil).
BLUE_H_MIN, BLUE_H_MAX =  95, 135
BLUE_S_MIN, BLUE_S_MAX =  60, 255
BLUE_V_MIN, BLUE_V_MAX =  35, 255

# 2026-08-17: gerçek uçuş kaydından alınan bir kareden (26.9m ALT, WP 14/19)
# cv2 ile PİXEL-EXACT ölçüldü (artık gözle tahmin değil):
#   MAVİ  iç bölge: H≈105-106, S≈183-187, V≈244-248 — mevcut BLUE_HSV aralığının
#         (95-135/60-255/35-255) TAMAMEN içinde, örneklenen pikselin %100'ü
#         maskeden geçiyor. Yani mavi için HSV hiç sorun değilmiş; MAVİ: --
#         sorunu is_square/ekstent tarafındandı (bkz. yukarıdaki ASPECT_RATIO/
#         SQUARE_MIN_EXTENT notu, 45° kamera açısı → trapezoid görünüm).
#   KIRMIZI/PEMBE panel: H≈154-163 (medyan 162), S≈51-124 (medyan 116, p5=51),
#         V≈105-120 — klasik kırmızı değil, gerçekten pembe/magenta'ya kaymış.
#         Eski RED2_H_MIN=170 bunu TAMAMEN dışarıda bırakıyordu (RED1 0-10 ve
#         RED2 170-180 ikisi de kapsamıyordu). RED2_H_MIN önce 150'ye çekildi;
#         ölçümle test edilince o aralık pikselin %84'ünü yakalıyordu — S_MIN=80
#         alt kuyruğu (S=51-79) fazla kesiyordu, 65'e indirildi (marj için).
RED1_H_MIN, RED1_H_MAX =   0,  10
RED2_H_MIN, RED2_H_MAX = 150, 180
RED_S_MIN,  RED_S_MAX  =  65, 255
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
# 2026-08-17: kamera CAMERA_PITCH=45° eğik monteli — yerdeki kare hedef nadir
# değil, perspektifle "keystone" trapezoide benzer şekilde görünüyor (yakın kenar
# geniş, uzak kenar dar). Bu, minAreaRect en-boy oranını 1'den saha testlerinde
# gözlemlendiği kadar uzaklaştırabiliyor — eski 0.7-1.3 aralığı sahada gerçek
# hedefi reddediyordu, bu yüzden gevşetildi.
ASPECT_RATIO_MIN = 0.55
ASPECT_RATIO_MAX = 1.7
# Doluluk oranı (kontur alanı / bounding box alanı): kare (nadir'den) ~0.9-1.0,
# altıgen ~0.65-0.87, üçgen ~0.5 — yarışma alanındaki altıgen/üçgen şekilleri
# kesin olarak elemek için (4 köşeye yaklaşma + en-boy oranı tek başına bunları
# ayırt edemiyor, 2026-07-26). 2026-08-17: 45° eğik kameradan görülen bir kare de
# trapezoide benzediği için doluluğu düşüyor (simetrik trapezoid modeli: extent ≈
# (1+t)/2, t=uzak/yakın kenar oranı) — 0.85 gerçek hedefi de eleyebiliyordu, 0.78'e
# indirildi. NOT: bu, altıgen aralığının (0.65-0.87) üst ucuyla örtüşüyor — hâlâ
# yanlış pozitif riski var, sahada _mask_debug_info loglarındaki gerçek
# oran/doluluk değerleriyle daha da kalibre edilmeli.
SQUARE_MIN_EXTENT = 0.78

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
# 3100'de yük altında ~2-3 dk'da bir USB disconnect (rtl88x2bu Runtime PM
# underflow) gözlendi; rtw_smart_ps=0 (bkz. /etc/modprobe.d/88x2bu.conf) ile
# birlikte 3000'de 23+ dk kesintisiz test edildi (2026-08-16). Menzil kaybı
# ~yok denecek kadar az (31→30 dBm, ~1dB).
WFB_TXPOWER_MBM    = 3000

# --- Port numaraları ---
# (RPICAM_TCP_PORT/FFMPEG_UDP_PORT kaldırıldı — TCP+decode pipeline'ı ile
# birlikte öldüler, bkz. pipeline.py raw stdout notu. NOT: pigpiod hâlâ
# 8779 portunda kalmalı, 8888'e geri alınmamalı — bkz. tests/legacy/.)
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

# Roll ile AYNI sebep, pitch için: burun kalkarsa (nose-up) ışının ufka göre
# çökme açısı daha da azalır, aynı sığ-açı büyütmesi oluşur — roll'da
# eskiden gözlendiği gibi (bkz. yukarısı) pitch de sınırsız bırakılırsa aynı
# hataya düşer, o yüzden roll'la SİMETRİK sınırlandı.
MAX_PITCH_FOR_DETECTION_DEG = 30.0

# Hesaplanan yatay mesafe (drone -> hedef) bu değeri aşarsa tespit
# REDDEDİLİR — 2026-08-12 masa/Gazebo testlerinde bulundu: CAMERA_PITCH=45°
# + CAMERA_FOV_V=48.8° ile frame'in ÜST kenarına yakın bir tespit, roll/pitch
# SIFIR olsa BİLE ~60m irtifada ~160m'ye projekte oluyor (60/tan(45-24.4°)),
# ve bu bölgede birkaç derecelik attitude gürültüsü mesafeyi/yönü onlarca-
# yüzlerce metre oynatabiliyor (dt/dθ, θ küçüldükçe patlıyor). ORBIT_RADIUS_M
# ve arama koridoru göz önüne alınarak seçildi — Gazebo'da farklı irtifa/FOV
# kombinasyonlarıyla yeniden doğrulanmalı, gerekirse ayarlanmalı.
MAX_DETECTION_DISTANCE_M = 200.0

# --- Hedef izleme: en-yakın-nokta kilidi (bkz. vision._update_detection) ---
# 45° açılı montaj KORUNUYOR (erken tespit/orbit planlama için gerekli,
# 2026-08-12 tartışması) — ama artık GPS kilidi hedefin İLK görüldüğü
# (genelde kenarda/uzak, en hatalı) karede değil, uçak yaklaşırken görülen
# EN İYİ (yer mesafesi en küçük) karede yapılıyor. Bu üç sabit o izleme
# döngüsünü kontrol eder:
#
#   DETECTION_CONFIRM_STREAK: en iyi örnekten sonra art arda bu kadar kare
#     "daha kötü" (mesafe artıyor) gelirse en yakın noktayı geçtik demektir
#     — o ana kadarki en iyi örnek kilitlenir.
#   DETECTION_LOST_STREAK: hedef art arda bu kadar karede HİÇ tespit
#     edilemezse (kadraj dışına çıktı/kalıcı kayıp) kilitlenir — tek bir
#     karenin parlama/motion-blur yüzünden kaçırılması yüzünden yanlışlıkla
#     "kayıp" sayılmasın diye (tam bir görsel tracker yerine ucuz tolerans).
#   DETECTION_TRACK_MAX_SEC: yukarıdakilerin hiçbiri tetiklenmezse (ör. hedef
#     tüm geçiş boyunca kadrajda kalıp hep "iyileşiyor" görünürse) süre
#     dolunca yine de elimizdeki en iyi örnekle kilitlenir — sonsuz beklemeyi
#     önler.
DETECTION_CONFIRM_STREAK = 8
DETECTION_LOST_STREAK    = 5
DETECTION_TRACK_MAX_SEC  = 4.0

# --- Ufak görsel tracker (MOSSE) ile kare köprüleme (bkz. vision._update_detection) ---
# Yukarıdaki DETECTION_LOST_STREAK "tam bir görsel tracker yerine ucuz
# tolerans" olarak yazılmıştı (2026-08-12) — geçiş süresi zaten çok kısa
# olduğundan (birkaç saniye) hafif bir MOSSE korelasyon tracker'ı eklemek
# ihmal edilebilir CPU maliyetiyle (~0.05ms/kare, 2026-08-19 ölçüldü)
# sağlamlığı artırıyor: HSV/kontur tespiti TEK bir karede (motion blur,
# H264 blok gürültüsü, anlık parlama) kaçırılırsa tracker son bilinen
# konumu tahmin edip miss_streak'i erken tetiklemeden köprü kurar. GPS
# KİLİDİ YİNE SADECE gerçek renk-doğrulanmış karelerden alınır — tracker
# tahmini asla pixel_to_gps'e beslenmez, yalnızca HUD sürekliliği ve
# miss_streak köprülemesi için kullanılır (sürüklenme riski GPS doğruluğunu
# etkilemesin diye). TRACKER_MAX_BRIDGE_FRAMES aşılırsa (hedef gerçekten
# kadraj dışına çıktı/MOSSE sürüklendi) normal DETECTION_LOST_STREAK
# mantığına düşülür.
TRACKER_MAX_BRIDGE_FRAMES = 4

# --- Kamera kalibrasyonu (cihaza özel, git'e girmez — bkz. tools/camera_calibrate.py) ---
CAMERA_CALIB_PATH = Path(__file__).parent / "camera_calib.json"
CAMERA_CALIBRATED = False
CAMERA_CALIB_ERROR = None   # dosya var ama okunamadıysa hata metni (main.py loglar)
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
    except Exception as e:
        # Bozuk/eksik dosya — FOV tabanlı yedek hesaba düşülür. Bu SESSİZCE
        # olmamalı: yanlış FOV ile piksel açıları ~1.7x hatalı çıkıp GPS
        # işaretlerini onlarca-yüzlerce metre saptırabilir (2026-08-17,
        # "belirsiz yönlerde sapma" analizi) — main.py açılışta bunu loglar.
        CAMERA_CALIB_ERROR = repr(e)

# --- Uçuş parametreleri ---
DRONE_SPEED_MS             = 15.0
# DROP_TRIGGER_RADIUS_M artık "tetik" değil "KURMA (arming)" yarıçapı: bu
# mesafeye girilince canlı drop değerlendirmesi başlar. Fiili bırakma kararı
# along/cross-track ayrışımıyla verilir (bkz. mission.drop_trigger_task,
# 2026-08-17): eskiden 40m'ye girildiği AN bırakılıyordu — bu, yükün release
# noktasından 40m öteye (şartnamedeki 20m ölçüm sınırının 2 katı!) atılmasına
# ve rota başka bacaktayken (ör. öteki hedefe giderken) yanlış tetiklemeye
# izin veriyordu.
DROP_TRIGGER_RADIUS_M      = 40
# Bırakma anı: release noktasına along-track kalan süre bu değerin altına
# inince tetiklenir (pozisyon akışı ~2-5Hz → tik başına ~0.2-0.5s; bir tik
# sonrasında noktayı geçmiş olmamak için akış periyodundan biraz büyük seçildi).
DROP_ALONG_TRIGGER_S       = 0.35
# Release noktasına dik (cross-track) sapma bu değerden büyükse BIRAKMA —
# şartname isabet ölçümünü hedef merkezinden 20m ile sınırlıyor (dışı 0 puan);
# 15m cross-track + balistik/rüzgar hatası ~20m bütçesinin içinde kalma çabası.
DROP_MAX_CROSS_TRACK_M     = 15.0
SCAN_EXIT_DELAY_SEC        = 15
SINGLE_TARGET_TIMEOUT_SEC  = 30
# Tespit aktif olduktan (DETECTION_ACTIVE_WP) sonra HİÇ hedef bulunamazsa
# arama döngüsü DO_JUMP nedeniyle sonsuza dek dönerdi — şartnamede Görev 2
# uçuş süresi azami 10 dakika, aşımı görevi geçersiz kılıyor (2026-08-17).
# Bu süre dolunca eldeki hedeflerle (0 dahil) iniş sekansına geçilir.
SEARCH_TOTAL_TIMEOUT_SEC   = 300
# nearest_telemetry_at eşleşmesi bu yaştan eskiyse tespit örneği REDDEDİLİR
# (eskiden yalnızca log uyarısı vardı, hesap yine de yapılıyordu) — bayat
# attitude/pozisyon ile projeksiyon "belirsiz yönlerde" büyük sapmaların ana
# kaynaklarından (2026-08-17, 277m sapma analizi).
TELEMETRY_MATCH_MAX_AGE_S  = 0.25
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

# --- FC Servo (tek yol — GPIO servo kaldırıldı, 2026-08-16) ---
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
CMD_DO_JUMP          = 177
CMD_DO_CHANGE_SPEED  = 178
CMD_DO_SET_SERVO     = 183
CMD_RTL              = 20

# --- Orbit test parametreleri (bkz. tests/test_mission_orbit_swap.py) ---
# Yük bırakma yerine hedef üstünde dönüş test edilirken kullanılır.
ORBIT_RADIUS_M   = 30.0   # hedef üstünde dönüş yarıçapı (m)
ORBIT_TURNS      = 3      # hedef başına tur sayısı
ORBIT_ALTITUDE_M = 60.0   # yaklaşma + loiter irtifası (relative alt, sabit)

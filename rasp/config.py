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
# 2026-08-23: 1000000 (1Mbps) canlı teşhiste hâlâ sık yerel-kuyruk paket
# kaybına yol açıyordu (wfb_tx.log "packets dropped", ~4-7 paket/s, OpenCV/
# encode tarafı temizken bile) — bufsize daraltma denendi, İYİLEŞTİRMEDİ
# (bkz. vision.py _start_encode_chain notu). Sıradaki A/B adımı: bitrate'in
# kendisini düşürüp gerçek payı (2.1Mbps kullanılabilir bant altında) artırmak.
BITRATE = 800000  # 2026-08-24: 600k'dan kademeli A/B artışı (1. adım) — kalite
                  # artırımı denemesi, wfb_tx.log'da yeni kayıp olup olmadığı
                  # izlenerek yapılıyor. Sorun çıkarsa 600000'e geri alınabilir.
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
BLUE_H_MIN, BLUE_H_MAX =  95, 120
BLUE_S_MIN, BLUE_S_MAX =  90, 255
BLUE_V_MIN, BLUE_V_MAX =  70, 255

# 2026-08-17: gerçek uçuş kaydından alınan bir kareden (26.9m ALT, WP 14/19)
# cv2 ile PİXEL-EXACT ölçüldü (artık gözle tahmin değil):
#   MAVİ  iç bölge: H≈105-106, S≈183-187, V≈244-248 — mevcut BLUE_HSV aralığının
#         (95-135/60-255/35-255) TAMAMEN içinde, örneklenen pikselin %100'ü
#         maskeden geçiyor. Yani mavi için HSV hiç sorun değilmiş; MAVİ: --
#         sorunu is_square/ekstent tarafındandı (bkz. yukarıdaki ASPECT_RATIO/
#         SQUARE_MIN_EXTENT notu, 45° kamera açısı → trapezoid görünüm).
# 2026-08-20: ikinci saha ölçümü (67 nokta, aynı panel) — H 106-107, S 184-190,
#         V 244-248 çıktı; 17 Ağustos ölçümüyle (H 105-106/S 183-187/V 244-248)
#         neredeyse birebir örtüşüyor → gerçek renk çok tutarlı, sahte-pozitif
#         riskini azaltmak için aralık HAFİFÇE daraltıldı (fazla geniş uçlar
#         kırpıldı, ölçülen kümenin etrafında hâlâ büyük pay var):
#           H  95-135 -> 95-120  (üstteki mor/magenta ucu kırpıldı; alt uç
#              aynı bırakıldı, gölgede/donuk ışıkta H biraz düşebilir)
#           S  60-255 -> 90-255  (ölçülen min 183-184'ün ~90-100 altı, toz/
#              gölge/sıkıştırma payı bol)
#           V  35-255 -> 70-255  (ölçülen min 244'ün ~170 altı, gölge payı bol)
#         Bilhassa BÜYÜK daraltma yapılmadı (kaçırma riskine karşı temkinli).
#   KIRMIZI/PEMBE panel: H≈154-163 (medyan 162), S≈51-124 (medyan 116, p5=51),
#         V≈105-120 — klasik kırmızı değil, gerçekten pembe/magenta'ya kaymış.
#         Eski RED2_H_MIN=170 bunu TAMAMEN dışarıda bırakıyordu (RED1 0-10 ve
#         RED2 170-180 ikisi de kapsamıyordu). RED2_H_MIN önce 150'ye çekildi;
#         ölçümle test edilince o aralık pikselin %84'ünü yakalıyordu — S_MIN=80
#         alt kuyruğu (S=51-79) fazla kesiyordu, 65'e indirildi (marj için).
# 2026-08-20: ikinci saha ölçümü (37 nokta, aynı panel) — H 162-164, S 90-133,
#         V 209-229 çıktı. H ve V zaten mevcut aralığın (150-180 / 50-255)
#         rahatça içinde, değişiklik gerekmedi. Ama S tarafında 17 Ağustos'un
#         p5=51 alt kuyruğu hâlâ RED_S_MIN=65'in DIŞINDA kalıyordu (65-51=14
#         puanlık kapanmamış boşluk) — bugünün ölçümü de S'nin ışığa göre
#         geniş salındığını doğruladığından (51-133 arası, iki uçuş birlikte)
#         RED_S_MIN 65 -> 45'e çekilerek bu boşluk kapatıldı ve p5'in altına
#         da biraz pay bırakıldı.
RED1_H_MIN, RED1_H_MAX =   0,  10
RED2_H_MIN, RED2_H_MAX = 150, 180
RED_S_MIN,  RED_S_MAX  =  45, 255
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
# 2026-08-21: approxPolyDP epsilon'u (0.02*çevre) çevreyle ORANTILI — yani
# teoride ölçekten bağımsız. Ama HSV maske sınırındaki piksel-kaynaklı gürültü
# (±1-2px, morfolojiden sonra bile) ölçekten BAĞIMSIZ SABİT bir miktar; küçük/
# uzak hedeflerde bu sabit gürültü epsilon'a göre orantısal büyüyor ve gerçek
# kareyi fazla köşeli/düzensiz gösteriyor. Saha kanıtı (2026-08-20 23:01:26,
# gerçek uçuş, tespit AKTİF): alan=311 (MIN_AREA sınırında), oran=1.65 (kare
# aralığında) ama köşe=9 (4 değil) VE doluluk=0.65 (SQUARE_MIN_EXTENT=0.78
# altında) olduğu için reddedildi — muhtemelen aynı anda kaybolan gerçek hedef,
# çevresi küçük olduğundan epsilon (~1.4px) piksel gürültüsünü yutamadı. Bu
# taban, epsilon'u küçük konturlarda yükseltip gürültü kaynaklı sahte köşeleri
# birleştirir (büyük/uzak konturları etkilemez, zaten epsilon'ları bunun
# üstünde). NOT: aynı taban küçük altıgen/üçgen tuzak şekillerinin de köşe
# sayısını düşürebilir — ASPECT_RATIO/SQUARE_MIN_EXTENT hâlâ birincil ayırt
# edici, sahada DEBUG loglarıyla doğrulanmalı.
SQUARE_APPROX_MIN_EPSILON_PX = 3.0
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
# Birincil anten. 2026-08-24: yedek anten (aynı marka/model, TP-Link 802.11ac
# NIC) eklendi — WFB_MAC_CANDIDATES sırayla denenir, hangisi TAKILIYSA o
# kullanılır (pipeline._find_iface_by_any_mac), tekli anten arızasında/
# değişiminde kod değişikliği gerekmez.
WFB_MAC          = "6c:4c:bc:0a:62:a0"
WFB_MAC_BACKUP   = "58:04:4f:6c:8e:ee"
WFB_MAC_CANDIDATES = [WFB_MAC, WFB_MAC_BACKUP]
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
#
# 2026-08-23: 3100 geçici test edildi (bitrate 1Mbps→600Kbps sonrası A/B) —
# iki ayrı ~9-10dk'lık run'da USB disconnect HİÇ görülmedi (eski sorun bir
# daha tetiklenmedi), ama wfb_tx yerel-kuyruk paket kaybı run'lar arası çok
# değişken çıktı (0 paket/s ile ~2-5 paket/s arası, statik sahnede bile) —
# net bir kazanç kanıtlanamadı, üstelik run2 3000'deki en kötü sonuçtan daha
# kötüydü. Kanıtlanmış 23+ dk'lık 3000 kaydına karşı riske değecek bir fayda
# yok → 3000'e geri alındı.
WFB_TXPOWER_MBM    = 3000

# --- Donanım heartbeat LED'i (2026-08-23, bkz. heartbeat.py) ---
# Fiziksel 40-pinli header: pin 40 = GPIO21 (BCM), pin 39 = GND (hemen
# yanında) — LED anot (+ uygun sınırlayıcı direnç) GPIO21'e, katot pin 39
# GND'ye. Servo/kamera bu pini kullanmıyor (servo FC üzerinden/AUX kanalları,
# kamera CSI hattı — GPIO'ya hiç dokunmuyorlar), çakışma riski yok.
HEARTBEAT_LED_GPIO = 21
# vision döngüsü bu süreden daha uzun süredir tık atmadıysa (donmuş sayılır)
# LED söner — DETECTION_LOST_STREAK gibi kısa görsel kesintilerle (kare
# kaybı, motion blur) karışmasın diye kare süresinin (33ms) çok üzerinde,
# ama gerçek bir donmayı da makul sürede yakalayacak bir eşik seçildi.
HEARTBEAT_STALE_SEC = 3.0

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
# NOT (2026-08-26): 2026-08-08'de 10°/15°'ye sıkılaştırılmıştı (bkz. üstteki
# not) — sahada bu eşik çok dar kaldığı için (arama bacağında normal
# dönüşlerde bile aşılıp geçerli tespitler reddediliyordu) kullanıcı isteğiyle
# 25°'ye gevşetildi. 25° hâlâ 30°+'daki gözlenen büyük hatanın (100-1000+ m)
# altında ama önceki 10°/15°'den daha toleranslı — arada bir yerde sahada
# doğrulama gerekebilir.
MAX_ROLL_FOR_DETECTION_DEG = 25.0

# Roll ile AYNI sebep, pitch için: burun kalkarsa (nose-up) ışının ufka göre
# çökme açısı daha da azalır, aynı sığ-açı büyütmesi oluşur — roll'da
# eskiden gözlendiği gibi (bkz. yukarısı) pitch de sınırsız bırakılırsa aynı
# hataya düşer, o yüzden roll'la SİMETRİK sınırlandı (2026-08-26: ikisi de 25°).
MAX_PITCH_FOR_DETECTION_DEG = 25.0

# Hesaplanan yatay mesafe (drone -> hedef) bu değeri aşarsa tespit
# REDDEDİLİR — 2026-08-12 masa/Gazebo testlerinde bulundu: CAMERA_PITCH=45°
# + CAMERA_FOV_V=48.8° ile frame'in ÜST kenarına yakın bir tespit, roll/pitch
# SIFIR olsa BİLE ~60m irtifada ~160m'ye projekte oluyor (60/tan(45-24.4°)),
# ve bu bölgede birkaç derecelik attitude gürültüsü mesafeyi/yönü onlarca-
# yüzlerce metre oynatabiliyor (dt/dθ, θ küçüldükçe patlıyor). ORBIT_RADIUS_M
# ve arama koridoru göz önüne alınarak seçildi — Gazebo'da farklı irtifa/FOV
# kombinasyonlarıyla yeniden doğrulanmalı, gerekirse ayarlanmalı.
MAX_DETECTION_DISTANCE_M = 200.0

# --- Tespit alanı jeofansı (bkz. geo.point_in_polygon) ---
# Hesaplanan hedef GPS'i bu poligonun (4 köşe, sırayla) DIŞINDA kalırsa
# tespit REDDEDİLİR — 2026-08-24: yarışma alanı dışında (örn. seyirci/pist
# tarafı) yanlışlıkla "hedef" sanılan bir şeklin GPS kilidine girmesini
# engellemek için eklendi. Köşeler (lat, lon) sırayla:
GEOFENCE_POLYGON = [
    (41.2439218, 29.0151072),
    (41.2429250, 29.0153164),
    (41.2428645, 29.0162337),
    (41.2439132, 29.0162498),
]

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

# MOSSE'nin PSR güven eşiği küçük yamalarda (gerçek fotoğrafla ölçüldü,
# 2026-08-19) güvenilir GEÇMİYOR: ~40px altında hemen hiç, ~40-60px arası
# tutarsız, ≥80px'de tutarlı çalışıyor. Uçuş boyunca hedefin GÖRÜNÜR
# boyutu irtifaya göre büyük ölçüde değişiyor (yüksek irtifa/uzak tespit
# küçük, orbit/yaklaşma büyük) — bu yüzden tracker'a ham tespit bbox'ı
# DEĞİL, hedefin merkezini koruyarak en az bu boyuta PADDING'lenmiş bir
# yama veriliyor (bkz. vision._pad_bbox_to_min_size). Böylece köprüleme
# yalnızca yakın/büyük tespitlerle sınırlı kalmıyor, irtifa değişimlerine
# dayanıklı oluyor — fazladan verilen çevresel doku MOSSE'nin kilitlenmesi
# için gerekli, hedefin kendisi yamanın küçük bir kısmı olsa bile işe
# yarıyor (18-150px arası gerçek fotoğrafla doğrulandı).
TRACKER_MIN_PATCH_PX = 80

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
# DROP_TRIGGER_RADIUS_M artık "tetik" değil "KURMA (arming)" yarıçapı: bu
# mesafeye girilince canlı drop değerlendirmesi başlar. Fiili bırakma kararı
# along/cross-track ayrışımıyla verilir (bkz. mission.drop_trigger_task,
# 2026-08-17): eskiden 40m'ye girildiği AN bırakılıyordu — bu, yükün release
# noktasından 40m öteye (şartnamedeki 20m ölçüm sınırının 2 katı!) atılmasına
# ve rota başka bacaktayken (ör. öteki hedefe giderken) yanlış tetiklemeye
# izin veriyordu.
DROP_TRIGGER_RADIUS_M      = 20
# Bırakma anı: release noktasına along-track kalan süre bu değerin altına
# inince tetiklenir (pozisyon akışı ~2-5Hz → tik başına ~0.2-0.5s; bir tik
# sonrasında noktayı geçmiş olmamak için akış periyodundan biraz büyük seçildi).
DROP_ALONG_TRIGGER_S       = 0.25
# Release noktasına dik (cross-track) sapma bu değerden büyükse BIRAKMA —
# şartname isabet ölçümünü hedef merkezinden 20m ile sınırlıyor (dışı 0 puan);
# 15m cross-track + balistik/rüzgar hatası ~20m bütçesinin içinde kalma çabası.
DROP_MAX_CROSS_TRACK_M     = 15.0
# Menzil telafisi (2026-09-01): geo.calculate_drop_point() saf balistik
# nokta-kütle modeli kullanıyor (hava sürtünmesi YOK, ani/gecikmesiz bırakma
# varsayılıyor) — gerçekte yük düşerken sürtünmeyle yavaşlıyor ve DO_SET_SERVO
# komutu ile mekanizmanın fiilen açılması arasında da küçük bir gecikme var;
# ikisi de aynı yönde hata yapıyor (gerçek ileri savrulma, modelin öngördüğünden
# AZ) — saha gözlemi: yükler TUTARLI şekilde hedefin ~5m gerisine düşüyor.
# Bu sabit, calculate_drop_point()'in hesapladığı geri-ofseti (horizontal_dist)
# bu kadar metre azaltıp release noktasını hedefe yaklaştırarak telafi eder.
# Yeni saha verisiyle güncellenmeli (işaret DAİMA gözlemlenen kısa mesafe kadar
# pozitif — uzun düşme gözlenirse negatif de verilebilir).
#
# GÜNCELLEME (2026-09-02): bu artık yalnızca RÜZGARSIZ/hafif rüzgar TABAN
# değeri — DROP_WIND_ALONG_GAIN ile birlikte kullanılıyor (bkz. aşağı ve
# geo.calculate_drop_point). Saha gözlemi: ~17km/h rüzgarda bu 5m'lik sabit
# TEK BAŞINA yetmiyor, yükler hâlâ ~6-7m geriye (hedefe ulaşamadan)
# düşüyordu — sabit değer rüzgara göre ölçeklenmiyordu.
DROP_RANGE_BIAS_M          = 5.0
# Rüzgara bağlı EK menzil telafisi (2026-09-02, bkz. geo.calculate_drop_point):
# ters rüzgar (headwind) bileşeni ne kadar güçlüyse, itkisiz/kaldırmasız
# düşen yük yer hızına göre o kadar erken/fazla yavaşlar ve ham "hız×düşüş
# süresi" tahmininin GERİSİNDE kalır — DROP_RANGE_BIAS_M'nin sabit tabanına
# `(-wind_along_m_s) * fall_time_s * DROP_WIND_ALONG_GAIN` eklenir (rota
# yönündeki arka rüzgarda azalır/negatif olabilir).
#
# Kalibrasyon (2026-09-02, TEK ölçüm noktası): ~17km/h (~4.7m/s) rüzgarda,
# 5m taban telafiden SONRA hâlâ tutarlı ~6.5m (6-7m ort.) kısa düşme
# gözlendi. DROP_TARGET_ALT_M=25m için fall_time≈2.26s → gerekli ek
# reduction 6.5m'yi vermesi için: 0.6 ≈ 6.5 / (4.7 * 2.26). Rüzgarın o an
# tam ters yönden estiği (saf headwind) VARSAYILDI — gerçek yön bileşeni
# ölçülmedi.
#
# NOT: aynı uçuşta yanal (cross) sapma da AYRICA gözlendi (~3-5m, bkz.
# DROP_WIND_CROSS_GAIN) — yani rüzgar aslında saf headwind DEĞİLDİ, bir
# kısmı yanaldı. Bu yüzden "tüm rüzgar hızı along'a ait" varsayımı bu
# kazancı GERÇEKTE OLMASI GEREKENDEN BÜYÜK göstermiş olabilir (aynı rüzgar
# vektörünün hem along hem cross'a birden tam büyüklükte katkı vermesi
# fiziksel olarak mümkün değil — component'lerin karesel toplamı toplam
# rüzgar hızını aşamaz). Artık drop_trigger_task logu rüzgar_n/rüzgar_e'yi
# ayrı ayrı bastığı için (2026-09-02) bir sonraki uçuşta gerçek along/cross
# bileşenleri log'dan okunup İKİ kazanç birden doğru ayrıştırılabilir
# (birden fazla nokta → lineer regresyon idealdir).
DROP_WIND_ALONG_GAIN       = 0.6
# Yanal (cross) rüzgar telafi kazancı (2026-09-02, bkz. geo.calculate_drop_point):
# aynı mantık along-track yerine rotaya DİK rüzgar bileşenine uygulanır —
# düşen yükü rotanın yanına sürükleyen rüzgarı telafi etmek için release
# noktasını ters yöne kaydırır ("sol geri düşme" şikayetinin yanal kısmı).
#
# Kalibrasyon (2026-09-02, aynı uçuş/aynı TEK ölçüm noktası): aynı ~17km/h
# (~4.7m/s) rüzgarda yanal sapma ~3-5m (ort. 4m) ölçüldü → 0.4 ≈
# 4.0 / (4.7 * 2.26). DROP_WIND_ALONG_GAIN'deki notla aynı çekince
# geçerli: yön bileşenleri ayrı ölçülmedi, "tüm rüzgar hızı cross'a ait"
# varsayıldı — gerçek değer muhtemelen biraz daha yüksek (çünkü aynı
# rüzgarın bir kısmı along'a da gitti). Yeni uçuşta rüzgar_n/rüzgar_e
# log'undan kesin ayrıştırma yapılıp ikisi birlikte düzeltilmeli.
DROP_WIND_CROSS_GAIN       = 0.4
# Hedef WP'lerinin (mavi+kırmızı yönlendirme öğeleri) SABİT irtifası —
# relative, MAV_FRAME_GLOBAL_RELATIVE_ALT (2026-08-27). Tarama irtifasından
# (daha yüksek) bu WP'ye doğru gerçek bir ALÇALMA/yaklaşma emri oluşturur —
# eskiden (2026-08-20…26) hedef WP'ler SEARCH_START_WP'nin planlı irtifasını
# devralıyordu, irtifa hiç değişmediği için yatay yönlendirmeden öte bir
# "yaklaşma" yoktu. rp["alt"] (tespit anındaki anlık/gürültülü telemetri)
# KULLANILMIYOR — sabit değer FC'ye ani tırmanış/dalış riskini taşımaz
# (bkz. mission._build_drop_items docstring, 2026-08-09 dalış olayı notu).
DROP_TARGET_ALT_M          = 25.0
# Hedef WP'lerinin MAVLink acceptance radius'u (NAV_WAYPOINT param2, bkz.
# mission._build_drop_items). NOT: bu değer uçağın İZLEDİĞİ ROTAYI değiştirmez
# (rota ArduPilot L1 kontrolcüsü + fiziksel dönüş yarıçapına/bank limitine
# göre belirlenir) — yalnızca WP'nin NE ZAMAN "ulaşıldı" sayılıp bir sonraki
# öğeye (hedef bloğunda bu, DO_JUMP retry'dir — bkz. _make_drop_retry_jump_item)
# geçileceğini kontrol eder. Eskiden 15m idi: uçak hedeften 15m gibi geniş
# bir mesafeden geçse bile WP "tamam" sayılıp DO_JUMP retry hiç tetiklenmeden
# atlanıyordu. Küçültülünce kötü/uzak geçişler daha geç "tamam" sayılır,
# DO_JUMP retry bloğu daha güvenilir devreye girer (2026-08-31 — saha
# gözlemi: uçak bazen hedefin tam üstünden geçmiyor).
DROP_WP_ACCEPT_RADIUS_M    = 10.0
SCAN_EXIT_DELAY_SEC        = 15
# SINGLE_TARGET_TIMEOUT_SEC KALDIRILDI (2026-08-27) — süre bazlı timeout
# testte işe yaramadı. Artık tek hedef bulunduğunda ikinci hedef için WP
# index'i SEARCH_LOOP_EXIT_WP'yi (aşağıda) GEÇENE kadar beklenir, süreye
# bakılmaz (bkz. mission.mission_task).
#
# Tespit aktif olduktan (DETECTION_ACTIVE_WP) sonra HİÇ hedef bulunamazsa
# mission_task (kod tarafı) hedef beklemeye devam eder — şartnamede Görev 2
# uçuş süresi azami 10 dakika, aşımı görevi geçersiz kılıyor (2026-08-17).
# Bu süre dolunca eldeki hedeflerle (0 dahil) build_and_start_drop_mission
# çağrılıp iniş sekansına geçilir.
#
# NOT (2026-08-19): eskiden bu satırın gerekçesi "arama döngüsü DO_JUMP
# nedeniyle sonsuza dek dönerdi" idi — GCS planı artık DO_JUMP KULLANMIYOR,
# tarama bacağı SEARCH_START_WP→SEARCH_LOOP_EXIT_WP arası düz/sonlu bir WP dizisi (bkz.
# SEARCH_LOOP_EXIT_WP notu). FC kendi başına bırakılsa bile bacağın sonuna
# ulaşıp GCS planındaki iniş sekansına geçer — ama bu SÜREYLE SINIRLI DEĞİL
# (hedefsiz sürünce ne kadar sürer bilinmez) ve daha önemlisi FC'nin kendi
# ilerleyişi hiçbir zaman build_and_start_drop_mission'ı TETİKLEMEZ; bu
# zaman aşımı hâlâ gerekli — kod tarafının sonsuza dek beklemeyip elindeki
# hedeflerle (varsa) drop misyonunu kurup FC'yi kontrollü şekilde devralması
# için.
SEARCH_TOTAL_TIMEOUT_SEC   = 300
# nearest_telemetry_at eşleşmesi bu yaştan eskiyse tespit örneği REDDEDİLİR
# (eskiden yalnızca log uyarısı vardı, hesap yine de yapılıyordu) — bayat
# attitude/pozisyon ile projeksiyon "belirsiz yönlerde" büyük sapmaların ana
# kaynaklarından (2026-08-17, 277m sapma analizi).
TELEMETRY_MATCH_MAX_AGE_S  = 0.25
# 2026-08-20: drone.connect()'in KENDİSİ bazen hiç dönmüyor (gözlemlenen saha
# olayı: seri port o an yoksa/USB-CDC ACM cihazı kaybolmuşsa mavsdk_server
# içeride askıda kalabiliyor) — main.py artık bu çağrıyı da ayrıca bu süreyle
# sarmalıyor, aksi halde tek bir askıda kalan deneme mission kodunu SAATLERCE
# (gözlemlenen: 19 saat) hiç çalıştırmadan durdurabiliyordu.
FC_CONNECT_ATTEMPT_TIMEOUT_SEC = 8
FC_CONNECT_TIMEOUT_SEC     = 10   # connect() döndükten SONRA is_connected için bekleme
FC_RECONNECT_INTERVAL_SEC  = 3    # Bağlantı kurulamadığında/koptuğunda (ör. kalibrasyon
                                   # sonrası FC reboot, ya da seri port henüz gelmemiş)
                                   # yeniden deneme aralığı — main.py'deki ilk-bağlantı
                                   # döngüsü VE mission.fc_connection_task'ın kopma-sonrası
                                   # yeniden bağlanması ikisi de bunu kullanır, SÜRESİZ dener.

# --- Tarama / drop WP yönetimi ---
# NOT (2026-08-20): hız yönetimi (DO_CHANGE_SPEED) koddan tamamen kaldırıldı
# — uçak kendi (GCS/FC'de ayarlı) hızında uçar, kod tarafı yalnızca WP koyar.
SEARCH_START_WP     = 15     # Bu WP'ye gelince tarama moduna geç (GCS planına göre ayarla)
DETECTION_ACTIVE_WP = 15     # Bu WP'ye gelince tespit aktif olur (GCS planına göre ayarla)
SEARCH_LOOP_EXIT_WP = 30     # Bu index'ten itibaren olan öğeler "iniş sekansı" sayılır ve yeni drop
                              # misyonuna olduğu gibi eklenir (bkz. mission.build_and_start_drop_mission).
                              # 2026-08-19: GCS planı artık DO_JUMP KULLANMIYOR (tarama döngüsü farklı
                              # şekilde kuruldu) — mission.py'deki DO_JUMP koruması (iniş sekansında
                              # kaçak bir DO_JUMP bulunursa öğeyi atıp yüksek sesle loglar) yine de
                              # SAFETY NET olarak kod tarafında duruyor, plan yanlışlıkla eski haline
                              # dönerse sessiz felaketi önler.

# Kod tarafının KENDİ oluşturduğu drop misyonundaki hedef WP bloğu (mavi+kırmızı)
# sonuna eklenen DO_JUMP'ın tekrar hakkı (2026-08-26): bir hedef WP'sinin kabul
# yarıçapına hiç girilemediği (dar dönüş/rüzgar) ya da along/cross-track
# penceresinin kaçırıldığı (bkz. mission.drop_trigger_task) durumda uçak hedef
# bloğunu bu kadar defa DAHA tekrar dener — SINIRLI, çünkü bir hedef gerçekten
# ulaşılamazsa uçak sonsuza dek dönmemeli (bkz. SEARCH_TOTAL_TIMEOUT_SEC gibi,
# süresiz bekleme köke inen bir hata sınıfı). Bu SATIR yukarıdaki "GCS planı
# DO_JUMP kullanmıyor" notuyla ÇELİŞMİYOR — o not GCS'ten YÜKLENEN plan için,
# bu ise koddan üretilen drop misyonuna ÖZGÜ, ayrı bir DO_JUMP.
#
# NOT (2026-08-27): TEK hedef varsa DO_JUMP bloğun başına (=tek hedefin
# kendi WP'si) DEĞİL, bir öncesindeki tarama-girişine-dönüş öğesine atlar —
# uçak WP'yi yeni geçmişken kendi üstüne sıçratılırsa hizalanacak bacağı
# olmadan kararsız/dar bir dönüşe zorlanır (bkz. mission.
# build_and_start_drop_mission). 2+ hedefte bloğun başına atlamak sorunsuz.
DROP_RETRY_PASS_COUNT = 2

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

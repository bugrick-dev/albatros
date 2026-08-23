"""Modul bazli log dosyalari + birlesik stdout akisi kurulumu.

Her modul kendi dosyasina yazar (pipeline.log, vision.log, ...) ve ayni anda
stdout'a da basar — stdout systemd tarafindan albatros.log'a yazilmaya devam
eder, boylece kronolojik/birlesik goruntu de korunur.
"""
import logging
import time

LOG_DIR = "/home/albatros/logs"
MODULES = ["main", "pipeline", "vision", "mission", "geo"]

# 2026-08-23: FLUSH_INTERVAL_SEC — bkz. BufferedFileHandler notu.
FLUSH_INTERVAL_SEC = 1.0


class BufferedFileHandler(logging.FileHandler):
    """Standart FileHandler her emit()'te (yani her log.info() çağrısında)
    flush() çağırır — bu, altta yatan disk (SD/SSD) o an başka bir yazma ile
    meşgulse senkron olarak bloklayabilir. Canlı teşhiste (2026-08-23) bunun
    gerçek bir maliyet olduğu ölçüldü: aynı kod bloğunda art arda basılan iki
    log satırı arasında ~2 saniyeye kadar gecikme gözlendi — ana işleme
    döngüsünü de birlikte bloklayabiliyordu (bkz. vision.py hud_preview
    thread ayrımı notu, aynı kökten farklı bir semptom).

    Bu handler flush()'ı SEYRELTİR: normal satırlar en fazla flush_interval
    saniyede bir gerçek diske yazılır (arada Python'ın kendi dosya tamponunda
    bekler). "Önemli" satırlar HER ZAMAN anında flush edilir — bir arıza/uyarı
    anındaki tanı bilgisi asla tampon içinde geciktirilmez.

    NOT: bu kod tabanında TÜM loglar log.info() ile basılıyor (log.warning/
    error hiç kullanılmıyor, bkz. 2026-08-23 taraması) — yani standart
    logging seviyesi (record.levelno) burada ayırt edici değil. Bunun yerine
    kod tabanının kendi (zaten tutarlı kullanılan) önem işaretlerine bakılıyor
    ("⚠", "✗", "HATA", "hata", "ÖLDÜ") — bu karakterleri içeren satırlar
    seviyesinden bağımsız hep anında flush edilir.

    Süreç TEMİZ kapanırsa (Ctrl+C, systemd stop) logging modülünün kendi
    atexit kancası (logging.shutdown()) tüm handler'ları otomatik flush eder,
    veri kaybı olmaz — yalnızca ANİ çökme/kill -9 durumunda son
    flush_interval saniyelik pencere kaybedilme riski taşır (rutin satırlar
    için kabul edilebilir; önem işaretli satırlar zaten anında yazılıyor).
    """

    _URGENT_MARKERS = ("⚠", "✗", "HATA", "hata", "ÖLDÜ")

    def __init__(self, *args, flush_interval=FLUSH_INTERVAL_SEC, **kwargs):
        super().__init__(*args, **kwargs)
        self._flush_interval = flush_interval
        self._last_flush = time.monotonic()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
        except Exception:
            self.handleError(record)
            return
        now = time.monotonic()
        urgent = any(marker in msg for marker in self._URGENT_MARKERS)
        if urgent or (now - self._last_flush) >= self._flush_interval:
            self.flush()
            self._last_flush = now


def setup_logging():
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    for name in MODULES:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_handler = BufferedFileHandler(f"{LOG_DIR}/{name}.log")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console)

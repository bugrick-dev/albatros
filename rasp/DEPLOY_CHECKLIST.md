# Yeni Raspberry Pi / Yeni Gövdeye Geçiş Kontrol Listesi

Bu repo (özellikle `config.py`) **donanıma özel** değerler barındırıyor. Diski
klonlayıp (`rpi-clone`) veya imaj (`dd`/Raspberry Pi Imager) alıp başka bir
Pi'ye/gövdeye taşırsan, aşağıdakileri **mutlaka** gözden geçir — aksi halde
yanlış donanımın ayarlarıyla uçmuş olursun.

## Disk imajı/klon almak için (özet — 2026-08-24 sohbetinde konuşuldu)

Şu an bu Pi, microSD değil **USB'ye takılı disk/SSD**'den çalışıyor
(`/dev/sda`, ~14.6GB — 512M boot/fat32 + 14.1G root/ext4).

- **Yöntem A — Klon (önerilen):** yeni SD kart/SSD'yi bu Pi'ye takıp
  `sudo systemctl stop albatros && sudo rpi-clone <yeni-disk>`
  (rpi-clone yoksa `sudo apt install rpi-clone`). Kullanılmayan alanı
  kopyalamaz, UUID'leri otomatik düzeltir, doğrudan bootable çıkar.
- **Yöntem B — Tam imaj dosyası (arşiv/paylaşım için):**
  `sudo systemctl stop albatros && sudo dd if=/dev/sda of=albatros-backup.img bs=4M status=progress conv=sync,noerror`
  gerekirse `pishrink.sh` ile küçültüp Raspberry Pi Imager'ın "Use custom
  image" seçeneğiyle yeni karta yaz.
- İkisinde de önce `albatros.service`'i durdurmak, log dosyalarının yarım
  yazılmasını önler.

## Sıfırdan kurulum (disk klonlamadan, temiz Raspberry Pi OS üzerine) — 2026-09-03

Disk klon/imaj değil de temiz bir Raspberry Pi OS kurulup üzerine bu repo
git clone edilerek deploy ediliyorsa (albatros2'ye yapılan ilk kurulum böyleydi),
yukarıdaki "Yöntem A/B" ile gelen OS-seviyesi dosyalar **hiç yok** —
apt paketleri + WFB anten sürücüsü elle kurulmalı. Aksi halde "servis
çalışıyor ama görüntü/telemetri gelmiyor" gibi sessiz bir arızayla
karşılaşırsın (2026-09-03, albatros2 ilk kurulumunda ~30dk kaybedildi):

- [ ] **`/boot/firmware/config.txt`** — `[all]` altına `dtoverlay=disable-bt`
      ve `enable_uart=1` eklenmeli (FC UART bağlantısı için), yoksa
      `/dev/ttyAMA0`/`/dev/serial0` hiç oluşmaz. Reboot gerektirir.
- [ ] **WFB anten sürücüsü (`rtl88x2bu` DKMS) — EN KRİTİK ADIM.** Temiz bir
      Raspberry Pi OS, TP-Link 802.11ac (RTL8812BU/8822BU) dongle'ı kendi
      **mainline `rtw88_8822bu`/`rtw88_usb` sürücüsüyle** otomatik yükler —
      bu sürücü wfb-ng'in ihtiyaç duyduğu monitor-mode injection'ı düzgün
      desteklemiyor VE kanal 157'yi (5785MHz, UNII-3) sessizce reddediyor
      (arayüz kanal 1'de/2.4GHz'de takılı kalıyor, hata basmıyor). Çözüm:
      rpi1'deki `/usr/src/rtl88x2bu-5.13.1` kaynağını + `/etc/modprobe.d/`
      altındaki `88x2bu.conf`, `blacklist-8192cu.conf`, `blacklist-rtw88.conf`,
      `rfkill_default.conf` dosyalarını yeni Pi'ye taşı, `dkms add/build/install`
      ile derle, `rmmod rtw88_8822bu rtw88_8822b rtw88_usb rtw88_core` ile
      yanlış sürücüyü kaldırıp `modprobe 88x2bu` ile doğrusunu yükle.
      Doğrulama: `ethtool -i wlan1` → `driver: rtl88x2bu` görünmeli
      (`rtw88_8822bu` DEĞİL).
- [ ] **Regulatory domain — `iw reg get` `US` olmalı.** rpi1 `US: DFS-FCC`
      kullanıyor (kanal 157/5785MHz bunun altında 30dBm'e kadar serbest).
      Türkiye'nin `TR` domain'i bu kanalı hiç tanımıyor — kernel reg domain'i
      TR ise kanal 157 ayarı sessizce yok sayılır. `sudo iw reg set US`
      SADECE o an için geçerli, **reboot'ta sıfırlanır** — kalıcı kaynak
      `/boot/firmware/cmdline.txt` içindeki `cfg80211.ieee80211_regdom=US`
      kernel parametresi (Raspberry Pi Imager'da kurulum sırasında ülke
      Türkiye seçilirse burada otomatik `TR` yazılıyor, bu durum albatros2'de
      tam olarak buydu). `cmdline.txt`'yi düzenleyip reboot at, `iw reg get`
      ile kalıcılığı doğrula.
- [ ] **Kart hâlâ "channel is disabled" diyorsa (yukarıdaki ikisi doğruyken)
      — `rtw_regd_src`/`rtw_country_code` modül parametrelerine bak.**
      albatros2'de kernel cmdline `US` VE global `iw reg get` `US` olduğu
      halde 88x2bu sürücüsü kanal 157'yi reddetmeye devam etti — kart
      `rtw_regd_src=0` (Realtek'in kendi dahili tablosu) ile yüklenmiş,
      OS'un regdomain'ini tamamen görmezden geliyordu. `88x2bu.conf`'a
      `rtw_regd_src=1 rtw_country_code=US` ekleyip (`rmmod 88x2bu &&
      modprobe 88x2bu`) düzeldi. Doğrulama: `cat
      /sys/module/88x2bu/parameters/rtw_regd_src` → `1`,
      `.../rtw_country_code` → `US`.
- [ ] Yukarıdaki maddeler düzeltildikten sonra `iw dev <iface> info`
      çıktısında `channel 157 (5785 MHz)` görünmeli VE `pipeline.log`'da
      `✓  dev wlan1 set channel 157 HT20` (✗ değil) yazmalı — görünmüyorsa
      yayın yer istasyonuna ulaşmaz. Kalıcılığı **mutlaka reboot ile test
      et** (module/regdomain state'i reboot'ta sıfırlanabiliyor,
      dosyaya yazmak yetmiyor — gerçekten uygulandığını görmek lazım).

## Yeni donanımda MUTLAKA güncellenmesi gerekenler

- [ ] **`WFB_MAC` / `WFB_MAC_BACKUP`** (`config.py`) — yeni Pi'nin
      anten(ler)inin MAC adresi farklı olacak (`ip link` ile bul, bkz.
      `pipeline._find_iface_by_any_mac` notu).
- [ ] **`GEOFENCE_POLYGON`** (`config.py`) — başka sahada uçacaksan
      mevcut 4 köşe (yarışma alanına özel) geçersiz, yeni sahanın
      koordinatlarıyla değiştir. Boş liste (`[]`) fence'i devre dışı
      bırakır.
- [ ] **`FC_PORT`** — `_resolve_fc_port()` `/dev/serial/by-id/` üzerinden
      otomatik buluyor ama farklı bir FC/USB kablosuysa doğrula.
- [ ] **Kamera kalibrasyonu** (`camera_calib.json` varsa) — kameraya özel,
      başka bir kamera ünitesinde geçersiz; yoksa `config.py`'deki
      `CAMERA_FOV_H/V` yedek değerleri kullanılır (daha az doğru).
- [ ] **`HEARTBEAT_LED_GPIO`** (`config.py`) — LED yeni gövdede farklı bir
      pine bağlanmışsa güncelle.
- [ ] **`SEARCH_START_WP` / `DETECTION_ACTIVE_WP` / `SEARCH_LOOP_EXIT_WP`**
      — GCS'de yüklenen misyon planı farklıysa bu WP index'leri de farklı
      olabilir.
- [ ] **`WFB_KEY_PATH`** (`/home/albatros/gs.key`) — yeni Pi'de bu anahtar
      dosyası yerinde mi, yer istasyonundaki eşiyle uyuşuyor mu kontrol et.

## Sonrasında

- `sudo systemctl status albatros` ile servisin ayağa kalktığını doğrula.
- Kısa bir masa testi + mümkünse kısa bir test uçuşuyla tüm zinciri
  (video, tespit, WFB, FC bağlantısı) doğrula.

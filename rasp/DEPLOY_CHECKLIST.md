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

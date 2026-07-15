#!/usr/bin/env python3
"""
HEARTBEAT + PARAMETRE OKUMA TESTİ — uçuş gerektirmez, elle çalıştırılır.
Kod tabanı orijinal olarak normal Pixhawk için yazılmıştı; artık Cube Orange
kullanılıyor. Bu test yalnızca MAVSDK bağlantısını doğrular — kamera, servo,
video pipeline YOK. Amaç: yeni FC ile
  1) HEARTBEAT akışının geldiğini,
  2) otopilot/donanım bilgisinin (Cube Orange) doğru okunduğunu,
  3) parametre okumanın (get_param_int/float) çalıştığını
elle görmek.

Ne yapar:
  1. FC'ye bağlanır (varsayılan: config.FC_PORT @ config.FC_BAUDRATE).
  2. core.connection_state() ile bağlantı kurulana kadar bekler.
  3. drone.info üzerinden version/product/identification bilgisini basar.
  4. mavlink_direct.message("HEARTBEAT") ile HEARTBEAT_COUNT adet ham
     HEARTBEAT mesajı dinler ve alanlarını (autopilot/type/base_mode/
     system_status) basar.
  5. PARAM_NAMES listesindeki parametreleri okur (önce int, olmazsa float
     olarak dener) ve değerlerini basar. Hiçbir parametre YAZILMAZ.

Çalıştırma:
    cd rasp/tests
    python3 test_heartbeat_params.py [--conn serial:///dev/ttyAMA3:115200]

Notlar:
  - FC bağlı olmalı: Cube Orange (bench'te de çalışır, arm gerekmez).
  - Salt okunur bir testtir — FC'ye hiçbir komut/misyon/parametre yazma
    gönderilmez.
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# SIGTERM'i (örn. `timeout`, `kill`, systemd stop) düzgün bir çıkışa çevir ki
# mavsdk'nin atexit/​__del__ temizliği çalışıp mavsdk_server alt sürecini
# öldürsün. Aksi halde SIGTERM Python'u atexit çalıştırmadan sonlandırır ve
# mavsdk_server öksüz kalıp portu (FC_PORT) açık tutmaya devam eder — bir
# sonraki çalıştırmada "port meşgul" hatasına yol açan asıl sebep budur.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

from mavsdk import System
from mavsdk.param import ParamError

import config

HEARTBEAT_COUNT = 5   # kaç HEARTBEAT mesajı dinlenip basılacak, sonra durulacak

# Otopilot/donanım doğrulaması için okunacak genel ArduPilot parametreleri.
# Hepsi salt okunur şekilde denenir — bulunamayan/tipi uymayan parametre
# testi durdurmaz, sadece HATA olarak basılır.
PARAM_NAMES = [
    "SYSID_THISMAV",
    "ARMING_CHECK",
    "BRD_SERIAL_NUM",
    "STAT_BOOTCNT",
    "BATT_CAPACITY",
]


def free_stale_mavsdk_server(port):
    """Önceki bir çalıştırma SIGKILL/bağlantı kopması gibi temiz olmayan bir
    şekilde sonlandıysa, mavsdk'nin başlattığı mavsdk_server alt süreci
    öksüz kalıp portu (TIOCEXCL ile) açık tutmaya devam edebilir — sonraki
    çalıştırma "port meşgul" ile başarısız olur. Bağlanmadan önce yalnızca
    mavsdk_server adlı öksüzleri temizle; portu başka biri (main.py,
    serial-getty, GCS) tutuyorsa dokunmadan sadece uyar."""
    try:
        pids = subprocess.check_output(
            ["fuser", port], stderr=subprocess.DEVNULL
        ).split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    killed = False
    for pid_bytes in pids:
        pid = int(pid_bytes)
        try:
            with open(f"/proc/{pid}/comm") as f:
                name = f.read().strip()
        except OSError:
            continue
        if name == "mavsdk_server":
            print(f"[TEST] ⚠ Öksüz mavsdk_server (PID {pid}) {port} portunu açık "
                  f"tutuyor — sonlandırılıyor")
            try:
                os.kill(pid, signal.SIGKILL)
                killed = True
            except OSError:
                pass
        else:
            print(f"[TEST] ⚠ {port} portu PID {pid} ({name}) tarafından açık — "
                  f"MAVSDK bağlanamayabilir")
    if killed:
        time.sleep(0.5)


async def wait_connected(drone, timeout=config.FC_CONNECT_TIMEOUT_SEC):
    print(f"[TEST] Bağlantı bekleniyor (max {timeout}s)...")

    async def _wait():
        async for s in drone.core.connection_state():
            if s.is_connected:
                return

    try:
        await asyncio.wait_for(_wait(), timeout=timeout)
        print("[TEST] ✓ Bağlandı")
    except asyncio.TimeoutError:
        print(f"[TEST] ✗ {timeout}s içinde HEARTBEAT alınamadı — MAVSDK bağlanamadı.")
        print("[TEST]   Fiziksel sinyal olması MAVLink paketi geldiği anlamına gelmez. Kontrol edin:")
        print(f"[TEST]   - Baud rate: config.FC_BAUDRATE={config.FC_BAUDRATE} FC'nin bu porta atanmış "
              f"SERIALx_BAUD parametresiyle birebir aynı mı? (Cube Orange'da UART haritası Pixhawk'tan farklı olabilir)")
        print("[TEST]   - SERIALx_PROTOCOL bu port için 2 (MAVLink2) mi? (1=MAVLink1, farklı bir değerse hiç paket üretilmez)")
        print("[TEST]   - Port başka bir process tarafından açık mı? (serial-getty@ttyAMA3, main.py, "
              "GCS/Mission Planner aynı anda bağlıysa MAVSDK veri okuyamaz)")
        print("[TEST]   - TX/RX kabloları çapraz mı bağlı? (FC TX → Pi RX, FC RX → Pi TX)")
        print("[TEST]   - `sudo systemctl status serial-getty@ttyAMA3` ile konsol servisi kapalı mı doğrulayın "
              "(RPi'de seri konsol açıksa portu paylaşır/karıştırır).")
        raise


async def print_autopilot_info(drone):
    print(f"\n{'='*72}\nOTOPİLOT / DONANIM BİLGİSİ\n{'='*72}")
    try:
        version = await drone.info.get_version()
        print(f"  Flight SW  : {version.flight_sw_major}.{version.flight_sw_minor}."
              f"{version.flight_sw_patch} (git={version.flight_sw_git_hash})")
        print(f"  OS SW      : {version.os_sw_major}.{version.os_sw_minor}."
              f"{version.os_sw_patch} (git={version.os_sw_git_hash})")
    except Exception as e:
        print(f"  ✗ get_version() başarısız: {e}")

    try:
        product = await drone.info.get_product()
        print(f"  Ürün       : vendor={product.vendor_name} ({product.vendor_id})  "
              f"product={product.product_name} ({product.product_id})")
    except Exception as e:
        print(f"  ✗ get_product() başarısız: {e}")

    try:
        ident = await drone.info.get_identification()
        print(f"  Donanım UID: {ident.hardware_uid}  legacy_uid={ident.legacy_uid}")
    except Exception as e:
        print(f"  ✗ get_identification() başarısız: {e}")
    print("=" * 72 + "\n")


async def listen_heartbeats(drone, count):
    print(f"[TEST] {count} adet ham HEARTBEAT mesajı dinleniyor "
          f"(mavlink_direct.message)...")
    n = 0
    async for msg in drone.mavlink_direct.message("HEARTBEAT"):
        n += 1
        fields = json.loads(msg.fields_json)
        print(f"[HEARTBEAT] #{n} sys={msg.system_id} comp={msg.component_id} "
              f"type={fields.get('type')} autopilot={fields.get('autopilot')} "
              f"base_mode={fields.get('base_mode')} "
              f"system_status={fields.get('system_status')}")
        if n >= count:
            break
    print(f"[TEST] ✓ {n} HEARTBEAT mesajı alındı — bağlantı canlı\n")


async def read_param(drone, name):
    """Önce int, olmazsa float olarak okumayı dener."""
    try:
        value = await drone.param.get_param_int(name)
        print(f"[PARAM] {name:<16} = {value}  (int)")
        return
    except ParamError:
        pass
    try:
        value = await drone.param.get_param_float(name)
        print(f"[PARAM] {name:<16} = {value}  (float)")
        return
    except ParamError as e:
        print(f"[PARAM] {name:<16} ✗ okunamadı: {e}")


async def read_all_params(drone, names):
    print(f"[TEST] {len(names)} parametre okunuyor...")
    for name in names:
        await read_param(drone, name)


async def main(conn):
    print("=" * 60)
    print("HEARTBEAT + PARAMETRE OKUMA TESTİ (Cube Orange)")
    print(f"  FC : {conn}")
    print("=" * 60)

    fc_port = conn.split("://", 1)[-1].rsplit(":", 1)[0]
    free_stale_mavsdk_server(fc_port)

    drone = System()
    await drone.connect(system_address=conn)
    try:
        await wait_connected(drone)
    except asyncio.TimeoutError:
        sys.exit(1)

    await print_autopilot_info(drone)
    await listen_heartbeats(drone, HEARTBEAT_COUNT)
    await read_all_params(drone, PARAM_NAMES)

    print("[TEST] ✓ Test tamamlandı")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conn", default=f"serial://{config.FC_PORT}:{config.FC_BAUDRATE}",
        help="MAVSDK bağlantı adresi (örn. serial://%(default)s veya udp://:14540 SITL için)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.conn))
    except KeyboardInterrupt:
        print("\n[TEST] Ctrl+C — durduruldu")

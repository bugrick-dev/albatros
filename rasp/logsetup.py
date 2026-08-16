"""Modul bazli log dosyalari + birlesik stdout akisi kurulumu.

Her modul kendi dosyasina yazar (pipeline.log, vision.log, ...) ve ayni anda
stdout'a da basar — stdout systemd tarafindan albatros.log'a yazilmaya devam
eder, boylece kronolojik/birlesik goruntu de korunur.
"""
import logging

LOG_DIR = "/home/albatros/logs"
MODULES = ["main", "pipeline", "vision", "mission", "geo"]


def setup_logging():
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    for name in MODULES:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_handler = logging.FileHandler(f"{LOG_DIR}/{name}.log")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console)

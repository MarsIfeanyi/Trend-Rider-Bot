"""
logger.py
─────────
Centralised logging configuration.
Import `log` from this module anywhere in the project.

Usage:
    from logger import log
    log.info("Bot started")
    log.warning("Low margin")
    log.error("Order failed")
"""

import logging
import config


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("EMABot")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(module)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # ── File handler ──────────────────────────────────────────────
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers on re-import
    if not logger.handlers:
        logger.addHandler(console)
        logger.addHandler(file_handler)

    return logger


log = _build_logger()

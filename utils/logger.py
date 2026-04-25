"""utils/logger.py — Logging setup."""
import logging
import sys


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        stream  = sys.stdout,
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "yfinance", "peewee",
                  "lightgbm", "xgboost", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

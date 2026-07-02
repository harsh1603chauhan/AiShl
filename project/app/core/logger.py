from __future__ import annotations

import logging

from app.core.config import settings


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format=_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

"""Logging setup and a small structured request logger."""
from __future__ import annotations

import hashlib
import json
import logging
import sys

_LOGGER_NAME = "acme"


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger(_LOGGER_NAME)
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def question_fingerprint(question: str) -> dict:
    """Log a question's size and a short hash rather than the raw text."""
    return {
        "question_chars": len(question),
        "question_sha1": hashlib.sha1(question.encode("utf-8")).hexdigest()[:10],
    }


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info("%s %s", event, json.dumps(fields, default=str, ensure_ascii=False, sort_keys=True))

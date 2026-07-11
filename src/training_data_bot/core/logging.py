"""
Logging helpers for the training data bot.
"""

from contextlib import ContextDecorator
import logging
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, Optional

from .config import settings


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (ValueError, UnicodeError):
        pass
    return value


def redact_log_value(value: Any) -> str:
    """Remove URL queries/credentials, secret values, and oversized payloads."""
    text = str(value)
    text = re.sub(r"https?://[^\s,;]+", lambda m: _sanitize_url(m.group(0)), text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)(api[_-]?key|authorization|token|password|secret)\s*[=:]\s*[^\s,;]+", r"\1=<redacted>", text)
    return text[:512]


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger without duplicating handlers."""

    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    return logger


class LogContext(ContextDecorator):
    """Small context manager that logs operation start and finish."""

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, **fields: Any):
        self.operation = operation
        self.logger = logger or get_logger("training_data_bot.context")
        self.fields: Dict[str, Any] = {key: redact_log_value(value) for key, value in fields.items()}

    def __enter__(self):
        self.logger.debug("Starting %s %s", self.operation, self.fields)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.debug("Failed %s: %s", self.operation, redact_log_value(exc_val))
        else:
            self.logger.debug("Finished %s", self.operation)
        return False

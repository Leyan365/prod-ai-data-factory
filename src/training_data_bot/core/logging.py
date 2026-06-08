"""
Logging helpers for the training data bot.
"""

from contextlib import ContextDecorator
import logging
from typing import Any, Dict, Optional

from .config import settings


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
        self.fields: Dict[str, Any] = fields

    def __enter__(self):
        self.logger.debug("Starting %s %s", self.operation, self.fields)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.debug("Failed %s: %s", self.operation, exc_val)
        else:
            self.logger.debug("Finished %s", self.operation)
        return False

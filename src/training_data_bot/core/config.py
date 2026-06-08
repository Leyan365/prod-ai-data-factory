"""
Application configuration for the training data bot.

This module intentionally keeps configuration small for the tutorial slice.
Later production steps can replace this with environment-backed settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings used by the current package implementation."""

    app_name: str = "training-data-bot"
    log_level: str = "INFO"
    default_encoding: str = "utf-8"
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 100
    output_dir: Path = Path("output")


settings = Settings()

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
    default_min_chunk_chars: int = 1
    preserve_paragraphs: bool = True
    default_ai_provider: str = "mock"
    gemini_model: str = "gemini-1.5-flash"
    ai_timeout: int = 60
    ai_max_retries: int = 3
    quality_min_input_length: int = 3
    quality_min_output_length: int = 10
    quality_min_overall_score: float = 0.6
    quality_min_metric_score: float = 0.5
    quality_min_relevance_overlap: float = 0.08
    quality_duplicate_threshold: float = 0.9
    quality_blocked_terms: tuple[str, ...] = (
        "hate",
        "kill",
        "violence",
        "slur",
    )
    output_dir: Path = Path("output")


settings = Settings()

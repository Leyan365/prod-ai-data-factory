"""
Application configuration for the training data bot.

This module intentionally keeps configuration small for the tutorial slice.
Later production steps can replace this with environment-backed settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet


@dataclass(frozen=True)
class RemoteFetchPolicy:
    """Secure-by-default policy for loading remote documents."""

    enabled: bool = False
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
    allow_http: bool = False


@dataclass(frozen=True)
class ResourceLimits:
    """Hard resource ceilings for one ingestion run."""

    max_document_bytes: int = 25 * 1024 * 1024
    max_run_bytes: int = 100 * 1024 * 1024
    max_remote_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 200
    max_csv_rows: int = 100_000
    request_timeout_seconds: int = 30
    max_concurrent_fetches: int = 4

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_remote_bytes > self.max_document_bytes:
            raise ValueError("max_remote_bytes cannot exceed max_document_bytes")


@dataclass(frozen=True)
class QualityGatePolicy:
    """Controls aggregate quality enforcement at publication boundaries."""

    enforce_aggregate: bool = True
    allow_low_quality_export: bool = False


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
    remote_fetch: RemoteFetchPolicy = field(default_factory=RemoteFetchPolicy)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    quality_gate: QualityGatePolicy = field(default_factory=QualityGatePolicy)


settings = Settings()

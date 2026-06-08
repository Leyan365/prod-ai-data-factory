"""
Domain exceptions for the training data bot.
"""

from typing import Iterable, Optional


class TrainingDataBotError(Exception):
    """Base exception for package-specific errors."""

    def __init__(self, message: str, *, detail: Optional[str] = None, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.cause = cause

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} ({self.detail})"
        if self.cause:
            return f"{self.message} ({self.cause})"
        return self.message


class ConfigurationError(TrainingDataBotError):
    """Raised when runtime configuration or component setup fails."""


class DocumentLoadError(TrainingDataBotError):
    """Raised when a document source cannot be loaded."""

    def __init__(
        self,
        message: str,
        *,
        file_path: Optional[str] = None,
        detail: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ):
        self.file_path = file_path
        super().__init__(message, detail=detail, cause=cause)


class DocumentLoadingError(DocumentLoadError):
    """Alias-style subclass used by the unified and web loaders."""


class UnsupportedFormatError(DocumentLoadError):
    """Raised when a source extension or type is unsupported."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        file_format: str = "unknown",
        supported_formats: Optional[Iterable[str]] = None,
        detail: Optional[str] = None,
    ):
        self.file_format = file_format
        self.supported_formats = list(supported_formats or [])
        supported = ", ".join(self.supported_formats) or "none"
        super().__init__(
            message or f"Unsupported format '{file_format}'. Supported formats: {supported}",
            detail=detail,
        )

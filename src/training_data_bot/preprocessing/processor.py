"""
Text preprocessing pipeline for loaded documents.
"""

from dataclasses import dataclass
import hashlib
import re
from typing import List
from uuid import NAMESPACE_URL, UUID, uuid5

from ..core.config import settings
from ..core.exceptions import PreprocessingError
from ..core.logging import get_logger
from ..core.models import Document, TextChunk


CHUNK_NAMESPACE = uuid5(NAMESPACE_URL, "training-data-bot:text-chunk:v1")


@dataclass(frozen=True)
class ChunkSpan:
    """A normalized text span selected for chunk creation."""

    start_index: int
    end_index: int
    content: str


class TextPreprocessor:
    """Normalize documents and split them into deterministic overlapping chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *,
        normalize_whitespace: bool = True,
        strip_control_chars: bool = True,
        preserve_paragraphs: bool | None = None,
        min_chunk_chars: int | None = None,
    ):
        self.logger = get_logger("training_data_bot.preprocessing")
        self.chunk_size = settings.default_chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap
        self.normalize_whitespace = normalize_whitespace
        self.strip_control_chars = strip_control_chars
        self.preserve_paragraphs = (
            settings.preserve_paragraphs if preserve_paragraphs is None else preserve_paragraphs
        )
        self.min_chunk_chars = (
            settings.default_min_chunk_chars if min_chunk_chars is None else min_chunk_chars
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_chars < 0:
            raise ValueError("min_chunk_chars cannot be negative")

    async def process_documents(self, document: Document) -> List[TextChunk]:
        """Return normalized chunks for one document."""

        try:
            if not isinstance(document.content, str):
                raise PreprocessingError(
                    "Document content must be a string",
                    detail=f"document_id={getattr(document, 'id', 'unknown')}",
                )

            normalized_text = self.normalize_text(document.content)
            if not normalized_text:
                return []

            spans = self.chunk_text(normalized_text)
            return [
                self.create_chunk(document, normalized_text, span, chunk_index)
                for chunk_index, span in enumerate(spans)
            ]
        except PreprocessingError:
            raise
        except Exception as exc:
            document_id = getattr(document, "id", "unknown")
            self.logger.error("Preprocessing failed for document %s: %s", document_id, exc)
            raise PreprocessingError(
                "Failed to preprocess document",
                detail=f"document_id={document_id}",
                cause=exc,
            ) from exc

    def normalize_text(self, text: str) -> str:
        """Normalize line endings, whitespace, and control characters."""

        if not text or not text.strip():
            return ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if self.strip_control_chars:
            normalized = "".join(
                char
                for char in normalized
                if char in {"\n", "\t"} or ord(char) >= 32
            )

        lines = normalized.split("\n")
        if self.normalize_whitespace:
            lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in lines]
        else:
            lines = [line.strip() for line in lines]

        if self.preserve_paragraphs:
            normalized = "\n".join(lines).strip()
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
            return normalized

        return "\n".join(line for line in lines if line).strip()

    def clean_text(self, text: str) -> str:
        """Backward-compatible alias for text normalization."""

        return self.normalize_text(text)

    def chunk_text(self, text: str) -> List[ChunkSpan]:
        """Split normalized text into deterministic character spans."""

        spans: List[ChunkSpan] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end].strip()

            if len(content) >= self.min_chunk_chars:
                spans.append(ChunkSpan(start_index=start, end_index=end, content=content))

            if end == len(text):
                break
            start += step

        return spans

    def create_chunk(
        self,
        document: Document,
        normalized_text: str,
        span: ChunkSpan,
        chunk_index: int,
    ) -> TextChunk:
        """Create a TextChunk with stable ID and preserved source metadata."""

        chunk_id = self.stable_chunk_id(
            document.id,
            chunk_index,
            span.start_index,
            span.end_index,
            span.content,
        )
        doc_type = getattr(document.doc_type, "value", document.doc_type)

        return TextChunk(
            id=chunk_id,
            document_id=document.id,
            content=span.content,
            start_index=span.start_index,
            end_index=span.end_index,
            chunk_index=chunk_index,
            preceding_context=normalized_text[
                max(0, span.start_index - self.chunk_overlap):span.start_index
            ] or None,
            following_context=normalized_text[
                span.end_index:min(len(normalized_text), span.end_index + self.chunk_overlap)
            ] or None,
            metadata={
                "source_document_title": document.title,
                "source_document_type": doc_type,
                "source_document_source": document.source,
                "source_metadata": dict(document.metadata or {}),
                "normalized": True,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_chunk_chars": self.min_chunk_chars,
                "normalize_whitespace": self.normalize_whitespace,
                "strip_control_chars": self.strip_control_chars,
                "preserve_paragraphs": self.preserve_paragraphs,
            },
        )

    def stable_chunk_id(
        self,
        document_id: UUID,
        chunk_index: int,
        start_index: int,
        end_index: int,
        content: str,
    ) -> UUID:
        """Create a deterministic chunk UUID from document and span data."""

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        seed = f"{document_id}:{chunk_index}:{start_index}:{end_index}:{content_hash}"
        return uuid5(CHUNK_NAMESPACE, seed)

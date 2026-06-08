"""
Basic text preprocessing for the tutorial implementation.
"""

from typing import List

from ..core.config import settings
from ..core.models import Document, TextChunk


class TextPreprocessor:
    """Clean documents and split them into overlapping text chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    async def process_documents(self, document: Document) -> List[TextChunk]:
        """Return cleaned chunks for one document."""

        content = self.clean_text(document.content)
        if not content:
            return []

        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end].strip()

            if chunk_content:
                chunks.append(
                    TextChunk(
                        document_id=document.id,
                        content=chunk_content,
                        start_index=start,
                        end_index=end,
                        chunk_index=chunk_index,
                        preceding_context=content[max(0, start - self.chunk_overlap):start] or None,
                        following_context=content[end:min(len(content), end + self.chunk_overlap)] or None,
                    )
                )
                chunk_index += 1

            if end == len(content):
                break
            start += step

        return chunks

    def clean_text(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks loosely."""

        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

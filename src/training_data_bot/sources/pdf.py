import asyncio
import logging
import shutil
from pathlib import Path
from typing import Union, Optional

from .base import BaseLoader 
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError
from ..core.config import settings
from ..core.logging import LogContext

logger = logging.getLogger(__name__)

class PDFLoader(BaseLoader):
    """
    Advanced PDF Loader optimized for LLM training data.
    Preserves document structure via Markdown and supports OCR for scanned files.
    """

    def __init__(self):
        super().__init__()
        self.supported_formats = [DocumentType.PDF]

    async def load_single(
            self, 
            source: Union[str, Path],
            **kwargs
    ) -> Document:
        """
        Load a PDF document with layout-aware extraction.

        Args:
            source: PDF file path
            **kwargs: 
                - use_ocr (bool): Force OCR on all pages. Default False.
                - extract_images (bool): Whether to include image placeholders.
        """
        source = Path(source)

        if not source.exists():
            raise DocumentLoadError(f"File not found: {source}")
        if source.stat().st_size > settings.resource_limits.max_document_bytes:
            raise DocumentLoadError(f"Document exceeds size limit: {source}")
        
        with LogContext("load_pdf", file=str(source)):
            try:
                # Run the heavy extraction in a separate thread to keep the loop free
                content = await self._extract_pdf_content(source, **kwargs)

                return self.create_document(
                    title=source.stem,
                    content=content,
                    source=source,
                    doc_type=DocumentType.PDF,
                    extraction_method="PDFLoader.pymupdf4llm",
                )
            
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to load PDF file: {source}",
                    file_path=str(source),
                    cause=e
                )

    async def _extract_pdf_content(self, path: Path, **kwargs) -> str:
        """Internal worker for content extraction."""
        def _process():
            try:
                import fitz  # PyMuPDF
            except ImportError as exc:
                raise DocumentLoadError(
                    "Optional dependencies missing for PDF loading: pymupdf and pymupdf4llm. "
                    "Install them with 'pip install pymupdf pymupdf4llm'."
                ) from exc

            try:
                import pymupdf4llm  # High-level LLM wrapper
            except ImportError:
                pymupdf4llm = None

            try:
                with fitz.open(path) as source_pdf:
                    if len(source_pdf) > settings.resource_limits.max_pdf_pages:
                        raise DocumentLoadError("PDF exceeds configured page limit")
            except DocumentLoadError:
                raise
            except Exception as exc:
                raise DocumentLoadError(f"Malformed or unreadable PDF file: {path}") from exc

            # 1. Primary attempt: layout-aware Markdown extraction when available.
            md_content = ""
            if pymupdf4llm is not None:
                try:
                    md_content = pymupdf4llm.to_markdown(str(path)) or ""
                except Exception as exc:
                    logger.warning("pymupdf4llm extraction failed for %s: %s", path.name, exc)
            if md_content.strip():
                return md_content

            # 2. Text PDFs can still be extracted directly by PyMuPDF when the
            # high-level converter returns no usable text.
            try:
                direct_text = self._direct_text_extraction(path, fitz)
            except Exception as exc:
                logger.warning("Direct PyMuPDF extraction failed for %s: %s", path.name, exc)
                direct_text = ""
            if direct_text.strip():
                return direct_text

            # 3. OCR is only attempted when both text paths are empty and the
            # host actually provides Tesseract support.
            if self._ocr_available():
                ocr_content = self._ocr_fallback(path)
                if ocr_content.strip():
                    return ocr_content

            raise DocumentLoadError(
                "PDF contains no extractable text; OCR/Tesseract is required for image-only PDFs"
            )

        return await asyncio.to_thread(_process)

    @staticmethod
    def _direct_text_extraction(path: Path, fitz_module) -> str:
        """Extract page text directly without requiring OCR or layout tooling."""
        with fitz_module.open(path) as source_pdf:
            return "\n\n".join(
                text.strip()
                for page in source_pdf
                if (text := page.get_text("text", sort=True)).strip()
            )

    @staticmethod
    def _ocr_available() -> bool:
        """Return whether the local OCR executable is available."""
        return shutil.which("tesseract") is not None


    def _ocr_fallback(self, path: Path) -> str:
        """Fallback method using PyMuPDF's integrated Tesseract support."""
        import fitz
        text_parts = []
        
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    # 'get_textpage_ocr' triggers Tesseract on the page image
                    # requires tesseract-ocr system package installed
                    try:
                        tp = page.get_textpage_ocr(flags=3, language="eng")
                        text = page.get_text(textpage=tp, sort=True)
                        if text.strip():
                            text_parts.append(text)
                    except Exception as ocr_err:
                        logger.warning(f"OCR failed on page {page.number}: {ocr_err}")
                        continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Total failure during OCR fallback: {e}")
            return ""

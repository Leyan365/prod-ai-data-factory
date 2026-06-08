"""Focused tests for document loader behavior."""

import asyncio
import builtins
import importlib.util
from pathlib import Path

import pytest

from training_data_bot.core.exceptions import DocumentLoadError, DocumentLoadingError
from training_data_bot.core.models import DocumentType
from training_data_bot.sources.documents import DocumentLoader
from training_data_bot.sources.pdf import PDFLoader
from training_data_bot.sources.unified import UnifiedLoader
from training_data_bot.sources.web import WebLoader


def run(coro):
    return asyncio.run(coro)


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_txt_and_markdown_load_plain_content(workspace_tmp):
    loader = DocumentLoader()
    txt = write(workspace_tmp / "notes.txt", "plain text")
    md = write(workspace_tmp / "readme.md", "# Heading\n\nBody text")

    txt_doc = run(loader.load_single(txt))
    md_doc = run(loader.load_single(md))

    assert txt_doc.doc_type == DocumentType.TXT
    assert txt_doc.content == "plain text"
    assert md_doc.doc_type == DocumentType.MD
    assert "# Heading" in md_doc.content


def test_html_strips_scripts_and_styles(workspace_tmp):
    loader = DocumentLoader()
    html = write(
        workspace_tmp / "page.html",
        "<html><head><style>.x{}</style><script>bad()</script></head>"
        "<body><h1>Hello</h1><p>Visible text</p></body></html>",
    )

    doc = run(loader.load_single(html))

    assert doc.doc_type == DocumentType.HTML
    assert "Hello" in doc.content
    assert "Visible text" in doc.content
    assert "bad()" not in doc.content
    assert ".x" not in doc.content


def test_json_dict_and_list_convert_to_readable_text(workspace_tmp):
    loader = DocumentLoader()
    data = write(workspace_tmp / "data.json", '{"name": "Ada", "role": "engineer"}')
    rows = write(workspace_tmp / "rows.json", '[{"id": 1}, {"id": 2}]')

    data_doc = run(loader.load_single(data))
    rows_doc = run(loader.load_single(rows))

    assert "name: Ada" in data_doc.content
    assert "role: engineer" in data_doc.content
    assert "Item 1: {'id': 1}" in rows_doc.content
    assert "Item 2: {'id': 2}" in rows_doc.content


def test_csv_includes_headers_and_valid_rows(workspace_tmp):
    loader = DocumentLoader()
    csv_path = write(workspace_tmp / "items.csv", "name,qty\napple,2\nbad-row\npear,3\n")

    doc = run(loader.load_single(csv_path))

    assert "Headers: name, qty" in doc.content
    assert "Row 1: name: apple | qty: 2" in doc.content
    assert "bad-row" not in doc.content
    assert "Row 3: name: pear | qty: 3" in doc.content


def test_empty_text_markdown_and_csv_are_supported(workspace_tmp):
    loader = DocumentLoader()
    txt = write(workspace_tmp / "empty.txt", "")
    md = write(workspace_tmp / "empty.md", "")
    csv_path = write(workspace_tmp / "empty.csv", "")

    assert run(loader.load_single(txt)).content == ""
    assert run(loader.load_single(md)).content == ""
    assert run(loader.load_single(csv_path)).content == ""


def test_malformed_and_empty_json_raise_document_load_error(workspace_tmp):
    loader = DocumentLoader()
    malformed = write(workspace_tmp / "bad.json", "{not-json")
    empty = write(workspace_tmp / "empty.json", "")

    with pytest.raises(DocumentLoadError):
        run(loader.load_single(malformed))
    with pytest.raises(DocumentLoadError):
        run(loader.load_single(empty))


def test_docx_success_skipped_without_optional_dependency(workspace_tmp):
    if importlib.util.find_spec("docx") is None:
        pytest.skip("python-docx is not installed")

    from docx import Document

    path = workspace_tmp / "sample.docx"
    document = Document()
    document.add_paragraph("Docx paragraph")
    document.save(path)

    doc = run(DocumentLoader().load_single(path))

    assert doc.doc_type == DocumentType.DOCX
    assert "Docx paragraph" in doc.content


def test_docx_dependency_error_names_python_docx(workspace_tmp, monkeypatch):
    loader = DocumentLoader()
    path = write(workspace_tmp / "sample.docx", "not a real docx")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(DocumentLoadError) as exc:
        run(loader.load_single(path))

    assert "python-docx" in str(exc.value)


def test_pdf_success_skipped_without_optional_dependencies(workspace_tmp):
    if importlib.util.find_spec("fitz") is None or importlib.util.find_spec("pymupdf4llm") is None:
        pytest.skip("PDF dependencies are not installed")

    import fitz

    path = workspace_tmp / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello from PDF loader. " * 5)
    pdf.save(path)
    pdf.close()

    doc = run(PDFLoader().load_single(path))

    assert doc.doc_type == DocumentType.PDF
    assert "Hello from PDF loader" in doc.content


def test_pdf_dependency_error_names_required_packages(workspace_tmp, monkeypatch):
    loader = PDFLoader()
    path = write(workspace_tmp / "sample.pdf", "%PDF-1.4\n")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"fitz", "pymupdf4llm"}:
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(DocumentLoadError) as exc:
        run(loader.load_single(path))

    message = str(exc.value)
    assert "pymupdf" in message
    assert "pymupdf4llm" in message


def test_malformed_pdf_raises_document_load_error_when_dependencies_exist(workspace_tmp):
    if importlib.util.find_spec("fitz") is None or importlib.util.find_spec("pymupdf4llm") is None:
        pytest.skip("PDF dependencies are not installed")

    path = write(workspace_tmp / "bad.pdf", "not a pdf")

    with pytest.raises(DocumentLoadError) as exc:
        run(PDFLoader().load_single(path))

    assert "Malformed or unreadable PDF" in str(exc.value)


def test_url_fallback_uses_mocked_fetch_without_network(monkeypatch):
    loader = WebLoader(use_decodo=False)

    async def fake_fetch(url):
        return "<html><title>Example Title</title><body>Example body</body></html>", "mock"

    monkeypatch.setattr(loader, "_fetch_with_fallback", fake_fetch)

    doc = run(loader.load_single("https://example.test/page"))

    assert doc.doc_type == DocumentType.URL
    assert doc.title == "Example Title"
    assert "Example body" in doc.content
    assert doc.extraction_method == "mock"


def test_web_fallback_dependency_error_names_packages(monkeypatch):
    loader = WebLoader(use_decodo=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"httpx", "bs4"}:
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(DocumentLoadingError) as exc:
        run(loader._fetch_with_fallback("https://example.test"))

    message = str(exc.value)
    assert "httpx" in message
    assert "beautifulsoup4" in message


def test_unified_directory_discovery_is_sorted_recursive_and_glob_filtered(workspace_tmp):
    loader = UnifiedLoader()
    write(workspace_tmp / "zeta.txt", "z")
    write(workspace_tmp / "alpha.md", "a")
    nested = workspace_tmp / "nested"
    nested.mkdir()
    write(nested / "beta.txt", "b")
    write(nested / "skip.json", '{"skip": true}')
    write(workspace_tmp / "ignore.bin", "x")

    all_files = loader._find_supported_files(workspace_tmp, recursive=True)
    txt_files = loader._find_supported_files(workspace_tmp, recursive=True, patterns=["*.txt"])
    nested_files = loader._find_supported_files(workspace_tmp, recursive=True, patterns=["nested/*.txt"])
    shallow_files = loader._find_supported_files(workspace_tmp, recursive=False)

    assert [p.relative_to(workspace_tmp).as_posix() for p in all_files] == [
        "alpha.md",
        "nested/beta.txt",
        "nested/skip.json",
        "zeta.txt",
    ]
    assert [p.relative_to(workspace_tmp).as_posix() for p in txt_files] == [
        "nested/beta.txt",
        "zeta.txt",
    ]
    assert [p.relative_to(workspace_tmp).as_posix() for p in nested_files] == ["nested/beta.txt"]
    assert [p.relative_to(workspace_tmp).as_posix() for p in shallow_files] == ["alpha.md", "zeta.txt"]


def test_unified_load_directory_loads_supported_files(workspace_tmp):
    write(workspace_tmp / "b.txt", "second")
    write(workspace_tmp / "a.txt", "first")
    write(workspace_tmp / "ignored.bin", "nope")

    docs = run(UnifiedLoader().load_directory(workspace_tmp, recursive=False))

    assert [doc.title for doc in docs] == ["a", "b"]


def test_unsupported_format_and_missing_file_raise_domain_errors(workspace_tmp):
    unsupported = write(workspace_tmp / "notes.xyz", "nope")
    missing = workspace_tmp / "missing.txt"

    with pytest.raises(DocumentLoadingError) as unsupported_exc:
        run(UnifiedLoader().load_single(unsupported))

    with pytest.raises(DocumentLoadingError) as missing_exc:
        run(UnifiedLoader().load_single(missing))

    assert "not supported" in str(unsupported_exc.value)
    assert "Failed to load document" in str(missing_exc.value)

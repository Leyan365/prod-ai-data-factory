"""Focused tests for document loader behavior."""

import asyncio
import builtins
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from training_data_bot.core.exceptions import DocumentLoadError, DocumentLoadingError
from training_data_bot.core.config import RemoteFetchPolicy
from training_data_bot.decodo import DecodoClient
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


def test_pdf_empty_markdown_falls_back_to_direct_pymupdf_text(workspace_tmp, monkeypatch):
    if importlib.util.find_spec("fitz") is None or importlib.util.find_spec("pymupdf4llm") is None:
        pytest.skip("PDF dependencies are not installed")
    import fitz
    import pymupdf4llm

    path = workspace_tmp / "direct-text.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Direct PyMuPDF text fallback")
    pdf.save(path)
    pdf.close()
    monkeypatch.setattr(pymupdf4llm, "to_markdown", lambda *args, **kwargs: "")

    doc = run(PDFLoader().load_single(path))

    assert "Direct PyMuPDF text fallback" in doc.content


def test_pdf_empty_text_paths_require_ocr_when_unavailable(workspace_tmp, monkeypatch):
    if importlib.util.find_spec("fitz") is None or importlib.util.find_spec("pymupdf4llm") is None:
        pytest.skip("PDF dependencies are not installed")
    import fitz
    import pymupdf4llm

    path = workspace_tmp / "image-only.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    monkeypatch.setattr(pymupdf4llm, "to_markdown", lambda *args, **kwargs: " ")
    monkeypatch.setattr(PDFLoader, "_ocr_available", staticmethod(lambda: False))

    with pytest.raises(DocumentLoadError, match="OCR/Tesseract is required"):
        run(PDFLoader().load_single(path))

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
    loader = WebLoader(
        use_decodo=False,
        remote_fetch_policy=RemoteFetchPolicy(enabled=True, allowed_hosts=frozenset({"example.test"})),
    )
    monkeypatch.setattr("training_data_bot.sources.web.socket.getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))])

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


def test_remote_policy_rejects_private_address(monkeypatch):
    loader = WebLoader(
        use_decodo=False,
        remote_fetch_policy=RemoteFetchPolicy(enabled=True, allowed_hosts=frozenset({"example.test"})),
    )
    monkeypatch.setattr(
        "training_data_bot.sources.web.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(DocumentLoadingError, match="non-public"):
        run(loader.load_single("https://example.test/private"))


def test_direct_decodo_call_fails_closed_by_default():
    with pytest.raises(DocumentLoadingError, match="disabled by policy"):
        run(DecodoClient().scrape_url("https://example.test/?token=secret"))


def test_direct_decodo_rejects_redirects_and_private_hosts(monkeypatch):
    policy = RemoteFetchPolicy(enabled=True, allowed_hosts=frozenset({"example.test"}))
    client = DecodoClient(remote_fetch_policy=policy)
    monkeypatch.setattr("training_data_bot.decodo.socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 443))])
    class Response:
        status_code = 302
        headers = {"location": "https://example.test/next"}
        encoding = "utf-8"
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def aiter_bytes(self): yield b""
        def raise_for_status(self): return None
    class FakeClient:
        def stream(self, *a, **k): return Response()
        async def aclose(self): return None
    client._client = FakeClient()
    with pytest.raises(DocumentLoadingError, match="redirects are disabled"):
        run(client.scrape_url("https://example.test/start"))


def test_log_redaction_removes_query_credentials_and_secrets():
    from training_data_bot.core.logging import redact_log_value
    value = "GET https://user:pass@example.test/path?api_key=abc#fragment authorization=Bearer-secret source=PRIVATE"
    redacted = redact_log_value(value)
    assert "abc" not in redacted and "pass" not in redacted and "Bearer-secret" not in redacted
    assert "?" not in redacted and "#fragment" not in redacted


class _StreamResponse:
    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.encoding = "utf-8"
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None
    async def aiter_bytes(self):
        yield self._body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("status")


def test_web_loader_rejects_excessive_redirect_chain(monkeypatch):
    import httpx
    loader = WebLoader(use_decodo=False, remote_fetch_policy=RemoteFetchPolicy(enabled=True, allowed_hosts=frozenset({"example.test"})))
    monkeypatch.setattr("training_data_bot.sources.web.socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 443))])
    class Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        def stream(self, *a, **k): return _StreamResponse(302, headers={"location": "https://example.test/loop"})
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with pytest.raises(DocumentLoadingError, match="Redirect limit"):
        run(loader._fetch_with_fallback("https://example.test/loop"))


def test_web_loader_enforces_streamed_response_limit(monkeypatch):
    import httpx
    import training_data_bot.sources.web as web_module
    limits = replace(web_module.settings.resource_limits, max_remote_bytes=4)
    monkeypatch.setattr(web_module, "settings", replace(web_module.settings, resource_limits=limits))
    loader = WebLoader(use_decodo=False, remote_fetch_policy=RemoteFetchPolicy(enabled=True, allowed_hosts=frozenset({"example.test"})))
    monkeypatch.setattr("training_data_bot.sources.web.socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 443))])
    class Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        def stream(self, *a, **k): return _StreamResponse(200, body=b"12345")
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with pytest.raises(DocumentLoadingError, match="size limit"):
        run(loader._fetch_with_fallback("https://example.test/large"))


def test_pdf_page_limit_rejects_before_extraction(workspace_tmp, monkeypatch):
    if importlib.util.find_spec("fitz") is None or importlib.util.find_spec("pymupdf4llm") is None:
        pytest.skip("PDF dependencies are not installed")
    import fitz
    from dataclasses import replace
    import training_data_bot.sources.pdf as pdf_module

    path = workspace_tmp / "two-pages.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.new_page()
    pdf.save(path)
    pdf.close()
    limits = replace(pdf_module.settings.resource_limits, max_pdf_pages=1)
    monkeypatch.setattr(pdf_module, "settings", replace(pdf_module.settings, resource_limits=limits))
    with pytest.raises(DocumentLoadError, match="page limit"):
        run(PDFLoader().load_single(path))

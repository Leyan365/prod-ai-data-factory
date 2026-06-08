"""Tests for the document preprocessing pipeline."""

import asyncio
from uuid import uuid4

import pytest

from training_data_bot.core.exceptions import PreprocessingError
from training_data_bot.core.models import Document, DocumentType, TextChunk
from training_data_bot.preprocessing import TextPreprocessor


def run(coro):
    return asyncio.run(coro)


def make_document(content: str) -> Document:
    return Document(
        id=uuid4(),
        title="Source Doc",
        content=content,
        source="memory://source",
        doc_type=DocumentType.TXT,
        metadata={"loader": "test", "nested": {"a": 1}},
    )


def test_normalization_collapses_spaces_line_endings_and_control_chars():
    preprocessor = TextPreprocessor()
    text = "  alpha\t\t beta\r\n\r\n\r\n gamma\x00  delta\r epsilon  "

    normalized = preprocessor.normalize_text(text)

    assert normalized == "alpha beta\n\ngamma delta\nepsilon"


def test_normalization_can_remove_blank_lines_when_not_preserving_paragraphs():
    preprocessor = TextPreprocessor(preserve_paragraphs=False)

    normalized = preprocessor.normalize_text(" alpha \n\n\n beta \n")

    assert normalized == "alpha\nbeta"


def test_normalization_handles_empty_and_whitespace_only_text():
    preprocessor = TextPreprocessor()

    assert preprocessor.normalize_text("") == ""
    assert preprocessor.normalize_text(" \n\t  ") == ""
    assert preprocessor.clean_text("  one\t two  ") == "one two"


def test_short_document_produces_single_chunk_with_metadata():
    document = make_document(" short text ")
    original_metadata = dict(document.metadata)
    chunks = run(TextPreprocessor(chunk_size=100, chunk_overlap=10).process_documents(document))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, TextChunk)
    assert chunk.document_id == document.id
    assert chunk.content == "short text"
    assert chunk.chunk_index == 0
    assert chunk.start_index == 0
    assert chunk.end_index == len("short text")
    assert chunk.token_count > 0
    assert chunk.preceding_context is None
    assert chunk.following_context is None
    assert chunk.metadata["source_document_title"] == "Source Doc"
    assert chunk.metadata["source_document_type"] == DocumentType.TXT.value
    assert chunk.metadata["source_document_source"] == "memory://source"
    assert chunk.metadata["source_metadata"] == original_metadata
    assert chunk.metadata["normalized"] is True
    assert chunk.metadata["chunk_size"] == 100
    assert chunk.metadata["chunk_overlap"] == 10
    assert document.metadata == original_metadata


def test_long_document_chunks_in_order_with_content_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = run(TextPreprocessor(chunk_size=10, chunk_overlap=3).process_documents(make_document(text)))

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.start_index for chunk in chunks] == [0, 7, 14, 21]
    assert chunks[0].content[-3:] == chunks[1].content[:3]
    assert chunks[1].content[-3:] == chunks[2].content[:3]
    assert chunks[1].preceding_context == "efg"
    assert chunks[1].following_context == "rst"


def test_zero_overlap_and_min_chunk_chars_filter_tiny_tail():
    document = make_document("abcdefghijx")
    chunks = run(
        TextPreprocessor(chunk_size=5, chunk_overlap=0, min_chunk_chars=2).process_documents(document)
    )

    assert [chunk.content for chunk in chunks] == ["abcde", "fghij"]
    assert all(chunk.preceding_context is None for chunk in chunks)


def test_invalid_chunk_configuration_raises_value_error():
    with pytest.raises(ValueError, match="chunk_size"):
        TextPreprocessor(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        TextPreprocessor(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap"):
        TextPreprocessor(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="min_chunk_chars"):
        TextPreprocessor(min_chunk_chars=-1)


def test_stable_chunk_ids_repeat_for_same_document_and_content():
    document = make_document("abcdefghijklmnopqrstuvwxyz")
    preprocessor = TextPreprocessor(chunk_size=8, chunk_overlap=2)

    first = run(preprocessor.process_documents(document))
    second = run(preprocessor.process_documents(document))

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_stable_chunk_ids_change_for_different_document_or_content():
    preprocessor = TextPreprocessor(chunk_size=100, chunk_overlap=0)
    doc_a = make_document("same content")
    doc_b = make_document("same content")
    doc_c = make_document("changed content")

    chunk_a = run(preprocessor.process_documents(doc_a))[0]
    chunk_b = run(preprocessor.process_documents(doc_b))[0]
    chunk_c = run(preprocessor.process_documents(doc_c))[0]

    assert chunk_a.id != chunk_b.id
    assert chunk_a.id != chunk_c.id


def test_non_string_content_raises_preprocessing_error():
    document = make_document("valid")
    document.content = None

    with pytest.raises(PreprocessingError, match="must be a string"):
        run(TextPreprocessor().process_documents(document))


def test_internal_failures_are_wrapped_in_preprocessing_error(monkeypatch):
    preprocessor = TextPreprocessor()

    def fail(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(preprocessor, "normalize_text", fail)

    with pytest.raises(PreprocessingError) as exc:
        run(preprocessor.process_documents(make_document("content")))

    assert "Failed to preprocess document" in str(exc.value)
    assert isinstance(exc.value.cause, RuntimeError)


def test_empty_document_returns_no_chunks():
    chunks = run(TextPreprocessor().process_documents(make_document(" \n\t ")))

    assert chunks == []


def test_training_bot_async_preprocessor_call_remains_compatible():
    document = make_document("compatibility text")
    preprocessor = TextPreprocessor()

    chunks = run(preprocessor.process_documents(document))

    assert chunks[0].document_id == document.id
    assert chunks[0].content == "compatibility text"

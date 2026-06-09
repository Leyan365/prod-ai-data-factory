"""Tests for core model defaults and serialization compatibility."""

from datetime import timezone
from uuid import uuid4

from training_data_bot.core.models import (
    Dataset,
    Document,
    DocumentType,
    ProcessingJob,
    TaskType,
    TextChunk,
    TrainingExample,
)


def make_example():
    return TrainingExample(
        input_text="input",
        output_text="output text long enough",
        task_type=TaskType.SUMMARIZATION,
        source_document_id=uuid4(),
    )


def test_document_counts_are_calculated_from_content_by_default():
    document = Document(
        title="Counts",
        content="one two three",
        source="memory://counts",
        doc_type=DocumentType.TXT,
    )

    assert document.word_count == 3
    assert document.char_count == len("one two three")


def test_document_explicit_counts_are_preserved():
    document = Document(
        title="Counts",
        content="one two three",
        source="memory://counts",
        doc_type=DocumentType.TXT,
        word_count=99,
        char_count=100,
    )

    assert document.word_count == 99
    assert document.char_count == 100


def test_text_chunk_token_count_is_calculated_by_default():
    chunk = TextChunk(
        document_id=uuid4(),
        content="x" * 20,
        start_index=0,
        end_index=20,
        chunk_index=0,
    )

    assert chunk.token_count == 5


def test_dataset_total_examples_is_calculated_by_default():
    dataset = Dataset(
        name="dataset",
        description="dataset",
        examples=[make_example(), make_example()],
    )

    assert dataset.total_examples == 2


def test_model_timestamps_are_timezone_aware_utc():
    document = Document(
        title="Time",
        content="timestamp check",
        source="memory://time",
        doc_type=DocumentType.TXT,
    )
    job = ProcessingJob(name="job", job_type="test")

    assert document.created_at.tzinfo is not None
    assert document.created_at.utcoffset() == timezone.utc.utcoffset(document.created_at)
    assert job.started_at.tzinfo is not None
    assert job.started_at.utcoffset() == timezone.utc.utcoffset(job.started_at)


def test_model_dump_json_serializes_core_types():
    dataset = Dataset(name="dataset", description="dataset", examples=[make_example()])
    data = dataset.model_dump(mode="json")
    json_text = dataset.model_dump_json()

    assert isinstance(data["id"], str)
    assert isinstance(data["created_at"], str)
    assert data["examples"][0]["task_type"] == "summarization"
    assert '"examples"' in json_text

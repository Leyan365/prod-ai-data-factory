"""Tests for dataset export hardening."""

import asyncio
import csv
import json
from uuid import uuid4

import pytest

from training_data_bot.bot import TrainingDataBot
from training_data_bot.core.exceptions import ExportError, StorageError
from training_data_bot.core.models import (
    Dataset,
    Document,
    DocumentType,
    ExportFormat,
    ProcessingJob,
    ProcessingStatus,
    QualityMetric,
    TaskType,
    TrainingExample,
)
from training_data_bot.storage import DatabaseManager, DatasetExporter


def run(coro):
    return asyncio.run(coro)


def make_example(index, task_type=TaskType.SUMMARIZATION):
    return TrainingExample(
        input_text=f"Input text {index}",
        output_text=f"Output text {index}",
        task_type=task_type,
        source_document_id=uuid4(),
        source_chunk_id=uuid4(),
        template_id=uuid4(),
        quality_scores={QualityMetric.RELEVANCE: 0.9},
        quality_approved=True,
        instruction=f"Instruction {index}",
        context=f"Context {index}",
        category="test",
        metadata={"rank": index, "tags": ["alpha", "beta"]},
    )


def make_dataset(count=5, **kwargs):
    return Dataset(
        name="export test",
        description="dataset export test",
        examples=[make_example(index) for index in range(count)],
        **kwargs,
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_job(**kwargs):
    data = {
        "name": "Process docs",
        "job_type": "document_processing",
        "status": ProcessingStatus.PROCESSING,
        "total_items": 3,
        "processed_items": 1,
        "failed_items": 0,
        "input_data": {"document_count": 1},
        "output_data": {"dataset_id": "pending"},
        "raw_output": {"provider": "mock"},
    }
    data.update(kwargs)
    return ProcessingJob(**data)


def test_jsonl_export_writes_stable_records(workspace_tmp):
    dataset = make_dataset(2)
    output_path = run(
        DatasetExporter().export_dataset(
            dataset,
            workspace_tmp / "examples.jsonl",
            format=ExportFormat.JSONL,
            split_data=False,
        )
    )

    records = read_jsonl(output_path)

    assert output_path == workspace_tmp / "examples.jsonl"
    assert len(records) == 2
    assert records[0]["input_text"] == "Input text 0"
    assert records[0]["task_type"] == "summarization"
    assert records[0]["quality_scores"] == {"relevance": 0.9}
    assert records[0]["metadata"] == {"rank": 0, "tags": ["alpha", "beta"]}
    assert dataset.export_path == output_path
    assert dataset.export_format == ExportFormat.JSONL
    assert dataset.exported_at is not None


def test_csv_export_flattens_nested_fields_predictably(workspace_tmp):
    dataset = make_dataset(1)
    output_path = run(
        DatasetExporter().export_dataset(
            dataset,
            workspace_tmp,
            format=ExportFormat.CSV,
            split_data=False,
        )
    )

    with output_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert output_path == workspace_tmp / "dataset.csv"
    assert rows[0]["input_text"] == "Input text 0"
    assert rows[0]["output_text"] == "Output text 0"
    assert rows[0]["task_type"] == "summarization"
    assert json.loads(rows[0]["quality_scores"]) == {"relevance": 0.9}
    assert json.loads(rows[0]["metadata"]) == {"rank": 0, "tags": ["alpha", "beta"]}


def test_split_export_writes_deterministic_train_validation_test_files(workspace_tmp):
    dataset = make_dataset(
        10,
        train_split=0.6,
        validation_split=0.2,
        test_split=0.2,
    )

    output_dir = run(
        DatasetExporter().export_dataset(
            dataset,
            workspace_tmp / "package.jsonl",
            format=ExportFormat.JSONL,
            split_data=True,
        )
    )

    assert output_dir == workspace_tmp / "package"
    assert [record["input_text"] for record in read_jsonl(output_dir / "train.jsonl")] == [
        f"Input text {index}" for index in range(6)
    ]
    assert [record["input_text"] for record in read_jsonl(output_dir / "validation.jsonl")] == [
        "Input text 6",
        "Input text 7",
    ]
    assert [record["input_text"] for record in read_jsonl(output_dir / "test.jsonl")] == [
        "Input text 8",
        "Input text 9",
    ]


def test_split_export_creates_empty_split_files_for_empty_dataset(workspace_tmp):
    output_dir = run(
        DatasetExporter().export_dataset(
            make_dataset(0),
            workspace_tmp / "empty",
            format=ExportFormat.CSV,
            split_data=True,
        )
    )

    assert (output_dir / "train.csv").read_text(encoding="utf-8").startswith("id,input_text")
    assert (output_dir / "validation.csv").read_text(encoding="utf-8").startswith("id,input_text")
    assert (output_dir / "test.csv").read_text(encoding="utf-8").startswith("id,input_text")


def test_unsupported_export_formats_raise_export_error(workspace_tmp):
    with pytest.raises(ExportError, match="not implemented"):
        run(
            DatasetExporter().export_dataset(
                make_dataset(1),
                workspace_tmp / "dataset.parquet",
                format=ExportFormat.PARQUET,
                split_data=False,
            )
        )


def test_output_suffix_must_match_requested_format(workspace_tmp):
    with pytest.raises(ExportError, match="suffix"):
        run(
            DatasetExporter().export_dataset(
                make_dataset(1),
                workspace_tmp / "dataset.jsonl",
                format=ExportFormat.CSV,
                split_data=False,
            )
        )


def test_split_ratios_must_sum_to_one(workspace_tmp):
    dataset = make_dataset(1, train_split=0.7, validation_split=0.2, test_split=0.2)

    with pytest.raises(ExportError, match="sum to 1.0"):
        run(
            DatasetExporter().export_dataset(
                dataset,
                workspace_tmp / "bad",
                format=ExportFormat.JSONL,
                split_data=True,
            )
        )


def test_training_bot_export_dataset_uses_hardened_exporter(workspace_tmp):
    bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "storage"})
    dataset = make_dataset(1)

    output_path = run(
        bot.export_dataset(
            dataset,
            workspace_tmp / "bot.csv",
            format=ExportFormat.CSV,
            split_data=False,
        )
    )
    run(bot.cleanup())

    assert output_path == workspace_tmp / "bot.csv"
    assert output_path.exists()
    assert dataset.export_path == output_path
    assert dataset.export_format == ExportFormat.CSV


def test_database_manager_saves_and_loads_dataset_records(workspace_tmp):
    manager = DatabaseManager(workspace_tmp / "storage")
    dataset = make_dataset(2)
    dataset.export_path = workspace_tmp / "exports" / "dataset.jsonl"
    dataset.export_format = ExportFormat.JSONL

    saved_path = run(manager.save_dataset(dataset))
    loaded = run(manager.load_dataset(dataset.id))

    assert saved_path == workspace_tmp / "storage" / "datasets" / f"{dataset.id}.json"
    assert loaded.id == dataset.id
    assert loaded.name == dataset.name
    assert len(loaded.examples) == 2
    assert loaded.examples[0].metadata == {"rank": 0, "tags": ["alpha", "beta"]}
    assert loaded.export_path == dataset.export_path
    assert loaded.export_format == ExportFormat.JSONL


def test_database_manager_saves_and_loads_job_records(workspace_tmp):
    manager = DatabaseManager(workspace_tmp / "storage")
    job = make_job(status=ProcessingStatus.FAILED, failed_items=2, error_message="boom")

    saved_path = run(manager.save_job(job))
    loaded = run(manager.load_job(str(job.id)))

    assert saved_path == workspace_tmp / "storage" / "jobs" / f"{job.id}.json"
    assert loaded.id == job.id
    assert loaded.status == ProcessingStatus.FAILED
    assert loaded.failed_items == 2
    assert loaded.error_message == "boom"
    assert loaded.input_data == {"document_count": 1}
    assert loaded.raw_output == {"provider": "mock"}


def test_database_manager_lists_records_in_filename_order(workspace_tmp):
    manager = DatabaseManager(workspace_tmp / "storage")
    datasets = [make_dataset(1) for _ in range(3)]

    for dataset in reversed(datasets):
        run(manager.save_dataset(dataset))

    listed = run(manager.list_datasets())

    assert [str(dataset.id) for dataset in listed] == sorted(str(dataset.id) for dataset in datasets)


def test_database_manager_repeated_save_overwrites_same_record(workspace_tmp):
    manager = DatabaseManager(workspace_tmp / "storage")
    dataset = make_dataset(1)

    first_path = run(manager.save_dataset(dataset))
    first_text = first_path.read_text(encoding="utf-8")
    dataset.description = "updated description"
    second_path = run(manager.save_dataset(dataset))
    second_text = second_path.read_text(encoding="utf-8")

    assert first_path == second_path
    assert first_text != second_text
    assert run(manager.load_dataset(dataset.id)).description == "updated description"


def test_database_manager_missing_and_malformed_records_raise_storage_error(workspace_tmp):
    manager = DatabaseManager(workspace_tmp / "storage")

    with pytest.raises(StorageError, match="not found"):
        run(manager.load_dataset(uuid4()))

    malformed_path = workspace_tmp / "storage" / "datasets" / "malformed.json"
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(StorageError, match="malformed"):
        run(manager.load_dataset("malformed"))


class StaticProvider:
    async def generate(self, prompt, *, timeout=None):
        from training_data_bot.ai import AIResponse

        return AIResponse(text="The stored output explains the source document.", token_usage=1)

    async def close(self):
        return None


def make_document():
    return Document(
        title="Storage Doc",
        content="Storage persistence keeps generated datasets and jobs durable.",
        source="memory://storage",
        doc_type=DocumentType.TXT,
    )


def test_training_bot_process_documents_persists_dataset_and_completed_job(workspace_tmp):
    from training_data_bot.ai import AIClient

    bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "storage"})
    bot.ai_client = AIClient(provider=StaticProvider(), max_retries=0)

    dataset = run(
        bot.process_documents(
            documents=[make_document()],
            task_types=[TaskType.SUMMARIZATION],
            quality_filter=False,
        )
    )
    persisted_datasets = run(bot.list_persisted_datasets())
    persisted_jobs = run(bot.list_persisted_jobs())
    run(bot.cleanup())

    assert [stored.id for stored in persisted_datasets] == [dataset.id]
    assert len(persisted_jobs) == 1
    assert persisted_jobs[0].status == ProcessingStatus.COMPLETED
    assert persisted_jobs[0].output_data["dataset_id"] == str(dataset.id)


def test_training_bot_persists_failed_job_state(workspace_tmp):
    class FailingPreprocessor:
        async def process_documents(self, document):
            raise RuntimeError("preprocessing failed")

    bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "storage"})
    bot.preprocessor = FailingPreprocessor()

    with pytest.raises(RuntimeError, match="preprocessing failed"):
        run(
            bot.process_documents(
                documents=[make_document()],
                task_types=[TaskType.SUMMARIZATION],
                quality_filter=False,
            )
        )

    persisted_jobs = run(bot.list_persisted_jobs())
    run(bot.cleanup())

    assert len(persisted_jobs) == 1
    assert persisted_jobs[0].status == ProcessingStatus.FAILED
    assert persisted_jobs[0].error_message == "preprocessing failed"


def test_training_bot_export_dataset_persists_export_metadata(workspace_tmp):
    bot = TrainingDataBot(config={"storage_dir": workspace_tmp / "storage"})
    dataset = make_dataset(1)

    output_path = run(
        bot.export_dataset(
            dataset,
            workspace_tmp / "exports" / "dataset.jsonl",
            format=ExportFormat.JSONL,
            split_data=False,
        )
    )
    loaded = run(bot.load_dataset(dataset.id))
    run(bot.cleanup())

    assert output_path == workspace_tmp / "exports" / "dataset.jsonl"
    assert loaded.export_path == output_path
    assert loaded.export_format == ExportFormat.JSONL
    assert loaded.id in bot.datasets

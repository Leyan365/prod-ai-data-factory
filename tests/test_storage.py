"""Tests for dataset export hardening."""

import asyncio
import csv
import json
from uuid import uuid4

import pytest

from training_data_bot.bot import TrainingDataBot
from training_data_bot.core.exceptions import ExportError
from training_data_bot.core.models import Dataset, ExportFormat, QualityMetric, TaskType, TrainingExample
from training_data_bot.storage import DatasetExporter


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
    bot = TrainingDataBot()
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

"""
Dataset export and storage placeholders.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import UUID

from .core.config import settings
from .core.exceptions import ExportError, StorageError
from .core.models import Dataset, ExportFormat, ProcessingJob, TrainingExample


def _model_to_record(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        data = model.model_dump(mode="json")
    else:
        data = model.dict()
    return _jsonable(data)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (datetime, Path, UUID)):
        return str(value)
    return value


class DatasetExporter:
    """Export datasets to deterministic local formats."""

    SUPPORTED_FORMATS = {ExportFormat.JSONL, ExportFormat.CSV}
    CSV_FIELDS = [
        "id",
        "input_text",
        "output_text",
        "task_type",
        "source_document_id",
        "source_chunk_id",
        "template_id",
        "quality_scores",
        "quality_approved",
        "instruction",
        "context",
        "category",
        "metadata",
        "created_at",
        "updated_at",
    ]

    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL,
        split_data: bool = True,
        **kwargs,
    ) -> Path:
        export_format = self._validate_format(format)
        self._validate_dataset(dataset)

        if split_data:
            output_dir = self._resolve_split_output_dir(Path(output_path), export_format)
            output_dir.mkdir(parents=True, exist_ok=True)
            splits = self._split_examples(dataset)
            for split_name, examples in splits:
                split_path = output_dir / f"{split_name}.{export_format.value}"
                self._write_examples(split_path, examples, export_format)
            dataset.export_path = output_dir
            dataset.export_format = export_format
            dataset.exported_at = datetime.now(timezone.utc)
            return output_dir

        resolved_path = self._resolve_single_output_path(Path(output_path), export_format)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_examples(resolved_path, dataset.examples, export_format)
        dataset.export_path = resolved_path
        dataset.export_format = export_format
        dataset.exported_at = datetime.now(timezone.utc)
        return resolved_path

    def _validate_format(self, format: ExportFormat) -> ExportFormat:
        try:
            export_format = format if isinstance(format, ExportFormat) else ExportFormat(format)
        except ValueError as exc:
            supported = ", ".join(sorted(item.value for item in self.SUPPORTED_FORMATS))
            raise ExportError(
                f"Unsupported export format '{format}'. Supported formats: {supported}",
            ) from exc

        if export_format not in self.SUPPORTED_FORMATS:
            supported = ", ".join(sorted(item.value for item in self.SUPPORTED_FORMATS))
            raise ExportError(
                f"Export format '{export_format.value}' is not implemented. Supported formats: {supported}",
            )
        return export_format

    def _validate_dataset(self, dataset: Dataset) -> None:
        if not isinstance(dataset, Dataset):
            raise ExportError("dataset must be a Dataset instance")
        if dataset.examples is None:
            raise ExportError("dataset.examples must be a list")
        for index, example in enumerate(dataset.examples):
            if not isinstance(example, TrainingExample):
                raise ExportError(f"dataset.examples[{index}] must be a TrainingExample instance")

    def _resolve_single_output_path(self, output_path: Path, format: ExportFormat) -> Path:
        suffix = f".{format.value}"
        if output_path.exists() and output_path.is_dir():
            return output_path / f"dataset{suffix}"
        if output_path.suffix:
            self._validate_suffix(output_path, format)
            return output_path
        return output_path / f"dataset{suffix}"

    def _resolve_split_output_dir(self, output_path: Path, format: ExportFormat) -> Path:
        if output_path.exists() and output_path.is_file():
            raise ExportError("split exports require an output directory, not an existing file")
        if output_path.suffix:
            self._validate_suffix(output_path, format)
            return output_path.with_suffix("")
        return output_path

    def _validate_suffix(self, output_path: Path, format: ExportFormat) -> None:
        expected = f".{format.value}"
        if output_path.suffix.lower() != expected:
            raise ExportError(
                f"Output path suffix '{output_path.suffix}' does not match export format '{format.value}'"
            )

    def _split_examples(self, dataset: Dataset) -> List[Tuple[str, Sequence[TrainingExample]]]:
        ratios = {
            "train": dataset.train_split,
            "validation": dataset.validation_split,
            "test": dataset.test_split,
        }
        for name, ratio in ratios.items():
            if ratio < 0 or ratio > 1:
                raise ExportError(f"{name}_split must be between 0 and 1")

        ratio_total = sum(ratios.values())
        if abs(ratio_total - 1.0) > 0.000001:
            raise ExportError("train_split, validation_split, and test_split must sum to 1.0")

        examples = list(dataset.examples)
        total = len(examples)
        train_count = int(total * dataset.train_split)
        validation_count = int(total * dataset.validation_split)
        test_start = train_count + validation_count

        return [
            ("train", examples[:train_count]),
            ("validation", examples[train_count:test_start]),
            ("test", examples[test_start:]),
        ]

    def _write_examples(
        self,
        output_path: Path,
        examples: Iterable[TrainingExample],
        format: ExportFormat,
    ) -> None:
        if format == ExportFormat.JSONL:
            self._write_jsonl(output_path, examples)
        elif format == ExportFormat.CSV:
            self._write_csv(output_path, list(examples))
        else:
            raise ExportError(f"Export format '{format.value}' is not implemented")

    def _write_jsonl(self, output_path: Path, examples: Iterable[TrainingExample]) -> None:
        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            for example in examples:
                f.write(json.dumps(self._example_to_record(example), ensure_ascii=False, sort_keys=True))
                f.write("\n")

    def _write_csv(self, output_path: Path, examples: List[TrainingExample]) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for example in examples:
                writer.writerow(self._example_to_csv_row(example))

    def _example_to_record(self, example: TrainingExample) -> Dict[str, Any]:
        return _model_to_record(example)

    def _example_to_csv_row(self, example: TrainingExample) -> Dict[str, Any]:
        record = self._example_to_record(example)
        row = {}
        for field in self.CSV_FIELDS:
            value = record.get(field)
            if isinstance(value, (dict, list)):
                row[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            elif value is None:
                row[field] = ""
            else:
                row[field] = value
        return row

    def _jsonable(self, value: Any) -> Any:
        return _jsonable(value)


class DatabaseManager:
    """Local JSON persistence for datasets and processing jobs."""

    def __init__(self, storage_dir: Path | str | None = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else settings.output_dir / "storage"
        self.datasets_dir = self.storage_dir / "datasets"
        self.jobs_dir = self.storage_dir / "jobs"

    async def save_dataset(self, dataset: Dataset) -> Path:
        if not isinstance(dataset, Dataset):
            raise StorageError("dataset must be a Dataset instance")
        return self._write_record(self.datasets_dir / f"{dataset.id}.json", _model_to_record(dataset))

    async def load_dataset(self, dataset_id: UUID | str) -> Dataset:
        path = self._entity_path(self.datasets_dir, dataset_id)
        return self._load_model(path, Dataset, "dataset")

    async def list_datasets(self) -> List[Dataset]:
        return [self._load_model(path, Dataset, "dataset") for path in self._iter_records(self.datasets_dir)]

    async def save_job(self, job: ProcessingJob) -> Path:
        if not isinstance(job, ProcessingJob):
            raise StorageError("job must be a ProcessingJob instance")
        return self._write_record(self.jobs_dir / f"{job.id}.json", _model_to_record(job))

    async def load_job(self, job_id: UUID | str) -> ProcessingJob:
        path = self._entity_path(self.jobs_dir, job_id)
        return self._load_model(path, ProcessingJob, "job")

    async def list_jobs(self) -> List[ProcessingJob]:
        return [self._load_model(path, ProcessingJob, "job") for path in self._iter_records(self.jobs_dir)]

    async def close(self) -> None:
        return None

    def _entity_path(self, base_dir: Path, entity_id: UUID | str) -> Path:
        entity_text = str(entity_id)
        path = base_dir / f"{entity_text}.json"
        if not path.exists():
            raise StorageError(f"Stored record not found: {entity_text}")
        return path

    def _iter_records(self, base_dir: Path) -> List[Path]:
        if not base_dir.exists():
            return []
        if not base_dir.is_dir():
            raise StorageError(f"Storage path is not a directory: {base_dir}")
        return sorted(base_dir.glob("*.json"), key=lambda path: path.name)

    def _write_record(self, path: Path, record: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            try:
                temp_path.replace(path)
            except PermissionError:
                with path.open("w", encoding="utf-8", newline="\n") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        except OSError as exc:
            raise StorageError(f"Failed to write storage record: {path}", cause=exc) from exc
        return path

    def _load_model(self, path: Path, model_type: Any, label: str):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise StorageError(f"Stored {label} record is malformed: {path}", cause=exc) from exc
        except OSError as exc:
            raise StorageError(f"Failed to read stored {label} record: {path}", cause=exc) from exc

        if not isinstance(data, dict):
            raise StorageError(f"Stored {label} record must be a JSON object: {path}")

        try:
            if hasattr(model_type, "model_validate"):
                return model_type.model_validate(data)
            return model_type(**data)
        except Exception as exc:
            raise StorageError(f"Stored {label} record is invalid: {path}", cause=exc) from exc

"""
Dataset export and storage placeholders.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import UUID

from .core.exceptions import ExportError
from .core.models import Dataset, ExportFormat, TrainingExample


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
        if hasattr(example, "model_dump"):
            data = example.model_dump(mode="json")
        else:
            data = example.dict()
        return self._jsonable(data)

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
        if isinstance(value, dict):
            return {str(self._jsonable(key)): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, (datetime, Path, UUID)):
            return str(value)
        return value


class DatabaseManager:
    """Placeholder database manager for lifecycle compatibility."""

    async def close(self) -> None:
        return None

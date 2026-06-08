"""
Dataset export and storage placeholders.
"""

import csv
import json
from pathlib import Path
from typing import Iterable, List

from .core.models import Dataset, ExportFormat, TrainingExample


class DatasetExporter:
    """Export datasets to simple local formats."""

    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: Path,
        format: ExportFormat = ExportFormat.JSONL,
        split_data: bool = True,
        **kwargs,
    ) -> Path:
        output_path = self._resolve_output_path(output_path, format)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ExportFormat.JSONL:
            self._write_jsonl(output_path, dataset.examples)
        elif format == ExportFormat.CSV:
            self._write_csv(output_path, dataset.examples)
        else:
            raise ValueError(f"Export format '{format}' is not implemented in this tutorial slice")

        return output_path

    def _resolve_output_path(self, output_path: Path, format: ExportFormat) -> Path:
        if output_path.suffix:
            return output_path
        return output_path / f"dataset.{format.value}"

    def _write_jsonl(self, output_path: Path, examples: Iterable[TrainingExample]) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(example.json())
                f.write("\n")

    def _write_csv(self, output_path: Path, examples: List[TrainingExample]) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input_text", "output_text", "task_type"])
            writer.writeheader()
            for example in examples:
                writer.writerow(
                    {
                        "input_text": example.input_text,
                        "output_text": example.output_text,
                        "task_type": example.task_type.value if hasattr(example.task_type, "value") else example.task_type,
                    }
                )


class DatabaseManager:
    """Placeholder database manager for lifecycle compatibility."""

    async def close(self) -> None:
        return None

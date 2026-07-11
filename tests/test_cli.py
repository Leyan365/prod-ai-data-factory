"""Tests for the tutorial command line interface."""

import json
import re

from training_data_bot.cli import main
from training_data_bot.core.models import Dataset


def run_cli(args):
    return main(args)


def write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


def dataset_id_from(output):
    match = re.search(r"dataset_id: ([0-9a-f-]+)", output)
    assert match, output
    return match.group(1)


def test_cli_version_commands_print_package_version(capsys):
    assert run_cli(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"

    assert run_cli(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_cli_status_reports_offline_defaults(workspace_tmp, capsys):
    storage_dir = workspace_tmp / "storage"

    assert run_cli(["status", "--storage-dir", str(storage_dir)]) == 0

    output = capsys.readouterr().out
    assert "training-data-bot 0.1.0" in output
    assert "default_provider: mock" in output
    assert "supported_exports: csv, jsonl" in output
    assert "persisted_datasets: 0" in output
    assert "persisted_jobs: 0" in output


def test_cli_smoke_uses_test_local_storage(workspace_tmp, capsys):
    storage_dir = workspace_tmp / "storage"

    assert run_cli(["smoke", "--storage-dir", str(storage_dir)]) == 0

    output = capsys.readouterr().out
    assert "smoke ok" in output
    assert str(storage_dir) in output
    assert (storage_dir / "_smoke" / "sample.txt").exists()


def test_cli_process_file_writes_jsonl_and_persists_dataset(workspace_tmp, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    storage_dir = workspace_tmp / "storage"
    source = write(
        workspace_tmp / "sample.txt",
        "The tutorial source explains local training data generation.",
    )
    output_path = workspace_tmp / "dataset.jsonl"

    code = run_cli(
        [
            "process",
            str(source),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
            "--storage-dir",
            str(storage_dir),
            "--no-quality-filter",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "process ok" in output
    assert "provider: mock" in output
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])["output_text"].startswith(
        "Mock response:"
    )
    dataset_id = dataset_id_from(output)
    assert (storage_dir / "datasets" / f"{dataset_id}.json").exists()


def test_cli_process_directory_writes_csv(workspace_tmp, capsys):
    storage_dir = workspace_tmp / "storage"
    source_dir = workspace_tmp / "sources"
    source_dir.mkdir()
    write(source_dir / "b.txt", "Second local tutorial document.")
    write(source_dir / "a.txt", "First local tutorial document.")
    output_path = workspace_tmp / "dataset.csv"

    code = run_cli(
        [
            "process",
            str(source_dir),
            "--output",
            str(output_path),
            "--format",
            "csv",
            "--storage-dir",
            str(storage_dir),
            "--task",
            "summarization",
            "--no-quality-filter",
        ]
    )

    assert code == 0
    assert "process ok" in capsys.readouterr().out
    csv_text = output_path.read_text(encoding="utf-8")
    assert csv_text.startswith("id,input_text,output_text")
    assert "Mock response:" in csv_text


def test_cli_export_reexports_persisted_dataset(workspace_tmp, capsys):
    storage_dir = workspace_tmp / "storage"
    source = write(
        workspace_tmp / "sample.txt",
        "Dataset export uses persisted local JSON records.",
    )
    process_output = workspace_tmp / "dataset.jsonl"

    assert run_cli(
        [
            "process",
            str(source),
            "--output",
            str(process_output),
            "--format",
            "jsonl",
            "--storage-dir",
            str(storage_dir),
            "--task",
            "summarization",
            "--no-quality-filter",
        ]
    ) == 0
    dataset_id = dataset_id_from(capsys.readouterr().out)

    export_output = workspace_tmp / "export.csv"
    assert run_cli(
        [
            "export",
            dataset_id,
            "--output",
            str(export_output),
            "--format",
            "csv",
            "--storage-dir",
            str(storage_dir),
            "--no-split",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "export ok" in output
    assert dataset_id in output
    assert export_output.exists()
    assert export_output.read_text(encoding="utf-8").startswith("id,input_text,output_text")


def test_cli_rejects_placeholder_export_format(workspace_tmp, capsys):
    code = run_cli(
        [
            "process",
            str(workspace_tmp),
            "--output",
            str(workspace_tmp / "dataset.parquet"),
            "--format",
            "parquet",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "invalid choice" in captured.err


def test_cli_gemini_requires_environment_key(workspace_tmp, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    source = write(workspace_tmp / "sample.txt", "Gemini should not start without a key.")

    code = run_cli(
        [
            "process",
            str(source),
            "--output",
            str(workspace_tmp / "dataset.jsonl"),
            "--format",
            "jsonl",
            "--provider",
            "gemini",
            "--storage-dir",
            str(workspace_tmp / "storage"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "GEMINI_API_KEY is required" in captured.err


def test_cli_default_process_does_not_require_gemini_key(workspace_tmp, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    source = write(workspace_tmp / "sample.txt", "Mock provider remains the default.")

    code = run_cli(
        [
            "process",
            str(source),
            "--output",
            str(workspace_tmp / "dataset.jsonl"),
            "--format",
            "jsonl",
            "--storage-dir",
            str(workspace_tmp / "storage"),
            "--task",
            "summarization",
            "--no-quality-filter",
        ]
    )

    assert code == 0
    assert "provider: mock" in capsys.readouterr().out


def test_cli_partial_processing_reports_ids_and_nonzero_status(workspace_tmp, capsys, monkeypatch):
    source = write(workspace_tmp / "source.txt", "partial processing source")
    output = workspace_tmp / "partial.jsonl"

    class PartialBot:
        async def load_documents(self, path): return [object()]
        async def process_documents(self, **kwargs):
            return Dataset(name="partial", description="partial", metadata={"processing_status": "partial"})
        async def export_dataset(self, dataset, output_path, **kwargs):
            output_path.write_text("", encoding="utf-8")
            return output_path
        async def cleanup(self): return None
        class db_manager:
            storage_dir = workspace_tmp / "storage"

    monkeypatch.setattr("training_data_bot.cli._create_bot", lambda storage_dir: PartialBot())
    code = run_cli(["process", str(source), "--output", str(output), "--format", "jsonl", "--storage-dir", str(workspace_tmp / "storage")])
    captured = capsys.readouterr().out
    assert code == 1
    assert "process partial" in captured
    assert "dataset_id:" in captured

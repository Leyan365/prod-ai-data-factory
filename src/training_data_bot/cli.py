"""Command line interface for tutorial/demo workflows."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse

from . import __version__
from .ai import AIClient
from .bot import TrainingDataBot
from .core.config import settings
from .core.exceptions import TrainingDataBotError
from .core.models import ExportFormat, TaskType
from .storage import DatasetExporter


SUPPORTED_CLI_FORMATS = tuple(sorted(item.value for item in DatasetExporter.SUPPORTED_FORMATS))
SUPPORTED_PROVIDERS = ("mock", "gemini")


class CLIUsageError(Exception):
    """Raised when arguments are valid syntactically but unsupported for the CLI."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training-data-bot",
        description="Offline-first Training Data Bot tutorial CLI.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        dest="show_version",
        help="Show the package version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Show the package version.")

    status_parser = subparsers.add_parser("status", help="Show package and local storage status.")
    _add_storage_arg(status_parser)

    smoke_parser = subparsers.add_parser("smoke", help="Run an offline smoke check.")
    _add_storage_arg(smoke_parser)

    process_parser = subparsers.add_parser(
        "process",
        help="Process a local file or directory and export a dataset.",
    )
    process_parser.add_argument("source", help="Local source file or directory to process.")
    process_parser.add_argument("--output", required=True, help="Output path for the export.")
    process_parser.add_argument("--format", choices=SUPPORTED_CLI_FORMATS, required=True)
    process_parser.add_argument(
        "--task",
        choices=[task.value for task in TaskType],
        nargs="+",
        help="Task type values to run. Omit to use the Python API defaults.",
    )
    process_parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="mock")
    process_parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write train/validation/test split files. Defaults to --no-split.",
    )
    process_parser.add_argument(
        "--quality-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply deterministic quality filtering. Defaults to --quality-filter.",
    )
    _add_storage_arg(process_parser)

    export_parser = subparsers.add_parser(
        "export",
        help="Export a persisted dataset by ID.",
    )
    export_parser.add_argument("dataset_id", help="Persisted dataset UUID.")
    export_parser.add_argument("--output", required=True, help="Output path for the export.")
    export_parser.add_argument("--format", choices=SUPPORTED_CLI_FORMATS, required=True)
    export_parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write train/validation/test split files. Defaults to --no-split.",
    )
    _add_storage_arg(export_parser)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.show_version or args.command == "version":
        print(__version__)
        return 0

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        return asyncio.run(async_main(args))
    except CLIUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TrainingDataBotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "status":
        return await _handle_status(args)
    if args.command == "smoke":
        return await _handle_smoke(args)
    if args.command == "process":
        return await _handle_process(args)
    if args.command == "export":
        return await _handle_export(args)
    raise CLIUsageError(f"Unknown command: {args.command}")


async def _handle_status(args: argparse.Namespace) -> int:
    bot = _create_bot(args.storage_dir)
    try:
        datasets = await bot.list_persisted_datasets()
        jobs = await bot.list_persisted_jobs()
        print(f"training-data-bot {__version__}")
        print(f"default_provider: {settings.default_ai_provider}")
        print(f"supported_exports: {', '.join(SUPPORTED_CLI_FORMATS)}")
        print(f"storage_dir: {bot.db_manager.storage_dir}")
        print(f"persisted_datasets: {len(datasets)}")
        print(f"persisted_jobs: {len(jobs)}")
        return 0
    finally:
        await bot.cleanup()


async def _handle_smoke(args: argparse.Namespace) -> int:
    bot = _create_bot(args.storage_dir)
    try:
        sample_dir = _smoke_dir(args.storage_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample = sample_dir / "sample.txt"
        sample.write_text("This is a small tutorial smoke test document.", encoding="utf-8")

        before = bot.get_statistics()
        documents = await bot.load_documents(sample)
        after = bot.get_statistics()

        if len(documents) != 1:
            raise TrainingDataBotError("Smoke check did not load exactly one document")
        if before["documents"]["total"] != 0 or after["documents"]["total"] != 1:
            raise TrainingDataBotError("Smoke check statistics did not update as expected")

        print("smoke ok")
        print(f"storage_dir: {bot.db_manager.storage_dir}")
        return 0
    finally:
        await bot.cleanup()


async def _handle_process(args: argparse.Namespace) -> int:
    source = Path(args.source)
    _validate_local_source(args.source, source)

    bot = _create_bot(args.storage_dir)
    try:
        _configure_provider(bot, args.provider)
        task_types = _parse_task_types(args.task)
        documents = await bot.load_documents(source)
        dataset = await bot.process_documents(
            documents=documents,
            task_types=task_types,
            quality_filter=args.quality_filter,
        )
        export_path = await bot.export_dataset(
            dataset,
            Path(args.output),
            format=ExportFormat(args.format),
            split_data=args.split,
        )

        print("process ok")
        print(f"dataset_id: {dataset.id}")
        print(f"examples: {len(dataset.examples)}")
        print(f"export_path: {export_path}")
        print(f"provider: {args.provider}")
        print(f"storage_dir: {bot.db_manager.storage_dir}")
        return 0
    finally:
        await bot.cleanup()


async def _handle_export(args: argparse.Namespace) -> int:
    bot = _create_bot(args.storage_dir)
    try:
        dataset = await bot.load_dataset(args.dataset_id)
        export_path = await bot.export_dataset(
            dataset,
            Path(args.output),
            format=ExportFormat(args.format),
            split_data=args.split,
        )

        print("export ok")
        print(f"dataset_id: {dataset.id}")
        print(f"examples: {len(dataset.examples)}")
        print(f"export_path: {export_path}")
        print(f"storage_dir: {bot.db_manager.storage_dir}")
        return 0
    finally:
        await bot.cleanup()


def _add_storage_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Local JSON storage directory. Defaults to output/storage.",
    )


def _create_bot(storage_dir: Optional[str]) -> TrainingDataBot:
    config = {"storage_dir": Path(storage_dir)} if storage_dir else {}
    return TrainingDataBot(config=config)


def _configure_provider(bot: TrainingDataBot, provider_name: str) -> None:
    if provider_name == "mock":
        bot.ai_client = AIClient.from_env("mock")
        return

    if provider_name == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise CLIUsageError(
                "GEMINI_API_KEY is required when using --provider gemini"
            )
        bot.ai_client = AIClient.from_env("gemini")
        return

    raise CLIUsageError(f"Unsupported provider: {provider_name}")


def _parse_task_types(task_values: Optional[Iterable[str]]) -> Optional[list[TaskType]]:
    if task_values is None:
        return None
    return [TaskType(value) for value in task_values]


def _validate_local_source(raw_source: str, source: Path) -> None:
    parsed = urlparse(raw_source)
    if parsed.scheme in {"http", "https"}:
        raise CLIUsageError(
            "The Phase 10 CLI supports local files and directories only"
        )
    if not source.exists():
        raise CLIUsageError(f"Source path does not exist: {source}")


def _smoke_dir(storage_dir: Optional[str]) -> Path:
    if storage_dir:
        return Path(storage_dir) / "_smoke"
    return settings.output_dir / "cli_smoke"


if __name__ == "__main__":
    raise SystemExit(main())

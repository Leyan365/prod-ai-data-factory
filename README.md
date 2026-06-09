# Production-Grade Data Factory

Training Data Bot is a tutorial-driven Python package for turning source documents into training data examples. The current implementation covers the local, offline baseline through Phase 8: package repair, loader hardening, preprocessing, task template execution, mock/Gemini AI provider plumbing, deterministic quality evaluation, hardened dataset export, durable local storage, and refreshed project documentation.

## Current Architecture

The package is organized around a local data flow:

1. `UnifiedLoader` loads files, directories, or URLs into `Document` models.
2. `TextPreprocessor` normalizes document text and creates stable overlapping `TextChunk` records.
3. `TaskManager` renders task templates and calls `AIClient`.
4. `AIClient` uses the offline `MockAIProvider` by default, or an opt-in Gemini provider.
5. `QualityEvaluator` scores generated examples with deterministic rule-based checks.
6. `DatasetExporter` writes deterministic JSONL/CSV exports and optional train/validation/test split packages.
7. `DatabaseManager` persists datasets and processing jobs as local JSON under `output/storage`.

The main public entry point is `TrainingDataBot` from `training_data_bot`.

## Features Through Phase 8

- Importable `training_data_bot` package with repaired public exports.
- Core Pydantic models for documents, chunks, task templates, task results, datasets, quality reports, and processing jobs.
- File and URL loading for TXT, MD, HTML, JSON, CSV, DOCX, PDF, and web fallback paths.
- Deterministic directory loading with recursive/non-recursive discovery and glob inclusion patterns.
- Optional dependency errors for DOCX, PDF, and web loaders that name the missing packages.
- Configurable text normalization and character chunking with overlap.
- Stable UUIDv5 chunk identifiers.
- Metadata preservation from documents into chunks and training examples.
- Template-driven task execution for the current `TaskType` values.
- Offline mock AI provider for tests and local use.
- Gemini provider design using `GEMINI_API_KEY` from the environment only.
- Rule-based quality checks for relevance, coherence, diversity, bias, and toxicity.
- Hardened JSONL and CSV dataset export with deterministic split handling.
- Durable local JSON persistence for datasets and processing jobs.
- Smoke script and pytest regression suite.

## Installation

Use the project from the repository root. The user environment for this project has been run with:

```powershell
C:\Users\USER\anaconda3\python.exe -m pip install -r requirements.txt
```

The dependency file includes:

- `pydantic`
- `pytest`
- `httpx`
- `beautifulsoup4`
- `python-docx`
- `pymupdf`
- `pymupdf4llm`

## Running Smoke Checks

```powershell
C:\Users\USER\anaconda3\python.exe scripts\smoke_check.py
```

Expected result:

```text
smoke ok
```

## Running Tests

```powershell
C:\Users\USER\anaconda3\python.exe -m pytest -q
```

Latest Phase 9 result:

```text
76 passed, 3 skipped
```

The skipped tests are optional dependency success-path checks when the environment does not provide the matching loader capability.

## Local Storage

`DatabaseManager` persists runtime records to local JSON files by default:

```text
output/storage/datasets/{dataset_id}.json
output/storage/jobs/{job_id}.json
```

Storage is offline-first, deterministic, and dependency-free. It supports saving, loading, and listing datasets and processing jobs. It does not provide indexing, concurrent write coordination, schema migrations, cloud storage, or distributed storage.

## Environment Variables

The default runtime is offline and does not require credentials.

Optional environment variables:

- `GEMINI_API_KEY`: Gemini API key used only when explicitly creating an AI client/provider configured for Gemini.

## Gemini Configuration

Do not hardcode API keys and do not paste keys into chat. The Gemini provider reads credentials only from the environment:

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Then configure the AI client in code with the Gemini provider path, for example:

```python
from training_data_bot.ai import AIClient

client = AIClient.from_env("gemini")
```

Without explicit Gemini configuration, `AIClient()` uses `MockAIProvider` and performs no network calls.

## Remaining Placeholders And Debt

- Decodo integration remains stubbed/fallback-only, not production scraping behavior.
- `ExportFormat.PARQUET`, `ExportFormat.HUGGINGFACE`, and `ExportFormat.OPENAI` are enum placeholders.
- `TextChunk.embeddings` and `TextChunk.topics` fields exist, but no embeddings or topic extraction pipeline is implemented.
- A fallback local `BaseModel` remains for no-Pydantic environments and may drift from Pydantic behavior.
- No CLI entry point or formal package metadata is implemented yet.

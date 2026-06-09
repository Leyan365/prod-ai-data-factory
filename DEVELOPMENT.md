# Development

This document describes the current workflow for extending the tutorial implementation without changing unrelated behavior.

## Development Workflow

1. Inspect the current layer before editing.
2. Keep each phase scoped to the requested subsystem.
3. Avoid changing public imports unless the phase explicitly requires it.
4. Preserve `TrainingDataBot` compatibility.
5. Prefer deterministic, offline behavior for tests.
6. Do not hardcode secrets or credentials.
7. Keep default AI behavior offline/mock unless a task explicitly changes provider plumbing.

Useful status command:

```powershell
git status --short
```

## Test Workflow

Run the full regression suite:

```powershell
C:\Users\USER\anaconda3\python.exe -m pytest -q
```

Current expected result after Phase 9:

```text
76 passed, 3 skipped
```

Run the smoke check:

```powershell
C:\Users\USER\anaconda3\python.exe scripts\smoke_check.py
```

Run a read-only syntax parse check:

```powershell
C:\Users\USER\anaconda3\python.exe -c "import ast, pathlib; paths=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in paths]; print(f'parsed {len(paths)} python files')"
```

Use `ast.parse` instead of `compileall` when `__pycache__` permissions prevent bytecode writes.

Current expected syntax result:

```text
parsed 29 python files
```

## Commit Strategy

- Commit one phase at a time.
- Stage only files in the requested phase scope.
- Do not stage tutorial PDFs, extracted tutorial text, temporary files, `__pycache__`, or user-owned notes.
- Use clear phase commit messages, for example:

```text
Phase 5: add quality evaluation layer
```

Recent phase commits:

- Phase 6 export hardening: `a9cfaff`
- Phase 7 durable local storage: `6e3619e`
- Cleanup after Phase 7: `8cbc110`
- Phase 8 documentation refresh: `e394986`

## Export And Storage Development

`DatasetExporter` and `DatabaseManager` live in `src/training_data_bot/storage.py`.

Current export behavior:

- JSONL and CSV are the supported first-class formats.
- Split exports write deterministic `train`, `validation`, and `test` files.
- Unsupported placeholder formats should raise clear `ExportError` messages.
- Export should remain offline and deterministic.

Current local storage behavior:

- Datasets are stored under `output/storage/datasets/{dataset_id}.json`.
- Processing jobs are stored under `output/storage/jobs/{job_id}.json`.
- Tests should use a workspace-local temporary `storage_dir` to avoid creating default `output/storage` artifacts.
- Storage should remain local JSON only unless a later production-hardening phase explicitly changes scope.

Placeholder export enum values:

- `parquet`
- `huggingface`
- `openai`

These are intentionally not implemented yet.

## Recommended Next Phase

Recommended Phase 10: packaging, CLI, and project metadata cleanup.

Suggested scope:

1. Add formal package metadata and editable-install support.
2. Add a minimal offline-first CLI for smoke-friendly local workflows.
3. Preserve current model field names and `TrainingDataBot` workflows.
4. Keep default AI behavior offline/mock.

Out of scope for the packaging and CLI phase:

- Vector databases
- Embeddings pipeline
- Production cloud storage
- Decodo production integration
- New AI providers
- Export format expansion

## Add A New Task Template

1. Add or confirm the matching `TaskType` in `core.models`.
2. Add a `TaskTemplate` in `TaskManager.default_templates()`.
3. Include a concise prompt with only supported variables.
4. If new variables are needed, update the rendering context deliberately.
5. Add tests in `tests/test_tasks.py`.

Supported template variables currently include:

- `content`
- `chunk_index`
- `document_id`
- `source_document_title`
- `source_document_type`
- `source_document_source`

Missing variables should raise `TemplateRenderError`.

## Add A New AI Provider

1. Implement the provider protocol from `ai.py`.
2. Return `AIResponse` from `generate()`.
3. Add `close()` even if it is a no-op.
4. Read credentials only from environment variables.
5. Do not accept API keys through chat, source code, tests, or config defaults.
6. Register the provider in `AIClient.from_env()` if it should be selectable by name.
7. Add offline tests with fake clients/transports and no real network calls.

The Gemini provider is the current reference implementation. It reads only `GEMINI_API_KEY` from `os.environ`.

## Add A New Loader

1. Add the loader implementation under `src/training_data_bot/sources`.
2. Return `Document` models with accurate `title`, `content`, `source`, `doc_type`, size, and extraction metadata.
3. Avoid eager imports for optional dependencies.
4. Raise `DocumentLoadError` or `DocumentLoadingError` with clear dependency and failure messages.
5. Register the format in `UnifiedLoader` only after the loader is tested.
6. Add tests for success, unsupported/missing inputs, malformed inputs, optional dependency errors, and directory discovery if relevant.

Loader behavior should remain deterministic and should not make live network calls in tests.

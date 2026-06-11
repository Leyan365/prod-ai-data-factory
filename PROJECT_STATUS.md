# Project Status

This project is being built in tutorial-aligned phases. The current implementation has completed Phases 1-8, including export hardening, durable local storage, cleanup, and documentation refresh work.

## Completed Phases

| Phase | Status | Commit |
| --- | --- | --- |
| Phase 1: repair package baseline and smoke checks | Completed | `28f362c` |
| Phase 2: harden loaders and add tests | Completed | `7ec134b` |
| Phase 3: add robust preprocessing pipeline | Completed | `9d8f6ba` |
| Phase 4: add task templates and AI client layer | Completed | `237b967` |
| Phase 5: quality evaluation layer | Completed | `12dde15` |
| Documentation refresh after Phase 5 | Completed | `ef8f2ff` |
| Phase 6: export and dataset packaging hardening | Completed | `a9cfaff` |
| Phase 7: durable local storage persistence | Completed | `6e3619e` |
| Cleanup: remove obsolete notes file | Completed | `8cbc110` |
| Phase 8: documentation refresh after export and storage phases | Completed | `e394986` |
| Phase 9: Pydantic v2 and core model cleanup | Completed | `e6b34e4` |

## Test Status

Latest full regression run after Phase 9:

```text
76 passed, 3 skipped
```

Additional checks:

```text
scripts/smoke_check.py -> smoke ok
syntax ast.parse pass -> parsed 29 python files
```

The direct `compileall` check is not used as the final syntax signal because Windows may deny writes into existing `__pycache__` files. The read-only `ast.parse` check validates syntax without modifying cache files.

## Current Capabilities

- Importable `training_data_bot` package with repaired public exports.
- File and URL loading for TXT, MD, HTML, JSON, CSV, DOCX, PDF, and web fallback paths.
- Deterministic directory loading with recursive/non-recursive discovery and glob inclusion patterns.
- Robust text normalization and deterministic overlapping chunk generation.
- Template-driven task execution using an offline mock AI provider by default.
- Optional Gemini provider that reads `GEMINI_API_KEY` from the environment only.
- Deterministic rule-based quality evaluation.
- Hardened JSONL and CSV dataset export.
- Deterministic train/validation/test split packaging.
- Durable local JSON persistence for datasets and processing jobs under `output/storage`.

## Known Limitations

- Decodo integration is stubbed/fallback-only and not production scraping behavior.
- `ExportFormat.PARQUET`, `ExportFormat.HUGGINGFACE`, and `ExportFormat.OPENAI` are placeholders.
- `TextChunk.embeddings` and `TextChunk.topics` fields exist, but no embeddings or topic extraction pipeline is implemented.
- Phase 5 quality scoring is deterministic and rule-based. It is useful as a baseline filter, not a substitute for model-based evaluation.
- Local JSON storage is durable and deterministic, but it is not a production database, migration system, concurrent writer system, cloud storage layer, or query engine.
- A fallback local `BaseModel` remains for no-Pydantic environments and may drift from Pydantic behavior.
- No CLI entry point or formal package metadata is implemented yet.

## Next Planned Phase

Recommended Phase 10: packaging, CLI, and project metadata cleanup.

Suggested scope:

- Add formal package metadata and editable-install support.
- Add a minimal offline-first CLI for smoke-friendly local workflows.
- Keep default AI behavior offline/mock.
- Avoid embeddings, vector databases, production scraping, cloud storage, and new AI providers in this phase.

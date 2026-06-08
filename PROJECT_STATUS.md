# Project Status

This project is being built in tutorial-aligned phases. The current implementation has completed Phases 1-5, with Phase 5 present in the working tree and awaiting its commit.

## Completed Phases

| Phase | Status | Commit |
| --- | --- | --- |
| Phase 1: repair package baseline and smoke checks | Completed | `28f362c` |
| Phase 2: harden loaders and add tests | Completed | `7ec134b` |
| Phase 3: add robust preprocessing pipeline | Completed | `9d8f6ba` |
| Phase 4: add task templates and AI client layer | Completed | `237b967` |
| Phase 5: quality evaluation layer | Completed in working tree | Pending commit |

## Test Status

Latest full regression run after Phase 5:

```text
54 passed, 3 skipped
```

Additional checks:

```text
scripts/smoke_check.py -> smoke ok
syntax ast.parse pass -> parsed 27 python files
```

The direct `compileall` check was not used as the final syntax signal because Windows denied writes into existing `__pycache__` files. The read-only `ast.parse` check validates syntax without modifying cache files.

## Known Limitations

- Phase 5 quality scoring is deterministic and rule-based. It is useful as a baseline filter, not a substitute for model-based evaluation.
- Dataset export remains basic JSONL/CSV behavior; richer export formats are placeholders.
- Storage persistence is not implemented beyond lightweight placeholder classes.
- Decodo integration is stubbed/fallback-only and not production scraping behavior.
- `AIClient()` defaults to the offline mock provider. Gemini is opt-in only.
- No embeddings, vector databases, semantic chunking, or advanced orchestration are implemented.
- Pydantic v1-style validators still produce Pydantic v2 deprecation warnings.
- `datetime.utcnow` use still produces deprecation warnings in newer Python/Pydantic combinations.
- A fallback local `BaseModel` remains for no-Pydantic environments, which may drift from Pydantic behavior.
- `files completed.txt` is a user-owned modified file and is intentionally outside phase commits.

## Next Planned Phase

Recommended Phase 6: export and dataset packaging hardening.

Suggested scope:

- Make `DatasetExporter` robust and tested.
- Keep JSONL and CSV as the first-class supported formats.
- Define clear behavior for unsupported placeholder formats such as Parquet, Hugging Face, and OpenAI.
- Add deterministic split handling if `split_data=True`.
- Preserve model compatibility and avoid storage persistence changes until a later phase.

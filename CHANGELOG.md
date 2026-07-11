# Changelog

## [0.2.0] - 2026-07-11

Security and runtime hardening release.

- Added secure-by-default remote URL policy with host allowlisting, public-IP validation, redirect revalidation, and bounded responses.
- Made direct Decodo use fail closed and route safe fetching through the validated transport.
- Added URL, credential, token, authorization, API-key, and source-context redaction.
- Moved Gemini authentication to provider-supported headers.
- Hardened retries with bounded backoff, jitter, retry hints, and transient-error classification.
- Added schema-v2 storage envelopes with v1 read compatibility and UUID-only record lookup.
- Added atomic, fsynced, locked storage and export writes with rollback behavior.
- Added split-package manifests with filename, size, format, and SHA-256 verification.
- Added partial processing state, bounded failure metadata, and non-zero CLI outcomes.
- Added configurable document, run, remote-response, PDF-page, CSV-row, timeout, and concurrency limits.
- Added direct PyMuPDF text fallback for minimal Linux environments when high-level PDF extraction returns no text.
- Added hashed dependency locking generated from the lowest supported Python baseline.
- Added Python 3.10-3.12 CI coverage, wheel builds, wheel installation, and CLI/version verification.

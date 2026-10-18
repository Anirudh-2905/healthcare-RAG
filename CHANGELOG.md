# Changelog

## 2026-10-18 — Robustness: configuration and startup errors

### Changed

- Aligned the code-level default retrieval threshold with the evaluated value of `0.40`.
- Added startup validation for retrieval, chunking, agent-step and Ollama URL settings.
- Added setting-specific errors for malformed integer and floating-point environment values.
- Replaced client-visible service initialization exceptions with a stable `503` message.
- Preserved full initialization exceptions in application logs for diagnosis.

### Tests

- Added invalid-configuration and valid-boundary test coverage.
- Added a test proving sensitive initialization details are logged but not returned to API clients.
- Verified 24 passing tests with 66% total application coverage; configuration coverage is 94%.

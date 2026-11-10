---
name: receiptiq-python-engineering
description: Implement and refactor ReceiptIQ Python code using repository conventions, maintainable boundaries, typing, configuration, error handling, and focused tests. Use for general backend or Python changes.
---

# ReceiptIQ Python Engineering

## First inspect, then implement
1. Read README, dependency/lock files, project configuration, source layout, and nearby tests.
2. Identify the actual Python version, framework, formatter, linter, test runner, and architecture from repository files. Do not introduce replacements by preference.
3. Trace callers and consumers before changing a function, schema, or endpoint.
4. Prefer a minimal, cohesive change over speculative abstractions or broad rewrites.

## Implementation standards
- Use type hints on public functions and meaningful domain objects.
- Keep business logic separate from transport, file I/O, and external OCR/model integrations where the existing architecture permits.
- Validate external input at boundaries; use explicit domain errors and safe API error responses.
- Keep secrets and machine-specific paths out of source control. Use environment configuration and update the example env file without real credentials.
- Avoid mutable global state and hidden side effects.
- Use Decimal or an equivalent exact representation for monetary calculations; do not use binary floating point for currency arithmetic.
- Make date, locale, and currency assumptions explicit.
- Add dependencies only when needed; explain alternatives and operational costs.
- Keep functions small enough to test and name concepts consistently with the codebase.

## Testing
- Add unit tests for changed business logic and edge cases.
- Add integration tests for changed boundaries when practical.
- Use temporary directories and generated fixtures; never rely on personal receipt data.
- Test malformed files, empty inputs, missing fields, decimal separators, rounding, and failure paths relevant to the change.
- Run the repository's existing checks. Do not claim tests pass unless they were actually run.

## Completion report
Summarize behavior changed, key files, tests and exact results, migration/configuration needs, and any known limitations. Do not silently change public contracts.

---
name: receiptiq-docs
description: Create and maintain ReceiptIQ technical documentation, setup instructions, API examples, architecture notes, and portfolio-facing project explanations. Use when documenting features or preparing the project for reviewers.
---

# ReceiptIQ Documentation

## Workflow
1. Inspect the implementation first. Documentation must describe what exists, not an imagined architecture.
2. Separate implemented behavior, planned work, and future ideas with explicit labels.
3. Use the actual commands, environment variables, routes, schemas, and dependency versions from the repository.
4. Keep setup instructions reproducible on a clean machine. Include prerequisites, installation, configuration, run, test, and troubleshooting steps.
5. Document OCR limitations, supported file types, languages/locales, confidence semantics, and known failure cases.
6. For API docs, show request/response examples with synthetic data only; document validation and error responses.
7. For architecture docs, explain data flow and module responsibilities without adding components that do not exist.
8. For portfolio materials, state the problem, scope, technical decisions, evaluation protocol, measured results, and limitations honestly.

## Privacy and accuracy
- Use synthetic examples; never publish real receipts or personally identifying content.
- Do not expose secrets, tokens, local paths, or private dataset files.
- Never claim accuracy, speed, production readiness, or supported formats without reproducible evidence.
- Keep README concise and link to deeper docs where useful.

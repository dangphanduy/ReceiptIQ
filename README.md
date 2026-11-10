# ReceiptIQ

ReceiptIQ is an intelligent receipt and invoice processing project. The current direction is a Python-based solo portfolio build with a 6–8 week target. This repository is at an early bootstrap stage; framework, OCR engine, database, API framework, and deployment choices must be confirmed before implementation.

## Claude Code skills

Project-specific skills live in `.claude/skills/`:

| Skill | Use it for |
|---|---|
| `receipt-ocr-pipeline` | Receipt/PDF ingestion, image preprocessing, OCR integration, troubleshooting |
| `receipt-field-extraction` | Extracting and validating merchant, date, currency, totals, tax, and line items |
| `receipt-dataset-evaluation` | Public dataset selection, annotation, leakage prevention, OCR/field metrics |
| `receiptiq-python-engineering` | Python implementation, refactoring, typing, errors, tests, configuration |
| `receiptiq-review` | Reviewing diffs for correctness, financial integrity, privacy, security, and tests |
| `receiptiq-docs` | Setup, architecture, API, evaluation, and portfolio documentation |

## Project rules

- Inspect existing decisions before choosing frameworks or adding dependencies.
- Distinguish implemented functionality from proposed work.
- Treat OCR and extracted fields as uncertain candidates, not verified financial truth.
- Preserve source evidence and provenance for extracted values.
- Use exact decimal arithmetic for money and explicit locale/currency handling.
- Never commit real receipts, personal data, secrets, or credentials; use synthetic fixtures.
- Never claim benchmark accuracy or performance without reproducible evaluation.

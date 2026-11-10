---
name: receipt-ocr-pipeline
description: Design, implement, debug, and improve ReceiptIQ's receipt and invoice image/PDF ingestion and OCR pipeline. Use for upload, preprocessing, OCR, page handling, and text extraction tasks.
---

# Receipt OCR Pipeline

## Project context
ReceiptIQ is an intelligent receipt and invoice processing project. The confirmed direction is a Python-based, solo portfolio project with a short 6–8 week development window. Do not assume a specific web framework, OCR engine, database, cloud provider, or deployment target until it is documented in the repository or explicitly selected by the user.

## Workflow
1. Inspect the existing source tree, dependency files, README, configuration, and tests before editing.
2. Trace the current document flow end to end: input validation → file decoding → page/image extraction → preprocessing → OCR → normalized OCR output.
3. Preserve the existing architecture and interfaces. Propose a small change before introducing a new framework or major dependency.
4. Handle JPG, PNG, and PDF only where current requirements and installed libraries support them. Validate file type from content where feasible; do not trust filename extensions alone.
5. Keep preprocessing deterministic and configurable. Compare original and processed images; avoid destructive transforms that remove faint text, decimal points, or currency symbols.
6. Preserve page number, text, bounding boxes, confidence (when available), and source metadata in a stable internal representation.
7. Handle rotated, low-resolution, blurred, skewed, multi-page, and empty documents with explicit errors or warnings.
8. Keep OCR engine calls behind a small adapter/service boundary when the project architecture supports it, so engines can be evaluated or replaced without rewriting business logic.
9. Add focused tests using small synthetic or legally usable fixtures. Do not commit private receipts or personal data.
10. Run the relevant tests and report commands, outcomes, limitations, and files changed.

## Quality and safety rules
- Never silently fabricate OCR text or confidence values.
- Do not treat OCR output as verified financial data.
- Do not log raw receipt contents, names, addresses, phone numbers, payment details, or full extracted text by default.
- Enforce upload size, page-count, MIME/type, and resource limits at the appropriate boundary.
- Keep temporary files isolated and clean them up on success and failure.
- Do not add heavyweight OCR dependencies without explaining size, runtime, licensing, and platform implications.

## Done criteria
A change is complete when valid inputs work, invalid inputs fail clearly, edge cases are covered, outputs retain provenance, and tests pass.

---
name: receipt-field-extraction
description: Extract and normalize structured receipt and invoice fields from OCR output, including merchant, date, totals, tax, currency, and line items. Use when implementing parsing or correction logic.
---

# Receipt and Invoice Field Extraction

## Goal
Turn noisy OCR text into structured candidate fields while preserving evidence and uncertainty. OCR is not ground truth; extracted values must remain reviewable.

## Before coding
- Read the current data model, API schemas, examples, and tests.
- Confirm which fields are in the current MVP. Do not expand scope by adding every possible invoice field.
- Follow existing naming, typing, validation, and serialization conventions.

## Extraction workflow
1. Keep raw OCR output immutable; build structured candidates separately.
2. Normalize whitespace and common OCR confusions conservatively. Retain the original text for every candidate.
3. Extract only fields supported by the document and current requirements: merchant/supplier, document date, subtotal, tax, total, currency, and line items where in scope.
4. Use locale-aware number and date parsing. Never assume US decimal/grouping separators or a single currency.
5. Distinguish subtotal, tax, discount, tip, amount due, amount paid, and grand total. Do not select a total merely because it is the largest number.
6. Attach evidence to each candidate: source text/span or bounding box, page, extraction method, and confidence/score when available.
7. Validate arithmetic relationships as warnings, not automatic corrections. Rounding, discounts, deposits, and multiple tax rates can explain differences.
8. Represent missing, ambiguous, and conflicting values explicitly (for example null plus a status), rather than guessing.
9. Make user corrections possible without overwriting the original OCR evidence. Store corrected value and provenance separately if persistence is implemented.
10. Add tests for Vietnamese and international number/date formats, multiple totals, missing tax, discounts, noisy text, and multi-page documents.

## Guardrails
- Never invent a merchant, date, currency, tax, or amount.
- Do not infer personal identity or payment-card data when it is not required.
- Avoid brittle positional regexes as the only extraction strategy.
- Do not claim a confidence score is calibrated unless calibration has been measured.
- Do not automatically alter monetary values to make totals balance.

## Output contract
Follow the repository's existing schema. If no schema exists, propose a minimal typed contract first and wait for confirmation before introducing a persistent schema or public API contract.

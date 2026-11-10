---
name: receiptiq-review
description: Review ReceiptIQ changes for correctness, data integrity, security, privacy, performance, and missing tests. Use before merging or after implementing a feature or bug fix.
---

# ReceiptIQ Code Review

Review the diff and relevant surrounding code. Prioritize actionable findings over style preferences.

## Review checklist
1. **Correctness:** Does the implementation satisfy the stated requirement? Are edge cases and failure paths handled?
2. **Financial integrity:** Are amounts represented and parsed safely? Are subtotal, tax, discounts, totals, and currencies kept distinct?
3. **OCR uncertainty:** Are extracted values traceable to source evidence? Are low-confidence or conflicting values surfaced rather than silently accepted?
4. **Privacy:** Could logs, exceptions, fixtures, analytics, or stored files expose receipt images, names, addresses, phone numbers, or payment data?
5. **File security:** Are uploads validated? Are size/page limits, temporary files, path traversal, malformed PDFs/images, and resource exhaustion considered?
6. **Data integrity:** Are schema constraints, migrations, transactions, and correction history consistent with existing design?
7. **API behavior:** Are validation errors, status codes, response schemas, and backward compatibility correct where applicable?
8. **Performance:** Are large images/PDFs, repeated OCR calls, memory use, and synchronous work handled appropriately for the current scope?
9. **Tests:** Do tests cover success, invalid input, ambiguity, locale variations, and regressions?
10. **Maintainability:** Does the change follow existing conventions without unnecessary dependencies or duplicated logic?

## Finding format
For each finding include:
- Severity: Critical / High / Medium / Low
- File and line
- Concrete scenario and impact
- Recommended fix

Report only demonstrable issues. If no findings exist, say so and list residual risks or untested areas. Do not edit code during a review unless asked.

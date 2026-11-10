---
name: receipt-dataset-evaluation
description: Plan, inspect, label, split, and evaluate receipt/invoice datasets for ReceiptIQ. Use when selecting public datasets, preparing annotations, measuring OCR or field-extraction quality, or preventing data leakage.
---

# Receipt Dataset and Evaluation

## Dataset principles
ReceiptIQ is intended to use public data for a solo portfolio project. Verify dataset provenance and terms before downloading, redistributing, or publishing samples. Never assume a dataset is public-domain merely because it is publicly accessible.

## Workflow
1. Define the task being evaluated: OCR text recognition, document classification, key-field extraction, line-item extraction, or end-to-end field accuracy.
2. Record dataset name, source URL, version/date, license/terms, languages, document types, annotation format, and known limitations.
3. Inspect a sample before designing the pipeline. Identify image quality, rotations, layouts, languages, currencies, and annotation consistency.
4. Create a data card describing intended use, exclusions, known bias, privacy risks, and permitted redistribution.
5. Split by document/receipt identity and, where possible, merchant/template identity. Never put near-duplicate pages or augmented copies of one receipt across train and test.
6. Keep a fixed, untouched test set. Fit preprocessing thresholds, vocabularies, and learned components using training data only.
7. Normalize annotations through a documented mapping; preserve original labels and record every transformation.
8. Measure OCR with suitable character/word error metrics and field extraction with exact match plus normalized match. Report per-field precision, recall, F1, and exact-match accuracy where applicable.
9. Report performance by language, document type, image quality, and field presence when sample size permits.
10. Include failure examples and confidence/coverage analysis; aggregate scores alone are insufficient.

## Required checks
- Duplicate and near-duplicate detection.
- Missing, malformed, and inconsistent annotations.
- Train/validation/test leakage.
- License and attribution review.
- Reproducible evaluation command and pinned dataset version.
- Baseline comparison before claiming an improvement.

## Reporting
State dataset size and split, preprocessing, model/engine versions, hardware, runtime, metric definitions, and limitations. Do not compare scores across incompatible datasets or evaluation protocols. Never fabricate benchmark results.

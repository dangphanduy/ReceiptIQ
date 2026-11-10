# OCR baseline

## Goal

Get an off-the-shelf OCR number on SROIE before building anything on top of it.
Later stages (field extraction, the MVP's confidence display) are measured against this.

- Engine: PaddleOCR 3.x (`lang="en"` resolves to PP-OCRv6_medium det + rec in 3.7.0), CPU,
  oneDNN off (see the script), **no training or fine-tuning**.
- Doc orientation, unwarping and text-line orientation models are disabled,
  because SROIE images are upright, flat scans.
- Unit: text lines. PaddleOCR returns line-level boxes, and SROIE `words` are
  also mostly line-level regions, so the two can be compared directly.

## What exists now

`scripts/run_ocr_baseline.py <image>` runs on **one image** and writes
`outputs/ocr/<stem>.json`:

```json
{
  "schema_version": "1.0",
  "image": {"path": "...", "width": 463, "height": 605},
  "engine": {"name": "paddleocr", "paddleocr_version": "...", "device": "cpu", "inference_seconds": 1.2},
  "num_lines": 35,
  "mean_confidence": 0.96,
  "lines": [
    {"text": "ABC HO TRADING", "confidence": 0.99,
     "polygon": [[x, y], [x, y], [x, y], [x, y]], "bbox": [x0, y0, x1, y1]}
  ]
}
```

`bbox` is `[x_min, y_min, x_max, y_max]` in pixels, the same convention as SROIE
`bboxes`. `confidence` is the recognition score. It is not calibrated and is not
a probability that the text is correct.

## How it will be measured (not yet implemented)

Evaluate on the **test split** only (361 images). Train is kept for later error analysis and tuning.

### OCR (SROIE Tasks 1–2)

1. **Detection**: greedily match predicted boxes to ground-truth boxes at IoU ≥ 0.5.
   Report precision, recall and F1.
2. **Recognition**, on matched pairs: character error rate (CER = edit distance /
   GT length) and line exact-match rate after normalization (uppercase, collapse whitespace).
3. **End-to-end**: word-level precision/recall/F1 on whitespace-split tokens
   (multiset match per image), close to the SROIE Task 2 protocol.
4. Also report mean inference time per image on CPU, and the confidence
   distribution for correct and incorrect lines. This shows whether confidence
   is useful for flagging lines the MVP should ask the user to review.

### Field extraction (SROIE Task 3)

For `company`, `date`, `address`, `total`:

- Normalize the prediction and the ground truth the same way (uppercase, collapse
  whitespace, strip). `total` is compared as a decimal string, and `date` as written.
- Report per-field exact-match accuracy, plus micro precision/recall/F1 where an
  empty prediction counts as a miss, not a false positive.
- The first extractor will be rules and heuristics on top of the OCR lines
  (regex for date/total, position for company/address). This gives a floor
  before trying layout-aware models.

## Known limitations

- Single image per run. No batch evaluation yet.
- English models only. Vietnamese receipts are not supported or evaluated.
- Numbers on SROIE only describe Malaysian scanned receipts (see [dataset.md](dataset.md)).
- Output is not validated for accounting or tax use.

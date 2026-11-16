# OCR baseline

## Goal

Get an off-the-shelf OCR number on SROIE before building anything on top of it.
Later stages (field extraction, the MVP's confidence display) are measured against this.

- Engine: PaddleOCR 3.x (`lang="en"` resolves to PP-OCRv6_medium det + rec in 3.7.0), CPU,
  oneDNN off (see `OCR_CONFIG` in `ml/ocr_engine.py`), **no training or fine-tuning**.
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

Both scripts get the PaddleOCR config (`OCR_CONFIG`) and the image → OCR document
step (`run_ocr`) from `ml/ocr_engine.py`, so single-image and batch outputs cannot drift.
`run_ocr` takes the encoded image bytes and decodes them the same way PaddleX decodes a
file path (`cv2.imdecode`), so the same JPEG gives the same OCR from either script. The
batch runner passes the original JPEG bytes stored in `data/sroie`. Note that
`data/samples/*.jpg` are re-encoded copies (quality 95), so their OCR can differ slightly
from the batch output for the same key.

### Batch over a split (RIQ-18)

```bash
$PY scripts/run_ocr_batch.py --split test --limit 2   # first 2 rows of the test split
$PY scripts/run_ocr_batch.py --split test             # all 361 rows, ~3.5 h on CPU
```

- Reads `data/sroie` with `load_from_disk` and writes one `<key>.json` per row (same
  schema as above) to `--out-dir`, default `outputs/ocr/<split>/`. `image.path` is
  `sroie:<split>/<key>`, because the image comes from the dataset and not from a file.
- PaddleOCR is built once per run. `--limit N` takes the first N rows of the split.
- **Resume:** a rerun skips every key whose `<key>.json` already exists. Each JSON is
  written to `<key>.json.tmp` and then renamed. An interrupted run therefore leaves no
  truncated `<key>.json`, and that key is OCR'd again next time. A leftover `.tmp` file
  is overwritten. To redo a key, delete its JSON.
- **Failures:** if one image fails, the key and the error are printed to stderr and
  the run continues. At the end the failed keys are listed and the exit code is 1.
  Failed keys have no JSON, so a rerun retries them.
- `manifest.json` in the out-dir describes the **latest** run. It is overwritten on
  every run, so on a resumed run `processed` only counts that run. It contains:
  - SROIE `repo_id` and `revision` (the `commit_sha` from `data/sroie/manifest.json`)
  - split, `limit`, the paddleocr and paddlepaddle versions, `OCR_CONFIG`
  - `counts` (`processed` / `skipped` / `failed`), the failed keys with their errors
  - `started_at` / `finished_at` (UTC)

## Sanity check on 30 test samples (RIQ-17)

A manual check before writing any evaluation code. It is not a benchmark.

- Data: the first 30 rows of the SROIE **test** split at the pinned revision
  (`download_sroie.py --export-samples 30`). Each image was run separately with
  `run_ocr_baseline.py` → 30 JSON files in `outputs/ocr/`.
- Engine: PaddleOCR 3.7.0 / paddlepaddle 3.3.1, CPU, oneDNN off.
- Method: OCR lines were compared with GT `words` and `entities` after uppercasing and
  collapsing whitespace. Every mismatch below was checked against the image.

### Date and total (manual count)

| Field | Read correctly by OCR | Notes |
|---|---|---|
| `date` | 30 / 30 | GT for `X51005442322` says `03 MAR 19`. The receipt prints `03 Mar 18`, and OCR read `18`. This is a GT error, not an OCR miss. |
| `total` | 30 / 30 | In every case the GT value appears on the TOTAL / Nett Total / Amount Due line or the value box next to it. |

"Read correctly" only means the string is in the OCR text. It says nothing about
whether an extractor would choose it. That is harder than it sounds:
- The same value often appears 2–7 times (item amount, subtotal, total, cash).
- Labels and values are separate lines that are not always in reading order. In
  `X51005268408`, `169.78` comes before its `Total Sales (Inclusive GST) RM` label.

### What OCR gets wrong

- **Character confusions:**
  - `8` read as `B` in a postcode (`X51005433548`, `81750` → `B1750`; GT has the same error).
  - `9` read as `8` in a small amount (`X51005442343`, `67.92` → `67.82`).
  - `i` read as `l` (`X51005442382`, `Tampoi` → `Tampol`; GT has the same error).
  - `W` read as `H` (`X51005442322`, `www` → `HWW`).
  - `I`/`1` confusion in item codes (`X51005442375`, `I001…` → `1001…`).
- **Overlap with handwriting or stamps:**
  - `&` read as `S` where a signature crosses the header (`X51005361908`).
  - A garbage character in `1x` next to a pen mark (`X51005442366`).
- **Line segmentation, not recognition:**
  - Most "unmatched" GT lines come from splitting differently. GT splits addresses at
    commas (`NO 2 & 4` / `JALAN BAYU 4`), while OCR returns the whole printed line.
  - OCR sometimes merges two GT boxes into one line
    (`Receipt No.: CS1803/28617 Date: 15/03/2018`).
  - OCR sometimes splits one printed row into fragments (`* RM 155.66 UNI (` + value),
    so line-level IoU matching needs care.
- **Missed lines:** No whole printed line was clearly dropped. Line counts are mostly
  within ±10% of GT. The worst case is `X51005361908` (50 vs 61), and that gap is
  explained by merged lines.
- **Damaged paper:** `X51005268275` has punched holes that cut `TOTAL` and `CASH`.
  OCR returns `TOT'` / `CASi.`, which is correct for what is visible.

### GT typos found (OCR was right)

`X51005442322` date year, `ROASTEO` (OCR: `ROASTED`). `X51005230621` `INVOLCE`,
`INCLUSLVE`. `X51005442375` `FRLED`. `X51005442366` `FUJ1500ML` (receipt: `FUJI500ml`).
This confirms that an exact-match score against GT will undercount OCR quality.

### Other observations

- `X51005268275` and `X51005301666` appear to be the same receipt (same lines and
  total). Duplicate receipts in the test split would inflate any score.
- Mean recognition confidence is 0.983 (per-image range 0.949–0.995). It is high even on
  lines with the errors above, so confidence alone will not flag them.
- Mean `inference_seconds` is 34.7 s per image on this CPU, slower than the 18 s
  measured on the first image. A full 361-image run takes about 3.5 hours of
  inference alone.

Text quoted above comes from SROIE ground truth and images. See the
[attribution](dataset.md#attribution-cc-by-40).

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

- Batch OCR exists, but metrics (detection, CER, field accuracy) are not implemented yet.
- The batch runner is sequential, one image at a time on CPU.
- English models only. Vietnamese receipts are not supported or evaluated.
- Numbers on SROIE only describe Malaysian scanned receipts (see [dataset.md](dataset.md)).
- Output is not validated for accounting or tax use.

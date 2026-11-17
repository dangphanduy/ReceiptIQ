# OCR baseline

## Goal

Get an off-the-shelf OCR number on SROIE before building anything on top of it.
Later stages (field extraction, the MVP's confidence display) are measured against this.

- Engine: PaddleOCR 3.x (`lang="en"` resolves to PP-OCRv6_medium det + rec in 3.7.0), CPU by
  default or one NVIDIA GPU (`--device`, RIQ-57), oneDNN off (see `OCR_CONFIG` in
  `ml/ocr_engine.py`), **no training or fine-tuning**.
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
  "engine": {"name": "paddleocr", "paddleocr_version": "3.7.0",
             "paddlepaddle_distribution": "paddlepaddle", "paddlepaddle_version": "3.3.1",
             "lang": "en", "device": "cpu", "inference_seconds": 1.2},
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

**Behaviour change for JPEGs with EXIF orientation (RIQ-18).** Before RIQ-18,
`run_ocr_baseline.py` took `image.width` / `image.height` from PIL, which ignores EXIF
orientation. PaddleX, however, OCR'd the image after `cv2` had already rotated it. For a
rotated JPEG this recorded, for example, 40×30 while every box was in 30×40 coordinates.
Both size and boxes now come from the same decoded (rotated) image, so `width` × `height`
is the frame the boxes are measured in. This change is kept on purpose and covered by
`test_exif_oriented_jpeg_records_rotated_size_matching_boxes`. It does not change any
SROIE output: no image in the pinned train (626) or test (361) split has an EXIF
orientation other than 1 (checked during RIQ-18).

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
  - split, `limit`, the paddleocr version, the paddle distribution and version,
    `device`, `OCR_CONFIG`
  - `counts` (`processed` / `skipped` / `failed`), the failed keys with their errors
  - `started_at` / `finished_at` (UTC)

### GPU and `--device` (RIQ-57)

```bash
$G scripts/run_ocr_batch.py --split test --device gpu:0 --out-dir outputs/ocr/test-gpu
$G scripts/run_ocr_baseline.py data/samples/<key>.jpg --device gpu:0
```

- `$G` is the separate GPU env. It is cloned from the CPU env, with `paddlepaddle-gpu==3.3.1`
  from the PaddlePaddle cu126 index and torch removed. The steps are in `CLAUDE.md` →
  Environment. `pyproject.toml` still declares CPU `paddlepaddle`, the default.
- `--device` is `cpu` (default) or `gpu:<int>`. Anything else is an argparse error.
  `create_engine` refuses a GPU when paddle has no CUDA build or the GPU index does not exist,
  so `engine.device` in each JSON and `ocr.device` in `manifest.json` name the device that ran.
- `engine` / `ocr` also record `paddlepaddle_distribution` (`paddlepaddle` or
  `paddlepaddle-gpu`). The field is additive, so `schema_version` stays `"1.0"`.
- **No mixing.** Before loading the model, the batch reads `manifest.json` and every
  `<key>.json` in `--out-dir`. If any of them came from another `device`, paddle distribution
  or paddle version, it exits with an error asking for another `--out-dir` and writes nothing.
  The `<key>.json` files are checked as well because a killed run leaves no manifest. Files from before RIQ-57
  have no distribution field and count as `paddlepaddle`, since only the CPU build could
  write them.
- **GPU and CPU outputs may differ slightly.** They use different kernels, so float results are
  not bit-identical. In a 5-key smoke test (first 5 test rows), line counts and every line's
  text matched. Boxes differed by at most 1 px and confidences by at most 0.0034. Compare
  numbers only between runs on the same device.
- On the RTX 3050 Laptop (4 GB) used here, GPU inference was about 0.5 s/image against
  15–31 s on CPU for the same 5 images, with about 1.2 GB VRAM in use during the run.
- The test split has 55 scans with a long side ≥ 4000 px (mostly 4961×7016, rows 183–343).
  PaddleOCR resizes them to a 4000 px long side, and VRAM then reaches about 3.9 GB of 4 GB.
  One full run on this card died at row 196 with no Python traceback (most likely GPU memory).
  Rerunning the same command resumed and finished it. If that happens, rerun; do not switch
  the out-dir to CPU.

### Detection metrics (RIQ-19)

```bash
$PY scripts/eval_ocr_detection.py --split test --ocr-dir outputs/ocr/test-gpu
#   [--out outputs/eval/detection_test.json] [--iou 0.5]
```

- Matching is in `ml/ocr_eval.py`, which is pure Python with no paddle / cv2 / datasets.
  - OCR `lines[].bbox` and SROIE `bboxes` are both `[x0, y0, x1, y1]`.
  - Every pred × GT pair with IoU ≥ threshold (inclusive, default 0.5) is a candidate.
  - Candidates are taken by IoU descending, with ties broken by (pred index, GT index).
  - Each pred and each GT box is matched at most once. Matched pairs are TP, leftover preds
    are FP, and leftover GT boxes are FN.
  - A malformed or zero-area box has IoU 0 with everything. The test split has one such GT
    box, and it always counts as a FN.
- Per image, P = TP/(TP+FP), R = TP/(TP+FN) and F1 = 2TP/(2TP+FP+FN). F1 equals the
  harmonic mean of P and R when both exist. A zero denominator gives `null`, so an image
  with no OCR lines has P = `null` and R = 0. P = R = 0 gives F1 = 0.0.
- The `micro` scores come from TP/FP/FN summed over the split, not from averaging
  per-image scores.
- GT is read with only the `key`, `bboxes` and `image_size` columns, so images are never
  decoded.
- A key without `<key>.json` goes to `missing_keys` and is left out of the metrics. It is
  never counted as all-FN, so a partial run can be scored, but only on the keys it covers.
- If the OCR width/height differs from GT `image_size`, the key goes to
  `size_mismatch_keys` but is still scored.
- The report (written atomically) contains `iou_threshold`, `split`, `sroie_revision`
  (from `data/sroie/manifest.json`), `ocr_dir`, `ocr_device` and `ocr_schema_version`.
  It also has `counts` {`evaluated`, `missing`}, `micro` {tp, fp, fn, precision, recall,
  f1}, the key lists and `per_image` rows.
  - `ocr_device` comes from the OCR `manifest.json`. Before the batch has written one, it
    comes from the per-key JSONs.
  - OCR JSON that mixes schema versions or devices is refused.
- **Caveat: line-level IoU undercounts.** GT often splits one printed line into several
  boxes, between label and value or between columns (`INVOICE NO` / `: PEGIV-…`, quantity
  `1` / amount `193.00`, `POS:` / `1`), while OCR returns the whole printed line. That merged
  OCR box matches at most one GT part, so the other parts become FN even though the text was
  read. These splits are not at commas. The separate comma problem (one box, text cut at the
  first comma; see "GT text cut at the first comma" below) leaves the box intact, so it
  barely affects detection. Some FP come from handwriting
  (e.g. a handwritten total), which SROIE GT does not annotate. Treat these scores as a
  lower bound on detection quality, not as a SROIE Task 1 leaderboard number.

### Recognition metrics (RIQ-20)

```bash
$PY scripts/eval_ocr_recognition.py --split test --ocr-dir outputs/ocr/test-gpu
#   [--out outputs/eval/recognition_test.json] [--iou 0.5]
```

- Pairs are exactly the detection TP pairs: `ml.ocr_eval.match_greedy` at IoU ≥ `--iou`.
  The GT text of `bboxes[i]` is `words[i]`; a row whose `words` and `bboxes` differ in
  length is refused (on the pinned revision they are equal on every train and test row).
- **Normalization** (`ml/text_norm.py`, `normalize_text`, the single copy that field
  evaluation in E3 must reuse): uppercase, collapse every whitespace run (tabs, newlines,
  U+00A0, ...) to one space, strip. No Unicode NFC/NFKC, no punctuation removal.
  Uppercasing can change length (`ß` → `SS`).
- Per pair: CER = Levenshtein distance over code points (`ml/ocr_recognition.py`, pure
  Python) / length of the normalized GT. CER can exceed 1 when the prediction is longer.
  Exact match = normalized texts are equal.
- **Empty-GT rule:** a pair whose normalized GT is empty has no CER. It is left out of the
  CER stats and counted in `skipped_empty_gt`, but its exact match still counts.
- The report has the same provenance and input contract as detection (`missing_keys`,
  `size_mismatch_keys`, mixed schema/device refused, malformed JSON fails naming the file,
  GT columns `key`/`words`/`bboxes`/`image_size` only, atomic write) plus `normalization`,
  `counts` {`images_evaluated`, `missing`, `pairs`, `cer_pairs`, `skipped_empty_gt`},
  `cer` {`mean`, `median`} over pairs, `exact_match_rate` and the 20 `worst_lines`
  (CER desc, ties by key and GT index). Any stat with an empty denominator is `null`.
- **Caveat: GT text cut at the first comma dominates the worst lines.** OCR reads the full
  line, so such a pair gets a CER far above 1 although OCR read it correctly. These pairs
  pull the mean CER up strongly; the median is much less affected. GT typos (see below) also
  count as OCR errors. Check the image before calling a worst line a model error.

### End-to-end word metrics (RIQ-21)

```bash
$PY scripts/eval_ocr_e2e.py --split test --ocr-dir outputs/ocr/test-gpu
#   [--out outputs/eval/e2e_test.json]
```

- Per image, every OCR `lines[].text` and every GT `words` string goes through
  `normalize_text` (the same function as recognition, `ml/text_norm.py`) and is split on
  whitespace. Punctuation stays attached to its token (`Date: 12/3/56` → `DATE:`, `12/3/56`).
  Boxes, line order and line boundaries are ignored.
- Tokens are matched as a **multiset** (`collections.Counter` intersection, `ml/ocr_e2e.py`,
  pure Python): TP = Σ min(pred count, GT count) per token, FP = pred tokens − TP,
  FN = GT tokens − TP. Pred `A A B` vs GT `A B B` gives TP 2, FP 1, FN 1.
- `micro` sums TP/FP/FN over the split first and then takes the ratios; it is not the mean of
  per-image F1. `per_image` rows carry each image's tp/fp/fn/precision/recall/f1.
- **Zero-denominator rule: `null`.** No predicted tokens gives P = `null`, no GT tokens
  gives R = `null`, and both empty gives P = R = F1 = `null`. Other defined ratios remain
  numeric, including 0.0 when the numerator is zero. This matches detection's convention.
- Same input contract and provenance as detection (`missing_keys`, `size_mismatch_keys`, mixed
  schema/device refused, malformed JSON fails naming the file, non-str OCR text or GT word
  refused, atomic write). GT columns read: `key`, `words`, `image_size` only. The report has
  `split`, `sroie_revision`, `ocr_dir`, `ocr_device`, `ocr_schema_version`, `normalization`,
  `counts` {`evaluated`, `missing`}, `micro` {tp, fp, fn, precision, recall, f1}, the key lists
  and `per_image`. There is no `--iou`, because no box matching is involved.
- **Caveat: comma-truncated GT text turns correct OCR words into FP** (see "GT text cut at the
  first comma"). No GT `words` element contains a comma, so every predicted token with a comma
  is a FP. Words after a cut comma may also become FP when they have no other matching GT
  occurrence in that image. On the full test split (`outputs/ocr/test-gpu`), 1,308 of 39,963
  predicted tokens contain a comma, against 6,166 FP in total. This GT can make precision
  pessimistic.

#### Differences from the official SROIE Task 2 protocol

Source for the official side: the [Task 2 description and evaluation protocol in Huang et al.,
ICDAR 2019](https://arxiv.org/html/2103.10213v1) and the
[RRC SROIE tasks page](https://rrc.cvc.uab.es/?ch=13&com=tasks). The
[RRC Task 2 clarification](https://rrc.cvc.uab.es/?ch=13&com=news&id=3&view=data)
covers splitting detected sentences into words and ignoring Chinese characters. The official
RRC evaluation script was not inspected.

Same as the official description:
- Strings are tokenized by splitting on space, and punctuation is part of a token (official
  example: `Date: 12/3/56` → `Date:`, `12/3/56`).
- Repeated words count repeatedly ("If certain words are repeated in an image, they are
  expected to also be repeated in the submission results"), hence the multiset match.
- P = correct matches / predicted words, R = correct matches / GT words, F1 is the headline.

Different (verified):
- **Ground truth.** Officially, GT is "the list of words that appear in the transcriptions".
  Here it is the `words` column of the HF copy `jsdnrs/ICDAR2019-SROIE` at the pinned
  revision, where many lines are cut at the first comma (see above). Scores here are not on
  the official GT.
- **Character scope.** Officially, the task is "restricted to words comprising Latin
  characters and numbers only", and Chinese characters are to be ignored. Here no token is
  filtered; every token is scored.
- **Split rule.** The official description says strings are split on space, without specifying
  the behavior for repeated spaces or other whitespace [unverified]. Here `str.split()` splits
  on any whitespace (tabs, newlines, U+00A0, ...) and drops empty tokens.
- **Input.** Officially, participants submit a list of recognized words. Here the words come
  from PaddleOCR line texts split on whitespace. This matches the RRC note that the submitted
  words are "split from the detected sentences".

Not stated in either official source (our choice, [unverified] whether the official script
does the same):
- Case: tokens are uppercased (case-insensitive match). [unverified]
- Match criterion: exact string equality after normalization; no edit-distance tolerance.
  [unverified]
- Aggregation: micro over the whole split, not a per-image average. [unverified]
- Empty denominators give `null`. [unverified]

So these numbers are close to Task 2 in spirit but are **not** comparable with the SROIE
Task 2 leaderboard.

### Latency and confidence (RIQ-22)

```bash
$PY scripts/run_ocr_batch.py --split test --limit 40 --device cpu --out-dir outputs/ocr/test-cpu-latency
$PY scripts/eval_ocr_confidence.py --split test
#   [--ocr-dir outputs/ocr/test-gpu] [--latency-dir outputs/ocr/test-cpu-latency]
#   [--out outputs/eval/confidence_test.json] [--chart outputs/eval/confidence_hist_test.png] [--iou 0.5]
```

Question: can OCR line confidence flag lines for user review in the MVP, and from which threshold?

- **Line** = a detection-matched pair from `ml.ocr_recognition.line_pairs` (IoU ≥ 0.5, same
  matching and normalization as recognition). Its confidence is the OCR line's `confidence`;
  it is **wrong** when `exact` is false. Rule: flag if confidence < t. Positive class = wrong
  line, so precision = share of flagged lines that are wrong, recall = share of wrong lines that
  get flagged. Unmatched OCR lines and missed GT boxes are not in this analysis.
- **Two views.** `all` = every pair. `clean` = all minus suspected GT comma-cuts
  (`ml.ocr_confidence.is_comma_cut`): the normalized OCR text has a comma and its part before
  the first comma, stripped, equals the normalized GT (see "GT text cut at the first comma").
  On test-gpu it removes 862 of 17,645 pairs, all of them wrong by construction.
- **Threshold grid** 0.50…0.99 step 0.01, plus 0.990…0.999 step 0.001 (most lines sit there),
  59 values in total. The candidate is the max-F1 row, ties to the lower t. Each view also has
  a **budget** point: the highest t that flags at most 15% of lines (`null` if none). The
  report keeps the whole sweep, so another operating point can be read from it without
  rerunning.
- **Latency** = `engine.inference_seconds` (`predict()` only; decode and model load excluded),
  for `--ocr-dir` and for `--latency-dir` (optional; `null` if it has no OCR JSON). Stats: n,
  mean, median, p95, min, max. **p95 is nearest-rank**: the ⌈0.95·n⌉-th smallest value, no
  interpolation. The CPU sample is the **first 40 test rows** (split order) on this machine
  (`enable_mkldnn=False`). No other OCR or heavy job ran, but short pytest/ruff runs happened
  now and then during the sample (see Known limitations).
- **Warm-up.** The first image of a fresh process can include warm-up, so each latency block
  also has `first_image_seconds` and `stats_excluding_first`. In the CPU sample the first image
  (24.6 s) was below the mean, so warm-up did not show. `outputs/ocr/test-gpu/` was written by
  several resumed runs, so there its first image is not the only one after a warm-up.
- GPU confidences are used for the analysis; RIQ-57 showed CPU and GPU confidences differ by
  ≤ 0.0034.
- The chart (Pillow, no matplotlib) shows the share of each class per bin: one `< 0.80` bin,
  then 0.01 bins up to 1.00, with the candidate threshold as a dashed line and the SROIE
  CC BY 4.0 credit.
- **Candidate threshold (clean view): t = 0.993**, precision 0.33, recall 0.60 (F1 0.42),
  flagging 28.4% of lines (all view: t = 0.996, P 0.36, R 0.64, flags 34.7%). **Verdict:**
  confidence is a weak but real signal, good enough to rank lines for review in the MVP but not
  to be the only review gate, because most wrong lines also have high confidence.
- **Tight review budget:** the budget point is t = 0.97 in both views (clean: P 0.41, R 0.29,
  flags 10.7%). It flags fewer lines with higher precision but misses most wrong lines. The
  next grid step, 0.98, already flags more than 15%. The final operating point belongs to the
  MVP field-level confidence (RIQ-28).
- Calibration and field-level confidence are out of scope (RIQ-49, RIQ-28). The full results
  table belongs to RIQ-23.

### GT text cut at the first comma (open data bug)

Two different GT problems were mixed up in earlier notes:

- **(a) One printed line, several GT boxes.** Real annotation granularity: label and value,
  or table columns, are separate boxes (`INVOICE NO` / `: PEGIV-…`, `POS:` / `1`). It hurts
  line-level detection (see the RIQ-19 caveat). No case was found where GT splits a line at
  a comma into two boxes.
- **(b) One GT box over the whole printed line, text cut at the first comma.** The box is
  right, but `words[i]` stops before the first comma and the rest is gone: GT `9` for the
  printed `9, JALAN SUBANG JASA 3,`, GT `NO 2 & 4` for `NO 2 & 4, JALAN BAYU 4,` (this
  example was earlier misread as (a)), GT `1` for the amount `1,007.50`. No GT `words`
  element in either split contains a comma, while most address entities do. It hits
  addresses most, but any line with a comma (amounts with a thousands separator, item
  lines) can be cut. It affects every metric that uses `words` text (recognition and
  end-to-end), on both splits.
- `entities` (company/date/address/total, used by field evaluation) are not cut: address
  entities keep their commas and the text the `words` lost, and a total like `1,007.50`
  is intact.
- Likely cause (hypothesis, not yet confirmed against the original SROIE box files): the
  conversion to the HF copy split each original `x1,y1,…,x4,y4,TEXT` line on every comma and
  kept only the first text field. Fixing GT is tracked as a separate bug; until then,
  recognition numbers on this GT are pessimistic and the metric rules stay unchanged.

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
  - Most "unmatched" GT lines come from splitting differently: GT has separate boxes for
    label and value or for columns, while OCR returns the whole printed line. (This note
    first cited `NO 2 & 4` / `JALAN BAYU 4` as a comma split; RIQ-20 showed that GT line is
    one box whose text is cut at the comma. See "GT text cut at the first comma".)
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

## How it will be measured

Detection (item 1 below) is implemented in RIQ-19, recognition (item 2) in RIQ-20 and
end-to-end word metrics (item 3) in RIQ-21.
Everything else in this section is planned, not implemented yet.

Evaluate on the **test split** only (361 images). Train is kept for later error analysis and tuning.

### OCR (SROIE Tasks 1–2)

1. **Detection**: greedily match predicted boxes to ground-truth boxes at IoU ≥ 0.5.
   Report precision, recall and F1.
2. **Recognition**, on matched pairs: character error rate (CER = edit distance /
   GT length) and line exact-match rate after normalization (uppercase, collapse whitespace).
3. **End-to-end**: word-level precision/recall/F1 on whitespace-split tokens
   (multiset match per image, micro over the split), close to the SROIE Task 2 protocol but not
   identical (see "Differences from the official SROIE Task 2 protocol").
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

- Detection (RIQ-19), recognition (RIQ-20: CER and line exact match on matched pairs) and
  end-to-end word (RIQ-21) metrics exist. Field-accuracy metrics are not implemented yet.
- End-to-end word scores use the HF copy's `words` as GT and score every token, so they are
  not comparable with the SROIE Task 2 leaderboard (see the RIQ-21 section).
- Recognition is only scored on detection-matched pairs, so unmatched lines never reach it.
  The same holds for the confidence analysis (RIQ-22).
- CPU latency (RIQ-22) is a 40-image sample (first 40 test rows) on this one machine, measured
  while short pytest/ruff runs happened now and then. It is not a full-split or cross-hardware
  benchmark. p95 at n = 40 is unstable: 51.9 s, or 62.3 s without the first image. Treat the
  CPU numbers as indicative.
- The comma-cut filter behind the `clean` confidence view is a heuristic: it can drop a real
  OCR error that looks like a cut, and it misses cut GT that does not fit the pattern. It also
  leaves other GT artefacts in place: e.g. `X51005677327` GT `ROUNDING :` has the colon far right
  in the same box, and OCR reads `Rounding` correctly at confidence 1.000. Such pairs count as
  wrong, so the precision for "line is wrong" is pessimistic.
- GT `words` text is cut at the first comma on many lines in both splits (open data bug,
  see "GT text cut at the first comma"). This inflates CER and end-to-end FP; `entities` are
  not affected.
- Detection is scored at line level against GT boxes that often split one printed line, so
  recall and precision are underestimated (see the RIQ-19 caveat above). There is no
  word-level or polygon matching, and no Hungarian (optimal) assignment.
- Handwriting and stamps are read by OCR but not annotated in SROIE GT, so they count as FP.
- On a 4 GB GPU the largest scans use nearly all VRAM, and a run can die without a
  traceback. Resuming works, but there is no automatic retry.
- The batch runner is sequential: one image per `predict` call, on CPU or a single GPU.
- GPU and CPU outputs are not bit-identical (see RIQ-57 above). Keep them in separate out-dirs.
- The GPU env only works without torch installed (cuDNN DLL clash on Windows). Paddle also
  warns that the cu126 wheel was compiled with cuDNN 9.9 but ships `nvidia-cudnn-cu12` 9.5.
- English models only. Vietnamese receipts are not supported or evaluated.
- Numbers on SROIE only describe Malaysian scanned receipts (see [dataset.md](dataset.md)).
- Output is not validated for accounting or tax use.

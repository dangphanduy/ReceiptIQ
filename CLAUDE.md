# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ReceiptIQ is a solo portfolio project (6–8 weeks): receipt OCR + key-field extraction
(`company`, `date`, `address`, `total`) on ICDAR 2019 SROIE. It is currently at **phase 1:
research foundation and OCR baseline**. Keep changes inside that scope. Do not add
FastAPI, databases, Redis/Celery, a frontend, deployment or heavy training frameworks,
and do not train models, unless the user asks for that phase.

## Environment

- Python 3.13 only (`requires-python = ">=3.13,<3.14"`), the only version tested with Paddle 3.3.1.
- The user's dev env is the Anaconda env at `E:\Anaconda\Enviroment\Data-Science`.
  Call its interpreter directly: `E:/Anaconda/Enviroment/Data-Science/python.exe -m <tool>`.
  System `python` is 3.14 and has none of the deps.
- Project is installed editable (`pip install -e ".[dev]"`), which is how `scripts/` can import `ml`.
- Do not add `opencv-python` / `opencv-python-headless`. `paddleocr` → `paddlex` already
  installs `opencv-contrib-python`, and two OpenCV wheels clobber the same `cv2` package.
- `OCR_CONFIG` in `ml/ocr_engine.py` sets `enable_mkldnn=False`. Paddle 3.3.1 oneDNN crashes on Windows CPU
  (`ConvertPirAttribute2RuntimeAttribute`). About 18 s/image on CPU as a result.
- `brotlicffi>=1.2` is required in that env, otherwise `huggingface_hub` downloads crash in
  `httpx2` with `unexpected keyword argument 'output_buffer_limit'`.
- GPU env (RIQ-57): `E:\Anaconda\Enviroment\riq-gpu`, a clone of Data-Science with
  `paddlepaddle-gpu==3.3.1` (cu126) instead of `paddlepaddle`. It is only for `--device gpu:N`
  OCR runs; do not modify Data-Science for GPU work, and do not touch `pyproject.toml` deps
  (CPU stays the declared default). Setup, in this order:
  ```bash
  conda create -y -p 'E:\Anaconda\Enviroment\riq-gpu' --clone 'E:\Anaconda\Enviroment\Data-Science'  # Windows paths; forward slashes fail
  G=/e/Anaconda/Enviroment/riq-gpu/python.exe
  $G -m pip uninstall -y paddlepaddle torch torchvision torchaudio
  $G -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/  # never PyPI (stale 2.x)
  $G -m pip install -e . --no-deps          # plain `-e .` would reinstall CPU paddlepaddle
  $G -c "import paddle; paddle.utils.run_check()"
  ```
  - torch must go: its bundled cuDNN and paddle-gpu's `nvidia-cudnn-cu12` cannot both load
    (WinError 127), and paddlex always imports torch via modelscope.
  - `--clone` reverts pip-upgraded packages to their conda versions (it reverted pyarrow 26 → 21,
    which fails to import, and brotlicffi 1.2 → 1.1). After cloning, diff
    `pip list --format=freeze` against Data-Science and pip-install (`--no-deps`) the
    Data-Science versions. After that, the only differences should be (checked 2026-11-16):
    - only in Data-Science: `paddlepaddle==3.3.1`, `torch==2.6.0+cu124`,
      `torchaudio==2.6.0+cu124`, `torchvision==0.21.0+cu124`
    - only in riq-gpu: `paddlepaddle-gpu==3.3.1`, `nvidia-cublas-cu12==12.6.4.1`,
      `nvidia-cuda-runtime-cu12==12.6.77`, `nvidia-cudnn-cu12==9.5.1.17`,
      `nvidia-cufft-cu12==11.3.0.4`, `nvidia-curand-cu12==10.3.7.77`,
      `nvidia-cusolver-cu12==11.7.1.2`, `nvidia-cusparse-cu12==12.5.4.2`,
      `nvidia-nvjitlink-cu12==12.9.86`
  - Paddle warns "compiled with CUDNN 9.9, but CUDNN version in your machine is 9.5". That is the
    official wheel's own pin. On 5 test keys GPU and CPU text matched exactly anyway.

## Commands

```bash
PY=/e/Anaconda/Enviroment/Data-Science/python.exe   # Git Bash path; bare `python` is the wrong (3.14) interpreter
$PY -m ruff format . && $PY -m ruff check .
$PY -m pytest                                       # unit tests; never load the OCR model or dataset
$PY -m pytest tests/test_ocr_schema.py::test_drops_blank_text
$PY scripts/download_sroie.py [--export-samples N] [--revision REV]   # ~1.1 GB into data/
$PY scripts/run_ocr_baseline.py data/samples/<key>.jpg              # -> outputs/ocr/<key>.json
$PY scripts/run_ocr_batch.py --split test [--limit N] [--out-dir D] # -> outputs/ocr/<split>/<key>.json + manifest.json; resumable
$PY scripts/eda_sroie.py [--seed 42 --n 20]                         # train-split EDA -> outputs/eda/, see docs/eda.md
$PY scripts/eval_ocr_detection.py --split test --ocr-dir outputs/ocr/test-gpu [--out F] [--iou 0.5]  # -> outputs/eval/detection_test.json
$PY scripts/eval_ocr_recognition.py --split test --ocr-dir outputs/ocr/test-gpu [--out F] [--iou 0.5]  # -> outputs/eval/recognition_test.json
$PY scripts/eval_ocr_e2e.py --split test --ocr-dir outputs/ocr/test-gpu [--out F]  # -> outputs/eval/e2e_test.json
$PY scripts/run_ocr_batch.py --split test --limit 40 --device cpu --out-dir outputs/ocr/test-cpu-latency  # CPU latency sample, ~20 min
$PY scripts/eval_ocr_confidence.py --split test [--ocr-dir outputs/ocr/test-gpu] [--latency-dir outputs/ocr/test-cpu-latency] [--out F] [--chart P] [--iou 0.5]  # -> outputs/eval/confidence_test.json + confidence_hist_test.png
$PY scripts/ocr_error_report.py --split test [--ocr-dir outputs/ocr/test-gpu] [--out F] [--iou 0.5] [--examples 3]  # -> outputs/eval/ocr_errors_test.json

G=/e/Anaconda/Enviroment/riq-gpu/python.exe        # GPU env, see Environment
$G -m pytest                                        # tests must pass in both envs
$G scripts/run_ocr_batch.py --split test --device gpu:0 [--out-dir D]   # ~0.5 s/image, ~1.2 GB VRAM
$G scripts/run_ocr_baseline.py data/samples/<key>.jpg --device gpu:0
```

`--device` is `cpu` (default) or `gpu:<int>`. A GPU device in the CPU env is a clear error,
not a silent CPU fallback. The batch refuses an `--out-dir` holding results from another
device or paddle distribution/version (see "No mixing" in `docs/baseline.md`).

The first OCR run downloads PP-OCRv6_medium det + rec (about 135 MB) to `~/.paddlex/official_models`.

## Architecture

- `ml/ocr_schema.py` is the engine-independent contract. `normalize_paddle_result` turns
  a PaddleOCR 3.x `predict()` item (dict-like, with `rec_texts` / `rec_scores` / `rec_polys`,
  numpy values) into `OcrLine`s, and `build_ocr_document` wraps them in the versioned JSON
  (`SCHEMA_VERSION`). It must not import paddle, so tests stay model-free. If the output
  shape changes, bump `SCHEMA_VERSION` and update `docs/baseline.md`.
- `OcrLine.bbox` uses `[x_min, y_min, x_max, y_max]` pixels on purpose, to match SROIE
  `bboxes`, so future evaluation can compare without conversion.
- `scripts/download_sroie.py` pins `jsdnrs/ICDAR2019-SROIE` to `PINNED_REVISION` and
  checks split sizes (626 train / 361 test) only at that revision. Everything goes under `data/`
  (`hf_cache/`, `sroie/` via `save_to_disk` + `manifest.json`, `samples/`). If the pin changes,
  update `EXPECTED_SPLIT_SIZES` and `docs/dataset.md` too.
- `ml/ocr_engine.py` holds `OCR_CONFIG`, `create_engine(device)` (imports paddle lazily,
  `paddleocr` before `paddle`, see the comment there) and `run_ocr(..., device=)`
  (encoded image bytes → OCR document, records `engine.device`). `engine_versions()` reports
  `paddlepaddle_distribution` (`paddlepaddle` or `paddlepaddle-gpu`) and its version. `run_ocr` decodes with `cv2.imdecode` like PaddleX does
  for a file path; do not switch it to PIL, which decodes JPEG differently and changes OCR output. Both OCR scripts use it; do not configure PaddleOCR anywhere else.
  `scripts/run_ocr_baseline.py` processes exactly one image. `scripts/run_ocr_batch.py` runs a
  split from `data/sroie`, skips keys that already have JSON (atomic tmp+rename writes) and keeps
  going on per-image failures (exit 1 at the end). Use the test split (361) for evaluation.
- `ml/ocr_eval.py` (RIQ-19) is pure-Python detection metrics: `iou`, greedy `match_greedy`
  (IoU ≥ 0.5 inclusive, IoU desc, ties by pred/GT index), `scores` (null on a zero denominator),
  `micro` (summed counts). Like `ocr_schema.py` it must not import paddle, cv2 or datasets.
  `scripts/eval_ocr_detection.py` reads GT `key`/`bboxes`/`image_size` only; keys without OCR
  JSON are listed as missing, never scored as all-FN. GT often splits one printed line into
  several boxes (label / value, table columns; not at commas) while OCR returns whole lines, so
  line-level IoU undercounts.
  The canonical full-test GPU output is `outputs/ocr/test-gpu/`; `outputs/ocr/test/` holds only a
  stale 2-row CPU manifest from RIQ-18 (leave it).
  SROIE ground truth has typos (e.g. `X00016469670` address says `B1750`, receipt shows `81750`),
  so when OCR and ground truth disagree, look at the image before blaming the model.
- `ml/text_norm.py` (RIQ-20) holds `normalize_text` (uppercase, collapse whitespace, strip), the
  ONE text normalization; field evaluation (E3 / RIQ-29) must import it, never copy it.
  `ml/ocr_recognition.py` is pure-Python `edit_distance` (Levenshtein, code points), `line_pairs`
  (CER / exact match on `match_greedy` pairs; empty normalized GT → CER None) and `summarize`.
  `scripts/eval_ocr_recognition.py` reuses `load_ocr` / `load_gt_rows` / `resolve_device` from
  `scripts/eval_ocr_detection.py` (same input contract) and reads GT `key`/`words`/`bboxes`/`image_size`.
  `ml/ocr_e2e.py` (RIQ-21) is pure-Python word-level P/R/F1: `normalize_text` tokens split on
  whitespace, `Counter` intersection per image, `micro_prf1` over summed counts, and None
  (JSON `null`) on a zero denominator. `scripts/eval_ocr_e2e.py` uses the same loaders,
  reads GT `key`/`words`/`image_size`. Field metrics are not implemented yet.
  `ml/ocr_confidence.py` (RIQ-22) is pure-Python (no paddle, cv2, datasets or PIL):
  `latency_stats` (p95 nearest-rank via `percentile`), `is_comma_cut` (suspected GT comma-cut
  pair), `histogram`, `threshold_sweep` over `THRESHOLDS` (flag if confidence < t, positive =
  wrong line, reuses `ml.ocr_e2e.prf1`), `pick_threshold` (max F1, ties to lower t) and
  `pick_budget` (highest t flagging ≤ a share of lines; the CLI uses 15%).
  `scripts/eval_ocr_confidence.py` runs it on `line_pairs` with the same loaders, reports
  latency for `--ocr-dir` and the optional `--latency-dir` CPU sample, and draws the chart with
  Pillow inside the script (never add matplotlib).
  `ml/ocr_errors.py` (RIQ-23) is pure-Python (same ban): `error_report` excludes GT artefacts in
  order (`is_comma_cut`, `is_comma_cut_variant`, `is_known_gt_typo`), then `classify` gives each
  wrong pair one heuristic category from a difflib diff of the space-free normalized texts
  (`spacing`, `punctuation`, `symbol`, `digit_letter`, `char_substitution`, `extra_text`,
  `missing_text`, `inner_insert_delete`, `mixed`). Add a token to `KNOWN_GT_TYPOS` only after
  checking the receipt image. `scripts/ocr_error_report.py` builds `line_pairs` with the same
  loaders. The results write-up is "Results (RIQ-23)" in `docs/baseline.md`.
- `ml/reading_order.py` (RIQ-24) is pure Python (no paddle, cv2, datasets, PIL or numpy):
  `group_rows(lines)` turns `OcrLine`s into frozen `OcrRow`s (`text` joined by one space in x
  order, union `bbox` tuple, `source_indices` into the input, `confidences`), rows top to bottom.
  Lines are visited by (y_min, x_min, index) and join the current row only if their vertical
  overlap with EVERY member is ≥ `ROW_OVERLAP_MIN` (0.5) of the smaller height; testing the
  growing union box instead let tall stamp boxes and skewed scans chain rows. Text stays raw
  (no `normalize_text`). Field extractors (RIQ-25..27) should cite `source_indices` as evidence.
  Known limits: two different columns at the same height still merge into one row
  (`Salesperson:` + `Date. 09/02/2018`); grouping is geometric by design and telling fields
  apart belongs to the extractors. A tall box joins one row only, and strong skew can still pair
  the wrong columns. `ROW_OVERLAP_MIN` = 0.5 is untuned; tune it after the RIQ-29 field metrics
  exist, on the train split, never on test.
- Open GT data bug (separate from box splitting): in both splits many GT `words` are cut at the
  first comma while the box spans the whole printed line (GT `9` for `9, JALAN SUBANG JASA 3,`,
  `1` for `1,007.50`); no `words` element contains a comma. It inflates CER and will affect
  end-to-end metrics; `entities` are intact. Likely a converter that split `x1,…,y4,TEXT` on every
  comma. Do not patch GT or change metric rules for it outside its bug ticket (RIQ-60); see
  "GT text cut at the first comma" in `docs/baseline.md`.

## Rules specific to this repo

- `data/`, `outputs/`, image files, model weights (`*.pdparams` etc.) and `.env` are
  git-ignored. Never commit datasets, receipt images or weights.
- End-of-sprint QA reports go in `reports/sprint-<N>/` (report `.docx` + `logs/` with every
  command output it cites), one folder per sprint. `reports/` is git-ignored.
- SROIE is CC BY 4.0. Any shown or redistributed derived material needs the attribution
  in `docs/dataset.md`.
- Do not claim Vietnamese support or suitability for accounting/tax use anywhere.
  The data is English Malaysian scanned receipts (2016–2018).

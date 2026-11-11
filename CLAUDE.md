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
- `run_ocr_baseline.py` sets `enable_mkldnn=False`. Paddle 3.3.1 oneDNN crashes on Windows CPU
  (`ConvertPirAttribute2RuntimeAttribute`). About 18 s/image on CPU as a result.
- `brotlicffi>=1.2` is required in that env, otherwise `huggingface_hub` downloads crash in
  `httpx2` with `unexpected keyword argument 'output_buffer_limit'`.

## Commands

```bash
PY=/e/Anaconda/Enviroment/Data-Science/python.exe   # Git Bash path; bare `python` is the wrong (3.14) interpreter
$PY -m ruff format . && $PY -m ruff check .
$PY -m pytest                                       # unit tests; never load the OCR model or dataset
$PY -m pytest tests/test_ocr_schema.py::test_drops_blank_text
$PY scripts/download_sroie.py [--export-samples N] [--revision REV]   # ~1.1 GB into data/
$PY scripts/run_ocr_baseline.py data/samples/<key>.jpg              # -> outputs/ocr/<key>.json
$PY scripts/eda_sroie.py [--seed 42 --n 20]                         # train-split EDA -> outputs/eda/, see docs/eda.md
```

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
- `scripts/run_ocr_baseline.py` processes exactly one image. Evaluation and metrics are
  planned in `docs/baseline.md` but not implemented yet. Use the test split (361) for evaluation.
  SROIE ground truth has typos (e.g. `X00016469670` address says `B1750`, receipt shows `81750`),
  so when OCR and ground truth disagree, look at the image before blaming the model.

## Rules specific to this repo

- `data/`, `outputs/`, image files, model weights (`*.pdparams` etc.) and `.env` are
  git-ignored. Never commit datasets, receipt images or weights.
- SROIE is CC BY 4.0. Any shown or redistributed derived material needs the attribution
  in `docs/dataset.md`.
- Do not claim Vietnamese support or suitability for accounting/tax use anywhere.
  The data is English Malaysian scanned receipts (2016–2018).

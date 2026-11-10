# ReceiptIQ

ReceiptIQ is a personal research project on reading receipts with OCR and pulling
out key fields (`company`, `date`, `address`, `total`). The planned MVP lets a user
upload a receipt, see the extracted fields with confidence scores, and correct them.

**Current stage: phase 1, research foundation and OCR baseline.** There is no API,
UI, database or deployment yet, and no model has been trained.

## Requirements

- Python **3.13** (the only version this baseline is tested on; PaddlePaddle wheels lag new Python releases)
- CPU is enough. First OCR run downloads about 135 MB of PaddleOCR models (PP-OCRv6_medium det + rec) to `~/.paddlex`
- About 1.1 GB disk for SROIE (HF cache + `save_to_disk` copy, both in `data/`)

## Setup

Any Python 3.13 environment works (venv, conda, uv):

```bash
# example with conda
conda create -n receiptiq python=3.13 && conda activate receiptiq
pip install -e ".[dev]"
```

Do not install an extra `opencv-python*` wheel. PaddleOCR already brings
`opencv-contrib-python`, and a second OpenCV build would overwrite the same `cv2` package.

If Hugging Face downloads fail with
`Decompressor.decompress() got an unexpected keyword argument 'output_buffer_limit'`,
upgrade brotli support: `pip install -U "brotlicffi>=1.2"`.

## Run the baseline

```bash
# 1. Download SROIE (pinned revision) to data/ and export 3 test images
python scripts/download_sroie.py

# 2. OCR one image -> outputs/ocr/<name>.json
python scripts/run_ocr_baseline.py data/samples/<key>.jpg
```

## Develop

```bash
ruff format . && ruff check .
pytest                                     # unit tests, no OCR model needed
pytest tests/test_ocr_schema.py -k bbox    # single test
```

## Layout

```
app/       application layer (empty until the MVP)
ml/        OCR result schema / normalization; field extraction later
scripts/   CLI entry points: dataset download, single-image OCR baseline
tests/     unit tests (no model, no dataset)
docs/      dataset.md (source, license, limits) and baseline.md (metrics plan)
data/      dataset + samples (git-ignored, created by scripts)
outputs/   OCR results (git-ignored)
```

## Current limitations

- OCR runs on one image per call. Batch evaluation and metrics are planned, see [docs/baseline.md](docs/baseline.md).
- Field extraction is not implemented yet.
- English-only models, evaluated only on Malaysian scanned receipts (2016–2018).
  Vietnamese receipts are **not** supported.
- Results are not validated for accounting or tax purposes.
- Confidence scores come straight from the OCR engine and are not calibrated.

## Data & attribution

Uses [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) (Huang et al., 2019) via
Hugging Face [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is
not included in this repository. See [docs/dataset.md](docs/dataset.md).

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

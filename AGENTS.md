# Repository Guidelines

ReceiptIQ is a phase-1 research baseline: OCR + key-field extraction on ICDAR 2019 SROIE
(English Malaysian scanned receipts). Keep work inside that phase unless the task explicitly
expands scope: no API, database, frontend, deployment or model training.

`CLAUDE.md` is the single source of truth for environment setup, the GPU env recipe and
architecture notes. Read it before running anything. `docs/baseline.md` defines the OCR JSON
schema, the batch/resume/no-mixing rules and the detection metric; `docs/dataset.md` the data
provenance and CC BY 4.0 attribution.

## Layout

- `ml/`: `ocr_schema.py` (versioned OCR JSON contract), `ocr_engine.py` (the only place
  PaddleOCR is configured: `OCR_CONFIG`, `create_engine(device)`, `run_ocr`), `ocr_eval.py`
  (IoU, greedy matching, P/R/F1), `text_norm.py` (`normalize_text`, the one text
  normalization), `ocr_recognition.py` (edit distance, CER, line exact match). These four
  stay pure Python: they never import paddle, cv2 or datasets.
- `scripts/`: download SROIE, single-image OCR, resumable batch OCR, detection and
  recognition eval, EDA.
- `tests/`: fast, model-free pytest. `docs/`: provenance, plans, findings.
- `data/` and `outputs/` are generated locally and stay untracked.

## Environments and commands

Python 3.13 only. Call an env's interpreter directly; bare `python` on this machine is 3.14
with no deps.

```bash
PY=/e/Anaconda/Enviroment/Data-Science/python.exe   # CPU env, the declared default
G=/e/Anaconda/Enviroment/riq-gpu/python.exe         # GPU env (paddlepaddle-gpu, no torch)
$PY -m ruff format . && $PY -m ruff check . && $PY -m pytest   # run pytest in both envs
$PY scripts/run_ocr_baseline.py data/samples/<key>.jpg [--device cpu]
$G  scripts/run_ocr_batch.py --split test --device gpu:0 --out-dir outputs/ocr/test-gpu
$PY scripts/eval_ocr_detection.py --split test --ocr-dir outputs/ocr/test-gpu
$PY scripts/eval_ocr_recognition.py --split test --ocr-dir outputs/ocr/test-gpu
```

- `--device` is `cpu` (default) or `gpu:<int>`. Each OCR JSON records `engine.device`; the
  batch `manifest.json` records `ocr.device` and the paddle distribution/version.
- One out-dir holds results from one device and one paddle build. The batch refuses a mixed
  out-dir; pick a new `--out-dir` instead of deleting results.
- The batch resumes by skipping keys that already have `<key>.json`. A run on the 4 GB GPU
  can die on the largest scans; rerun the same command to resume.
- Canonical full test-split OCR output: `outputs/ocr/test-gpu/`. `outputs/ocr/test/` is a stale
  2-row CPU manifest from RIQ-18; leave it.

## Coding and tests

Four-space indentation, type hints on public interfaces, one-line comments for the non-obvious
why. Ruff enforces a 100-character line limit. Tests live in `tests/test_<feature>.py` as
`test_<behavior>` and cover success, empty and malformed input. Tests use fakes, `tmp_path`
fixtures and injectable loaders, so they run without the OCR model, a GPU or `data/sroie`.
If the OCR JSON shape changes, bump `SCHEMA_VERSION` and update `docs/baseline.md`.

## Commits and pull requests

Prefix commits and PR titles with the Jira key: `RIQ-19: OCR detection metrics (...)`.
Branch from `develop` as `feature/RIQ-<n>-<slug>` and open PRs into `develop`. A PR links
the issue, explains the change and its limits, lists the verification commands run, and calls
out schema, dataset or dependency changes.

## Data handling

Commit code, synthetic fixtures and docs only; datasets, receipt images, OCR output, model
weights and `.env` stay local. Keep `pyproject.toml` dependencies as they are and keep the
single OpenCV wheel PaddleOCR installs (`opencv-contrib-python`). Treat OCR fields as
uncertain candidates, use exact decimals for money, and describe the data honestly: English
Malaysian receipts, no Vietnamese support, not validated for accounting or tax use.

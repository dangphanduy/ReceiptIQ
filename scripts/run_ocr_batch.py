"""Run PaddleOCR over a SROIE split from data/sroie, one JSON per receipt, resumable.

Usage:
    python scripts/run_ocr_batch.py --split test --limit 2
    python scripts/run_ocr_batch.py --split test            # all 361 rows, ~3.5 h on CPU

Writes <out-dir>/<key>.json (schema: ml/ocr_schema.py) and <out-dir>/manifest.json.
Default out-dir: outputs/ocr/<split>/. A rerun skips keys whose <key>.json exists.
"""

import argparse
import itertools
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from ml.ocr_engine import OCR_CONFIG, create_engine, engine_versions, run_ocr

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sroie"
SAFE_KEY = re.compile(r"[\w.-]+", re.ASCII)  # same charset download_sroie keeps in filenames


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, obj) -> None:
    # Rename is atomic, so an interrupted run never leaves a truncated <key>.json behind.
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run_batch(
    rows: Iterable[Mapping],
    ocr: Callable[[bytes, str], dict],
    out_dir: Path,
    *,
    split: str,
    dataset_manifest: Mapping,
    limit: int | None = None,
) -> dict:
    """OCR the first `limit` rows (all if None) that have no output yet; return the manifest.

    Rows look like data/sroie with the image column undecoded: {"key", "image": {"bytes"}}.
    """
    started_at = _now()
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = skipped = 0
    failed = []
    for row in itertools.islice(rows, limit):
        key = row["key"]
        out_path = out_dir / f"{key}.json"
        if SAFE_KEY.fullmatch(key) and out_path.exists():
            skipped += 1
            continue
        try:
            if not SAFE_KEY.fullmatch(key):  # key is remote data; never let it leave out_dir
                raise ValueError(f"unsafe key {key!r}")
            _write_json_atomic(out_path, ocr(row["image"]["bytes"], f"sroie:{split}/{key}"))
        except Exception as exc:  # one bad image must not stop a multi-hour run
            error = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {key}: {error}", file=sys.stderr, flush=True)
            failed.append({"key": key, "error": error})
            continue
        processed += 1
        print(f"[{processed + skipped + len(failed)}] {key}", flush=True)

    manifest = {
        "dataset": {
            "repo_id": dataset_manifest["repo_id"],
            "revision": dataset_manifest["commit_sha"],
        },
        "split": split,
        "limit": limit,
        "ocr": {"engine": "paddleocr", **engine_versions(), "device": "cpu", "config": OCR_CONFIG},
        "counts": {"processed": processed, "skipped": skipped, "failed": len(failed)},
        "failed": failed,
        "started_at": started_at,
        "finished_at": _now(),
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=_positive_int, help="only the first N rows of the split")
    parser.add_argument("--out-dir", type=Path, help="default: outputs/ocr/<split>/")
    args = parser.parse_args()
    out_dir = args.out_dir or ROOT / "outputs" / "ocr" / args.split

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"{manifest_path} not found; run scripts/download_sroie.py first")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    from datasets import Image, load_from_disk

    # Original JPEG bytes, decoded later the same way PaddleX decodes a file (see run_ocr).
    rows = load_from_disk(str(DATA_DIR))[args.split].cast_column("image", Image(decode=False))
    engine = create_engine()  # once per run; model init is slow
    manifest = run_batch(
        rows,
        lambda image, path: run_ocr(engine, image, path),
        out_dir,
        split=args.split,
        dataset_manifest=dataset_manifest,
        limit=args.limit,
    )

    counts = manifest["counts"]
    print(
        f"processed {counts['processed']}, skipped {counts['skipped']}, "
        f"failed {counts['failed']} -> {out_dir}"
    )
    if manifest["failed"]:
        print("failed keys: " + ", ".join(f["key"] for f in manifest["failed"]), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

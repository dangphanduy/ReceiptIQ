"""OCR detection metrics on a SROIE split: greedy IoU matching, micro + per-image P/R/F1.

Usage:
    python scripts/eval_ocr_detection.py --split test --ocr-dir outputs/ocr/test-gpu
    python scripts/eval_ocr_detection.py --split test --iou 0.5 --out outputs/eval/det.json

Reads GT (`key`, `bboxes`, `image_size` only; images are never decoded) from data/sroie and
<ocr-dir>/<key>.json from run_ocr_batch.py. Keys without OCR JSON are listed in
"missing_keys" and left out of the metrics, never counted as all-FN.
"""

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from ml.ocr_eval import detection_counts, micro, scores

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sroie"
SAFE_KEY = re.compile(r"[\w.-]+", re.ASCII)  # same rule as run_ocr_batch.py


def evaluate(gt_rows: Iterable[Mapping], ocr_dir: Path, iou_threshold: float) -> dict:
    """Match each GT row with <ocr_dir>/<key>.json; return counts, micro and per-image scores.

    Raises ValueError for an unsafe key, an unreadable or malformed <key>.json (naming the
    file), or OCR JSON with more than one schema_version or device.
    """
    per_image, missing, size_mismatch = [], [], []
    schemas, devices = set(), set()
    for row in gt_rows:
        key = row["key"]
        if not SAFE_KEY.fullmatch(key):
            raise ValueError(f"unsafe key {key!r}")
        path = ocr_dir / f"{key}.json"
        if not path.exists():
            missing.append(key)
            continue
        try:  # a broken file fails the run with its name; never skip a key silently
            doc = json.loads(path.read_text(encoding="utf-8"))
            schema, device = doc["schema_version"], doc["engine"]["device"]
            ocr_size = (doc["image"]["width"], doc["image"]["height"])
            pred_boxes = [ln["bbox"] for ln in doc["lines"]]
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            raise ValueError(f"{path}: malformed OCR JSON ({type(exc).__name__}: {exc})") from exc
        schemas.add(schema)
        devices.add(device)
        size = row["image_size"]
        if ocr_size != (size["width"], size["height"]):
            size_mismatch.append(key)
        counts = detection_counts(pred_boxes, row["bboxes"], iou_threshold)
        per_image.append({"key": key, **counts, **scores(**counts)})
    if len(schemas) > 1:
        raise ValueError(f"OCR JSON in {ocr_dir} mixes schema versions {sorted(schemas)}")
    if len(devices) > 1:
        raise ValueError(f"OCR JSON in {ocr_dir} mixes devices {sorted(devices)}")
    return {
        "ocr_schema_version": next(iter(schemas), None),
        "ocr_devices": sorted(devices),
        "counts": {"evaluated": len(per_image), "missing": len(missing)},
        "micro": micro(per_image),
        "missing_keys": missing,
        "size_mismatch_keys": size_mismatch,
        "per_image": per_image,
    }


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _iou_threshold(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0 < x <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--ocr-dir", type=Path, help="default: outputs/ocr/<split>/")
    parser.add_argument("--out", type=Path, help="default: outputs/eval/detection_<split>.json")
    parser.add_argument("--iou", type=_iou_threshold, default=0.5, help="match if IoU >= this")
    args = parser.parse_args()
    ocr_dir = args.ocr_dir or ROOT / "outputs" / "ocr" / args.split
    out = args.out or ROOT / "outputs" / "eval" / f"detection_{args.split}.json"

    dataset_manifest_path = DATA_DIR / "manifest.json"
    if not dataset_manifest_path.exists():
        sys.exit(f"{dataset_manifest_path} not found; run scripts/download_sroie.py first")
    if not any(p.name != "manifest.json" for p in ocr_dir.glob("*.json")):
        sys.exit(f"no OCR JSON in {ocr_dir}; run scripts/run_ocr_batch.py first")
    revision = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))["commit_sha"]

    from datasets import load_from_disk

    rows = load_from_disk(str(DATA_DIR))[args.split].select_columns(["key", "bboxes", "image_size"])
    try:
        result = evaluate(rows, ocr_dir, args.iou)
    except ValueError as exc:
        sys.exit(str(exc))
    if result["counts"]["evaluated"] == 0:  # e.g. only manifest.json / unrelated *.json
        sys.exit(f"no OCR JSON for any {args.split} key in --ocr-dir {ocr_dir}; no report written")

    # The batch writes its manifest only at the end; until then every <key>.json names the device.
    ocr_manifest = ocr_dir / "manifest.json"
    devices = result.pop("ocr_devices")
    if ocr_manifest.exists():
        device = json.loads(ocr_manifest.read_text(encoding="utf-8"))["ocr"]["device"]
        if devices and devices != [device]:
            sys.exit(f"{ocr_manifest} says device {device}, but OCR JSON says {devices}")
    else:
        device = devices[0] if devices else None

    report = {
        "iou_threshold": args.iou,
        "split": args.split,
        "sroie_revision": revision,
        "ocr_dir": str(ocr_dir),
        "ocr_device": device,
        **result,
    }
    _write_json_atomic(out, report)
    m, c = report["micro"], report["counts"]

    def fmt(v):
        return "null" if v is None else f"{v:.4f}"

    print(
        f"evaluated {c['evaluated']}, missing {c['missing']}, "
        f"size mismatch {len(report['size_mismatch_keys'])} | IoU >= {args.iou}: "
        f"P {fmt(m['precision'])} R {fmt(m['recall'])} F1 {fmt(m['f1'])} "
        f"(TP {m['tp']} FP {m['fp']} FN {m['fn']}) -> {out}"
    )


if __name__ == "__main__":
    main()

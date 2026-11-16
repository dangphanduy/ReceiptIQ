"""Run PaddleOCR 3.x (CPU) on ONE receipt image and write normalized JSON.

Usage:
    python scripts/run_ocr_baseline.py data/samples/<key>.jpg
    python scripts/run_ocr_baseline.py path/to/receipt.jpg --out-dir outputs/ocr

Output: <out-dir>/<image stem>.json (default out-dir `outputs/ocr`, git-ignored).
Schema: see ml/ocr_schema.py. First run downloads PaddleOCR models to ~/.paddlex.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ml.ocr_engine import create_engine, run_ocr

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "ocr")
    args = parser.parse_args()

    try:
        with Image.open(args.image) as im:
            im.verify()
        image_bytes = args.image.read_bytes()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        sys.exit(f"cannot read image {args.image}: {exc}")

    doc = run_ocr(create_engine(), image_bytes, str(args.image))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.image.stem}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    mean = doc["mean_confidence"]
    print(
        f"{doc['num_lines']} lines, mean confidence {mean:.3f}" if mean is not None else "0 lines",
        f"in {doc['engine']['inference_seconds']:.1f}s -> {out_path}",
    )


if __name__ == "__main__":
    main()

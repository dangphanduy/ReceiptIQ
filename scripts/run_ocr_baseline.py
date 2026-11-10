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
import time
from importlib.metadata import version
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ml.ocr_schema import build_ocr_document, normalize_paddle_result

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "ocr")
    args = parser.parse_args()

    try:
        with Image.open(args.image) as im:
            im.verify()
        with Image.open(args.image) as im:
            width, height = im.size
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        sys.exit(f"cannot read image {args.image}: {exc}")

    from paddleocr import PaddleOCR  # slow import; keep after input validation

    # Scanned receipts are upright and flat; disable the extra doc-preprocessing models.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # ponytail: paddlepaddle 3.3.1 oneDNN CPU kernels crash on PIR models on Windows
        # (ConvertPirAttribute2RuntimeAttribute). Slower without it; retry True on upgrades.
        enable_mkldnn=False,
    )
    start = time.perf_counter()
    results = ocr.predict(str(args.image))
    elapsed = time.perf_counter() - start

    lines = normalize_paddle_result(results[0])
    engine = {
        "name": "paddleocr",
        "paddleocr_version": version("paddleocr"),
        "paddlepaddle_version": version("paddlepaddle"),
        "lang": "en",
        "device": "cpu",
        "inference_seconds": round(elapsed, 3),
    }
    doc = build_ocr_document(str(args.image), width, height, lines, engine)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.image.stem}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    mean = doc["mean_confidence"]
    print(
        f"{len(lines)} lines, mean confidence {mean:.3f}" if mean is not None else "0 lines",
        f"in {elapsed:.1f}s -> {out_path}",
    )


if __name__ == "__main__":
    main()

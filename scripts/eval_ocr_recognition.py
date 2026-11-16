"""OCR recognition metrics on a SROIE split: CER and line exact match on detection-matched pairs.

Usage:
    python scripts/eval_ocr_recognition.py --split test --ocr-dir outputs/ocr/test-gpu
    python scripts/eval_ocr_recognition.py --split test --iou 0.5 --out outputs/eval/rec.json

Pairs are ml.ocr_eval.match_greedy matches (IoU >= --iou) between OCR lines and GT boxes; the
GT text of bboxes[i] is words[i]. Same input contract as eval_ocr_detection.py: GT columns
`key`, `words`, `bboxes`, `image_size` only (images are never decoded); keys without OCR JSON
are listed in "missing_keys", never scored. GT often uses several boxes per printed line;
separately, some GT text is cut at the first comma even when its box spans the whole line.
"""

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from eval_ocr_detection import (
    ROOT,
    load_gt_rows,
    load_ocr,
    parse_iou_threshold,
    resolve_device,
    write_json_atomic,
)

from ml.ocr_recognition import line_pairs, summarize

GT_COLUMNS = ["key", "words", "bboxes", "image_size"]
NORMALIZATION = (
    "ml.text_norm.normalize_text: uppercase, collapse whitespace runs to one space, strip; "
    "CER = edit_distance / len(normalized GT); pairs with empty normalized GT skip CER"
)


def _ocr_lines(doc: dict) -> list[tuple[list, str]]:
    lines = [(ln["bbox"], ln["text"]) for ln in doc["lines"]]
    for _, text in lines:
        if type(text) is not str:
            raise TypeError(f"line text must be str, got {type(text).__name__}")
    return lines


def evaluate(gt_rows: Iterable[Mapping], ocr_dir: Path, iou_threshold: float) -> dict:
    """Recognition metrics over all keys with OCR JSON.

    Raises ValueError as eval_ocr_detection.load_ocr does, or for a GT row whose words and
    bboxes differ in length.
    """
    loaded = load_ocr(gt_rows, ocr_dir, _ocr_lines)
    pairs = []
    for row, preds in loaded["images"]:
        words, bboxes = row["words"], row["bboxes"]
        if len(words) != len(bboxes):
            raise ValueError(f"GT {row['key']}: {len(words)} words but {len(bboxes)} bboxes")
        for i, w in enumerate(words):
            if type(w) is not str:
                raise ValueError(f"GT {row['key']}: words[{i}] is {type(w).__name__}, not str")
        pairs += line_pairs(row["key"], preds, list(zip(bboxes, words, strict=True)), iou_threshold)
    summary = summarize(pairs)
    return {
        "ocr_schema_version": loaded["ocr_schema_version"],
        "ocr_devices": loaded["ocr_devices"],
        "counts": {
            "images_evaluated": len(loaded["images"]),
            "missing": len(loaded["missing_keys"]),
            **summary["counts"],
        },
        "cer": summary["cer"],
        "exact_match_rate": summary["exact_match_rate"],
        "worst_lines": summary["worst_lines"],
        "missing_keys": loaded["missing_keys"],
        "size_mismatch_keys": loaded["size_mismatch_keys"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--ocr-dir", type=Path, help="default: outputs/ocr/<split>/")
    parser.add_argument("--out", type=Path, help="default: outputs/eval/recognition_<split>.json")
    parser.add_argument("--iou", type=parse_iou_threshold, default=0.5, help="pair if IoU >= this")
    args = parser.parse_args()
    ocr_dir = args.ocr_dir or ROOT / "outputs" / "ocr" / args.split
    out = args.out or ROOT / "outputs" / "eval" / f"recognition_{args.split}.json"

    rows, revision = load_gt_rows(args.split, GT_COLUMNS, ocr_dir)
    try:
        result = evaluate(rows, ocr_dir, args.iou)
    except ValueError as exc:
        sys.exit(str(exc))
    if result["counts"]["images_evaluated"] == 0:
        sys.exit(f"no OCR JSON for any {args.split} key in --ocr-dir {ocr_dir}; no report written")
    device = resolve_device(ocr_dir, result.pop("ocr_devices"))

    report = {
        "iou_threshold": args.iou,
        "split": args.split,
        "sroie_revision": revision,
        "ocr_dir": str(ocr_dir),
        "ocr_device": device,
        "ocr_schema_version": result.pop("ocr_schema_version"),
        "normalization": NORMALIZATION,
        **result,
    }
    write_json_atomic(out, report)
    c, cer = report["counts"], report["cer"]

    def fmt(v):
        return "null" if v is None else f"{v:.4f}"

    print(
        f"evaluated {c['images_evaluated']}, missing {c['missing']}, "
        f"size mismatch {len(report['size_mismatch_keys'])} | IoU >= {args.iou}: "
        f"pairs {c['pairs']} (CER on {c['cer_pairs']}, empty GT {c['skipped_empty_gt']}) "
        f"CER mean {fmt(cer['mean'])} median {fmt(cer['median'])} "
        f"exact {fmt(report['exact_match_rate'])} -> {out}"
    )


if __name__ == "__main__":
    main()

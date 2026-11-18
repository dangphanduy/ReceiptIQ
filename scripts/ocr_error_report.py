"""OCR error categories on a SROIE split: counts and example keys for wrong line pairs (RIQ-23).

Usage:
    python scripts/ocr_error_report.py --split test --ocr-dir outputs/ocr/test-gpu
    python scripts/ocr_error_report.py --split test --examples 3 --out outputs/eval/errors.json

Pairs are ml.ocr_recognition.line_pairs (IoU >= --iou), the same as recognition. GT comma cuts
and known GT typos are excluded, then each wrong pair gets one heuristic category
(ml/ocr_errors.py). Same input contract as eval_ocr_recognition.py: GT columns `key`, `words`,
`bboxes`, `image_size` only; keys without OCR JSON are listed in "missing_keys", never scored.
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
from eval_ocr_recognition import GT_COLUMNS, _ocr_lines

from ml.ocr_errors import error_report
from ml.ocr_recognition import line_pairs


def evaluate(
    gt_rows: Iterable[Mapping], ocr_dir: Path, iou_threshold: float, n_examples: int
) -> dict:
    """Error report over all keys with OCR JSON. Raises ValueError as load_ocr does, or for a GT
    row with a non-str word or words and bboxes of different length."""
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
    return {
        "images_evaluated": len(loaded["images"]),
        "ocr_device": resolve_device(ocr_dir, loaded["ocr_devices"]),
        "ocr_schema_version": loaded["ocr_schema_version"],
        **error_report(pairs, n_examples),
        "missing_keys": loaded["missing_keys"],
        "size_mismatch_keys": loaded["size_mismatch_keys"],
    }


def _print_summary(report: dict, images: int, out: Path) -> None:
    c = report["counts"]
    print(
        f"evaluated {images}, missing {len(report['missing_keys'])} | pairs {c['pairs']}, "
        f"wrong {c['wrong']}, excluded: comma-cut {c['excluded_comma_cut']}, "
        f"comma-cut variant {c['excluded_comma_cut_variant']}, "
        f"known GT typo {c['excluded_known_gt_typo']} | classified {c['classified']}"
    )
    for cat in report["categories"]:
        print(f"  {cat['category']:<20} {cat['count']:>5}  {cat['share']:6.1%}")
        for e in cat["examples"]:
            print(f"      {e['key']}[{e['gt_idx']}]  GT {e['gt_text']!r}  OCR {e['pred_text']!r}")
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--ocr-dir", type=Path, help="default: outputs/ocr/<split>-gpu/")
    parser.add_argument("--out", type=Path, help="default: outputs/eval/ocr_errors_<split>.json")
    parser.add_argument("--iou", type=parse_iou_threshold, default=0.5, help="pair if IoU >= this")
    parser.add_argument("--examples", type=int, default=3, help="example keys per category")
    args = parser.parse_args()
    ocr_dir = args.ocr_dir or ROOT / "outputs" / "ocr" / f"{args.split}-gpu"
    out = args.out or ROOT / "outputs" / "eval" / f"ocr_errors_{args.split}.json"

    rows, revision = load_gt_rows(args.split, GT_COLUMNS, ocr_dir)
    try:
        result = evaluate(rows, ocr_dir, args.iou, args.examples)
    except ValueError as exc:
        sys.exit(str(exc))
    images = result.pop("images_evaluated")
    if images == 0:
        sys.exit(f"no OCR JSON for any {args.split} key in --ocr-dir {ocr_dir}; no report written")
    report = {
        "split": args.split,
        "sroie_revision": revision,
        "iou_threshold": args.iou,
        "ocr_dir": str(ocr_dir),
        **result,
    }
    write_json_atomic(out, report)
    _print_summary(report, images, out)


if __name__ == "__main__":
    main()

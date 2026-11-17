"""OCR end-to-end metrics on a SROIE split: word-level P/R/F1, micro + per image.

Usage:
    python scripts/eval_ocr_e2e.py --split test --ocr-dir outputs/ocr/test-gpu
    python scripts/eval_ocr_e2e.py --split test --out outputs/eval/e2e_test.json

Per image, all OCR line texts and all GT `words` are normalized (ml.text_norm), split on
whitespace and matched as a multiset; boxes are not used. Same input contract as
eval_ocr_detection.py: GT columns `key`, `words`, `image_size` only (images are never
decoded); keys without OCR JSON are listed in "missing_keys", never scored. Some GT `words`
are cut at the first comma (see docs/baseline.md), which can increase FP.
"""

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from eval_ocr_detection import ROOT, load_gt_rows, load_ocr, resolve_device, write_json_atomic

from ml.ocr_e2e import micro_prf1, prf1, word_counts

GT_COLUMNS = ["key", "words", "image_size"]
NORMALIZATION = (
    "ml.text_norm.normalize_text, then split on whitespace: uppercase, no punctuation removal; "
    "tokens matched as a multiset per image; each zero-denominator metric is null"
)


def _ocr_texts(doc: dict) -> list[str]:
    texts = [ln["text"] for ln in doc["lines"]]
    for text in texts:
        if type(text) is not str:
            raise TypeError(f"line text must be str, got {type(text).__name__}")
    return texts


def evaluate(gt_rows: Iterable[Mapping], ocr_dir: Path) -> dict:
    """Word counts per image with OCR JSON, and micro scores over them.

    Raises ValueError as eval_ocr_detection.load_ocr does, or for a non-str GT word.
    """
    loaded = load_ocr(gt_rows, ocr_dir, _ocr_texts)
    per_image = []
    for row, pred_texts in loaded["images"]:
        for i, w in enumerate(row["words"]):
            if type(w) is not str:
                raise ValueError(f"GT {row['key']}: words[{i}] is {type(w).__name__}, not str")
        counts = word_counts(pred_texts, row["words"])
        per_image.append({"key": row["key"], **counts, **prf1(**counts)})
    return {
        "ocr_schema_version": loaded["ocr_schema_version"],
        "ocr_devices": loaded["ocr_devices"],
        "counts": {"evaluated": len(per_image), "missing": len(loaded["missing_keys"])},
        "micro": micro_prf1(per_image),
        "missing_keys": loaded["missing_keys"],
        "size_mismatch_keys": loaded["size_mismatch_keys"],
        "per_image": per_image,
    }


def _score_text(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--ocr-dir", type=Path, help="default: outputs/ocr/<split>/")
    parser.add_argument("--out", type=Path, help="default: outputs/eval/e2e_<split>.json")
    args = parser.parse_args()
    ocr_dir = args.ocr_dir or ROOT / "outputs" / "ocr" / args.split
    out = args.out or ROOT / "outputs" / "eval" / f"e2e_{args.split}.json"

    rows, revision = load_gt_rows(args.split, GT_COLUMNS, ocr_dir)
    try:
        result = evaluate(rows, ocr_dir)
    except ValueError as exc:
        sys.exit(str(exc))
    if result["counts"]["evaluated"] == 0:
        sys.exit(f"no OCR JSON for any {args.split} key in --ocr-dir {ocr_dir}; no report written")
    device = resolve_device(ocr_dir, result.pop("ocr_devices"))

    report = {
        "split": args.split,
        "sroie_revision": revision,
        "ocr_dir": str(ocr_dir),
        "ocr_device": device,
        "ocr_schema_version": result.pop("ocr_schema_version"),
        "normalization": NORMALIZATION,
        **result,
    }
    write_json_atomic(out, report)
    m, c = report["micro"], report["counts"]
    print(
        f"evaluated {c['evaluated']}, missing {c['missing']}, "
        f"size mismatch {len(report['size_mismatch_keys'])} | words: "
        f"P {_score_text(m['precision'])} R {_score_text(m['recall'])} "
        f"F1 {_score_text(m['f1'])} "
        f"(TP {m['tp']} FP {m['fp']} FN {m['fn']}) -> {out}"
    )


if __name__ == "__main__":
    main()

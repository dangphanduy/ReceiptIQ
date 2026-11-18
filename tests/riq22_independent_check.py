"""Independent RIQ-22 audit of local SROIE OCR output; run with the CPU environment.

This intentionally does not import ml.ocr_confidence or the confidence evaluator.
"""

import json
import math
import statistics
from pathlib import Path

from datasets import load_from_disk

from ml.ocr_recognition import line_pairs

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT / "outputs/eval/confidence_test.json").read_text(encoding="utf-8"))
ROWS = load_from_disk(str(ROOT / "data/sroie"))["test"].select_columns(
    ["key", "words", "bboxes", "image_size"]
)


def stats(values):
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": values[math.ceil(0.95 * n) - 1],
        "min": values[0],
        "max": values[-1],
    }


def latency(directory):
    values = []
    first = None
    for row in ROWS:
        path = directory / f"{row['key']}.json"
        if path.exists():
            seconds = json.loads(path.read_text(encoding="utf-8"))["engine"]["inference_seconds"]
            if first is None:
                first = seconds
            values.append(seconds)
    return {"stats": stats(values), "first_image_seconds": first,
            "stats_excluding_first": stats(values[1:])}  # fmt: skip


def scores(items, threshold):
    tp = sum(wrong and conf < threshold for conf, wrong in items)
    fp = sum(not wrong and conf < threshold for conf, wrong in items)
    fn = sum(wrong and conf >= threshold for conf, wrong in items)
    flagged = tp + fp
    return {
        "t": threshold,
        "precision": tp / flagged if flagged else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "flagged_rate": flagged / len(items) if items else None,
        "raw": {"tp": tp, "fp": fp, "fn": fn, "flagged": flagged},
    }


def main():
    for label, directory in (
        ("gpu", ROOT / "outputs/ocr/test-gpu"),
        ("cpu", ROOT / "outputs/ocr/test-cpu-latency"),
    ):
        actual = latency(directory)
        reported = REPORT["latency"][label]
        for key in actual:
            assert actual[key] == reported[key], (label, key, actual[key], reported[key])
        print(label, actual)

    all_items, clean_items, examples = [], [], []
    removed = 0
    for row in ROWS:
        path = ROOT / "outputs/ocr/test-gpu" / f"{row['key']}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        lines = doc["lines"]
        predictions = [(line["bbox"], line["text"]) for line in lines]
        truth = list(zip(row["bboxes"], row["words"], strict=True))
        for pair in line_pairs(row["key"], predictions, truth, 0.5):
            conf = lines[pair["pred_idx"]]["confidence"]
            wrong = not pair["exact"]
            item = (conf, wrong)
            all_items.append(item)
            pred = pair["pred_norm"]
            comma_cut = "," in pred and pred.split(",", 1)[0].strip() == pair["gt_norm"]
            if comma_cut:
                removed += 1
            else:
                clean_items.append(item)
            if wrong and conf >= 0.9999:
                examples.append(
                    (
                        conf,
                        row["key"],
                        pair["pred_idx"],
                        pair["gt_idx"],
                        pair["pred_text"],
                        pair["gt_text"],
                        comma_cut,
                    )
                )
    counts = {
        "pairs": len(all_items),
        "correct": sum(not wrong for _, wrong in all_items),
        "wrong": sum(wrong for _, wrong in all_items),
        "comma_cut_removed": removed,
    }
    assert counts == REPORT["counts"], (counts, REPORT["counts"])
    print("counts", counts)
    for label, items in (("all", all_items), ("clean", clean_items)):
        candidate = REPORT["confidence"][label]["candidate"]
        actual = scores(items, candidate["t"])
        for key in candidate:
            assert actual[key] == candidate[key], (label, key, actual[key], candidate[key])
        print(label, actual)
    print("high-confidence wrong examples", sorted(examples, reverse=True)[:20])


if __name__ == "__main__":
    main()

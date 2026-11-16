"""OCR detection metrics: greedy IoU matching of boxes, then precision / recall / F1.

Boxes are [x_min, y_min, x_max, y_max] pixels (OcrLine.bbox and SROIE `bboxes`).
Pure Python on purpose: never import paddle, cv2 or datasets here.
"""

import math
from collections.abc import Iterable, Mapping, Sequence

Box = Sequence[float]


def _area(box: Box) -> float:
    """Area of a valid box; 0.0 for anything malformed, inverted or degenerate."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return 0.0
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0) if x1 > x0 and y1 > y0 else 0.0


def iou(a: Box, b: Box) -> float:
    """Intersection over union; 0.0 if either box is invalid or has zero area."""
    area_a, area_b = _area(a), _area(b)
    if not area_a or not area_b:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = w * h if w > 0 and h > 0 else 0.0
    return inter / (area_a + area_b - inter)


def match_greedy(
    preds: Sequence[Box], gts: Sequence[Box], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """(pred index, GT index, IoU) pairs; each box matched at most once.

    All pairs with IoU >= threshold, taken by IoU descending, ties by (pred, GT) index.
    """
    pairs = [
        (score, p, g)
        for p, pred in enumerate(preds)
        for g, gt in enumerate(gts)
        if (score := iou(pred, gt)) >= threshold and score > 0
    ]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_p, used_g, matches = set(), set(), []
    for score, p, g in pairs:
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            matches.append((p, g, score))
    return matches


def detection_counts(preds: Sequence[Box], gts: Sequence[Box], threshold: float = 0.5) -> dict:
    tp = len(match_greedy(preds, gts, threshold))
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp}


def scores(tp: int, fp: int, fn: int) -> dict:
    """P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); None where the denominator is 0.

    F1 equals the harmonic mean of P and R when both exist, and is 0.0 when P = R = 0.
    """

    def ratio(num: int, den: int) -> float | None:
        return num / den if den else None

    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def micro(per_image: Iterable[Mapping]) -> dict:
    """Scores from TP/FP/FN summed over all images (not a mean of per-image scores)."""
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for counts in per_image:
        for k in totals:
            totals[k] += counts[k]
    return {**totals, **scores(**totals)}

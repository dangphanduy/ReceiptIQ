"""OCR detection metrics: greedy IoU matching of boxes, then precision / recall / F1.

Boxes are [x_min, y_min, x_max, y_max] pixels (OcrLine.bbox and SROIE `bboxes`).
Pure Python on purpose: never import paddle, cv2 or datasets here.
"""

import math
from collections.abc import Iterable, Mapping, Sequence

Box = Sequence[float]


def _valid(box: Box) -> bool:
    """Exactly 4 finite numbers with x1 > x0 and y1 > y0 (comparisons only, no arithmetic)."""
    try:
        return (
            len(box) == 4
            and all(math.isfinite(v) for v in box)
            and box[2] > box[0]
            and box[3] > box[1]
        )
    except (TypeError, ValueError, OverflowError):
        return False


def iou(a: Box, b: Box) -> float:
    """Intersection over union in [0, 1]; 0.0 if either box is invalid (see _valid).

    Both boxes are first scaled by the same power of two so the largest |coordinate| lies in
    [0.5, 1). IoU is scale-invariant and power-of-two scaling is exact in binary floating point,
    so pixel-box results are bit-identical to unscaled arithmetic and nothing can overflow.
    """
    if not (_valid(a) and _valid(b)):
        return 0.0
    e = math.frexp(max(abs(v) for v in (*a, *b)))[1]
    ax0, ay0, ax1, ay1 = (math.ldexp(v, -e) for v in a)
    bx0, by0, bx1, by1 = (math.ldexp(v, -e) for v in b)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    if area_a <= 0 or area_b <= 0:  # a box far smaller than the other can underflow to 0
        return 0.0
    w = min(ax1, bx1) - max(ax0, bx0)
    h = min(ay1, by1) - max(ay0, by0)
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

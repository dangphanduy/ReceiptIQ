"""OCR detection metrics: greedy IoU matching of boxes, then precision / recall / F1.

Boxes are [x_min, y_min, x_max, y_max] pixels (OcrLine.bbox and SROIE `bboxes`).
Pure Python on purpose: never import paddle, cv2 or datasets here.
"""

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction

Box = Sequence[float]


def _finite_real(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    return isinstance(v, numbers.Integral) or math.isfinite(v)  # isfinite(10**400) overflows


def _valid(box: Box) -> bool:
    """Exactly 4 finite real numbers (bool rejected) with x1 > x0 and y1 > y0.

    Comparisons only, on the original values, so the check itself is exact.
    """
    try:
        return len(box) == 4 and all(map(_finite_real, box)) and box[2] > box[0] and box[3] > box[1]
    except TypeError:  # not a sequence
        return False


def _iou_exact(a: Box, b: Box) -> Fraction:
    """Exact IoU as a Fraction in [0, 1]; 0 if either box is invalid or they do not overlap.

    Floats convert to Fraction exactly, so no width, area or union can overflow or underflow.
    """
    if not (_valid(a) and _valid(b)):
        return Fraction(0)
    # Fast reject with exact comparisons only: most pred x GT pairs do not touch.
    if min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1]):
        return Fraction(0)
    ax0, ay0, ax1, ay1 = map(Fraction, a)
    bx0, by0, bx1, by1 = map(Fraction, b)
    inter = (min(ax1, bx1) - max(ax0, bx0)) * (min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union


def iou(a: Box, b: Box) -> float:
    """Intersection over union in [0, 1], correctly rounded from the exact value (_iou_exact).

    0.0 if either box is invalid: not exactly 4 finite real numbers, or x1 <= x0, or y1 <= y0.
    """
    return float(_iou_exact(a, b))


def match_greedy(
    preds: Sequence[Box], gts: Sequence[Box], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """(pred index, GT index, IoU as float) pairs; each box matched at most once.

    All pairs with exact IoU >= threshold, taken by exact IoU descending, ties by (pred, GT)
    index. Comparing Fractions means a true IoU just below the threshold never rounds into a
    match (Fraction vs float comparison is exact too).
    """
    pairs = [
        (score, p, g)
        for p, pred in enumerate(preds)
        for g, gt in enumerate(gts)
        if (score := _iou_exact(pred, gt)) >= threshold and score > 0
    ]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_p, used_g, matches = set(), set(), []
    for score, p, g in pairs:
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            matches.append((p, g, float(score)))
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

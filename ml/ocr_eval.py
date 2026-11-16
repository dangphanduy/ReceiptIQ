"""OCR detection metrics: greedy IoU matching of boxes, then precision / recall / F1.

Boxes are [x_min, y_min, x_max, y_max] pixels (OcrLine.bbox and SROIE `bboxes`).
Pure Python on purpose: never import paddle, cv2 or datasets here.

IoU input contract (closed on purpose): boxes come from JSON (OCR `bbox`) and SROIE `bboxes`,
so a coordinate is a Python int (any size, not bool) or a finite float (float subclasses such as
numpy.float64 included). Any other type (bool, Fraction, Decimal, complex, str, None, numpy
int / float32 scalars), NaN, +-inf, a wrong length, a non-sequence, or x1 <= x0 / y1 <= y0
makes the box INVALID, and its IoU is 0.0. iou() never raises.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction

Box = Sequence[float]


def _valid_coord(v: object) -> bool:
    """int (any size) but not bool, or a finite float; see the module contract."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True  # never via math.isfinite: float(10**400) overflows
    return isinstance(v, float) and math.isfinite(v)


def _coords(box: object) -> tuple | None:
    """The 4 coordinates of a VALID box, read once; None for anything else, never raises.

    Comparisons between ints and floats are exact, so validation needs no arithmetic.
    """
    try:
        if len(box) != 4:
            return None
        c = (box[0], box[1], box[2], box[3])
        if all(map(_valid_coord, c)) and c[2] > c[0] and c[3] > c[1]:
            return c
    except Exception:  # an odd object is INVALID, not an error
        pass
    return None


def _iou_exact(a: Box, b: Box) -> Fraction:
    """Exact IoU as a Fraction in [0, 1]; 0 if either box is INVALID or they do not overlap.

    Fraction(int) and Fraction(float) are exact, so nothing can overflow or underflow.
    """
    ca, cb = _coords(a), _coords(b)
    if ca is None or cb is None:
        return Fraction(0)
    ax0, ay0, ax1, ay1 = ca
    bx0, by0, bx1, by1 = cb
    # Fast reject with exact comparisons only: most pred x GT pairs do not touch.
    if min(ax1, bx1) <= max(ax0, bx0) or min(ay1, by1) <= max(ay0, by0):
        return Fraction(0)
    ax0, ay0, ax1, ay1 = map(Fraction, ca)
    bx0, by0, bx1, by1 = map(Fraction, cb)
    inter = (min(ax1, bx1) - max(ax0, bx0)) * (min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union


def iou(a: Box, b: Box) -> float:
    """IoU in [0, 1], correctly rounded from the exact value; never raises.

    0.0 if either box is INVALID (see the module contract) or the boxes do not overlap.
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

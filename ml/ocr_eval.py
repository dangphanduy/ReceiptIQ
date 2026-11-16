"""OCR detection metrics: greedy IoU matching of boxes, then precision / recall / F1.

Boxes are [x_min, y_min, x_max, y_max] pixels (OcrLine.bbox and SROIE `bboxes`).
Pure Python on purpose: never import paddle, cv2 or datasets here.

IoU input contract (closed on purpose): real boxes are exact built-ins (json.loads gives
list/int/float for OCR `bbox`, SROIE via Arrow gives lists of int), so only those are accepted,
checked by exact type before any method of the object is used:
- box: type(box) is list or tuple, with exactly 4 coordinates (subclasses are INVALID);
- coordinate: type(v) is int (any size; bool is not int by type) or type(v) is float and finite
  (int/float subclasses such as numpy.float64, and every other type, are INVALID);
- and x1 > x0, y1 > y0.
Anything else is INVALID and has IoU 0.0; iou() never raises for any Python object.
Callers: convert numpy/other numbers with int()/float() before calling.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction

Box = Sequence[float]
ExactBox = tuple[Fraction, Fraction, Fraction, Fraction]


def _to_exact_box(box: object) -> ExactBox | None:
    """The box as 4 exact Fractions if VALID (see the module contract), else None.

    Exact type checks come first, so no foreign method (len, getitem, comparison, as_integer_ratio)
    ever runs. Fraction() of an exact int or float is exact, never rounds and never raises.
    """
    if type(box) not in (list, tuple) or len(box) != 4:
        return None
    for v in box:
        if not (type(v) is int or (type(v) is float and math.isfinite(v))):
            return None  # math.isfinite is only reached for exact floats, never for big ints
    x0, y0, x1, y1 = (Fraction(v) for v in box)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _iou_of_exact(a: ExactBox, b: ExactBox) -> Fraction:
    """Exact IoU of two VALID boxes, in Fractions only."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    w = min(ax1, bx1) - max(ax0, bx0)
    h = min(ay1, by1) - max(ay0, by0)
    if w <= 0 or h <= 0:  # most pred x GT pairs do not touch
        return Fraction(0)
    inter = w * h
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def _iou_exact(a: object, b: object) -> Fraction:
    """Exact IoU as a Fraction in [0, 1]; 0 if either box is INVALID or they do not overlap."""
    ea, eb = _to_exact_box(a), _to_exact_box(b)
    return Fraction(0) if ea is None or eb is None else _iou_of_exact(ea, eb)


def iou(a: Box, b: Box) -> float:
    """IoU in [0, 1], correctly rounded from the exact value; never raises.

    0.0 if either box is INVALID (see the module contract) or the boxes do not overlap.
    Convert numpy/other numbers with int()/float() before calling.
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
    exact_preds = [_to_exact_box(b) for b in preds]  # convert each box once, not per pair
    exact_gts = [_to_exact_box(b) for b in gts]
    pairs = [
        (score, p, g)
        for p, pred in enumerate(exact_preds)
        if pred is not None
        for g, gt in enumerate(exact_gts)
        if gt is not None and (score := _iou_of_exact(pred, gt)) >= threshold and score > 0
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

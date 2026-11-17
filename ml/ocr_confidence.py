"""Latency stats and OCR-confidence analysis for flagging lines for review (RIQ-22).

A "line" is a ml.ocr_recognition.line_pairs record; it is wrong when pair["exact"] is False.
Flag rule: confidence < t. Pure Python: never import paddle, cv2, datasets or PIL here.
"""

import bisect
import statistics
from collections.abc import Mapping, Sequence

from ml.ocr_e2e import prf1

# 0.50..0.99 step 0.01, then 0.995 and 0.999; rounded so 0.57 is 0.57, not 0.5700000000000001
THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + i / 100, 2) for i in range(50)) + (0.995, 0.999)


def percentile(values: Sequence[float], p: int) -> float | None:
    """Nearest-rank percentile: the ceil(p/100 * n)-th smallest value (1-based), no
    interpolation. p is an integer percent in [1, 100]; None for no values.

    Integer arithmetic, so 0.95 * 20 cannot round up to rank 20.
    """
    if not values:
        return None
    rank = max(1, -(-p * len(values) // 100))  # ceil without float error
    return sorted(values)[rank - 1]


def latency_stats(seconds: Sequence[float]) -> dict:
    """{n, mean, median, p95, min, max}; p95 is nearest-rank (see percentile).

    Every stat but n is None for no values. Median of an even n is the mean of the middle two.
    """
    if not seconds:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(seconds),
        "mean": statistics.fmean(seconds),
        "median": statistics.median(seconds),
        "p95": percentile(seconds, 95),
        "min": min(seconds),
        "max": max(seconds),
    }


def is_comma_cut(pair: Mapping) -> bool:
    """Suspected GT comma-cut (docs/baseline.md): the normalized OCR text has a comma and its
    part before the first comma, stripped, equals the normalized GT text.

    A heuristic: it also catches a real OCR error that happens to look like this.
    """
    pred = pair["pred_norm"]
    return "," in pred and pred.split(",", 1)[0].strip() == pair["gt_norm"]


def histogram(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    """Counts per bin [edges[i], edges[i+1]); the last bin also includes edges[-1].

    Raises ValueError for edges that are not strictly increasing (at least two) and for a
    value outside [edges[0], edges[-1]].
    """
    if len(edges) < 2 or any(a >= b for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"edges must be at least two strictly increasing numbers: {edges}")
    counts = [0] * (len(edges) - 1)
    for v in values:
        if not edges[0] <= v <= edges[-1]:
            raise ValueError(f"value {v} outside [{edges[0]}, {edges[-1]}]")
        counts[min(bisect.bisect_right(edges, v) - 1, len(counts) - 1)] += 1
    return counts


def threshold_sweep(
    confidences: Sequence[float], wrong_flags: Sequence[bool], thresholds: Sequence[float]
) -> list[dict]:
    """One row per t: {t, flagged, tp, fp, fn, precision, recall, f1}.

    Flagged = confidence < t; positive = wrong line. tp = flagged wrong, fp = flagged correct,
    fn = unflagged wrong. Ratios with a zero denominator are None (ml.ocr_e2e.prf1).
    Raises ValueError if confidences and wrong_flags differ in length.
    """
    pairs = list(zip(confidences, wrong_flags, strict=True))
    n_wrong = sum(w for _, w in pairs)
    rows = []
    for t in thresholds:
        flagged = [w for c, w in pairs if c < t]
        tp = sum(flagged)
        fp, fn = len(flagged) - tp, n_wrong - tp
        rows.append({"t": t, "flagged": len(flagged), "tp": tp, "fp": fp, "fn": fn,
                     **prf1(tp, fp, fn)})  # fmt: skip
    return rows


def pick_threshold(rows: Sequence[Mapping]) -> Mapping | None:
    """The row with the highest F1, ties to the lower t; rows with F1 None are skipped.

    None if no row has an F1.
    """
    scored = [r for r in rows if r["f1"] is not None]
    return max(scored, key=lambda r: (r["f1"], -r["t"])) if scored else None

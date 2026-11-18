"""OCR recognition metrics on detection-matched line pairs: CER and line exact match.

Pairs come from ml.ocr_eval.match_greedy (same IoU rule as detection). Text is compared after
ml.text_norm.normalize_text. Pure Python: never import paddle, cv2 or datasets here.
"""

import statistics
from collections.abc import Sequence

from ml.ocr_eval import Box, match_greedy
from ml.text_norm import normalize_text

Line = tuple[Box, str]  # (bbox [x0, y0, x1, y1], text)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over Unicode code points; insert, delete, substitute cost 1 each.

    O(len(a) * len(b)) time, O(min(len(a), len(b))) memory.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def line_cer(pred: str, gt: str) -> float | None:
    """edit_distance / len of the normalized GT; None if the normalized GT is empty. Can be > 1."""
    p, g = normalize_text(pred), normalize_text(gt)
    return edit_distance(p, g) / len(g) if g else None


def line_pairs(
    key: str, preds: Sequence[Line], gts: Sequence[Line], threshold: float = 0.5
) -> list[dict]:
    """One record per match_greedy pair (IoU >= threshold), in match order."""
    pairs = []
    for p, g, score in match_greedy([b for b, _ in preds], [b for b, _ in gts], threshold):
        pred_text, gt_text = preds[p][1], gts[g][1]
        pn, gn = normalize_text(pred_text), normalize_text(gt_text)
        pairs.append(
            {
                "key": key,
                "pred_idx": p,
                "gt_idx": g,
                "iou": score,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "gt_norm": gn,
                "pred_norm": pn,
                "edit_distance": edit_distance(pn, gn),
                "cer": line_cer(pred_text, gt_text),  # the one CER formula; None for empty GT
                "exact": pn == gn,
            }
        )
    return pairs


def summarize(pairs: Sequence[dict], worst_n: int = 20) -> dict:
    """Counts, mean/median CER (empty-GT pairs excluded), exact-match rate, worst CER pairs.

    Worst pairs: CER descending, ties by (key, gt_idx). Any stat with no data is None.
    """
    cer_pairs = [p for p in pairs if p["cer"] is not None]
    cers = [p["cer"] for p in cer_pairs]
    worst = sorted(cer_pairs, key=lambda p: (-p["cer"], p["key"], p["gt_idx"]))[:worst_n]
    return {
        "counts": {
            "pairs": len(pairs),
            "cer_pairs": len(cer_pairs),
            "skipped_empty_gt": len(pairs) - len(cer_pairs),
        },
        "cer": {
            "mean": statistics.fmean(cers) if cers else None,
            "median": statistics.median(cers) if cers else None,
        },
        "exact_match_rate": sum(p["exact"] for p in pairs) / len(pairs) if pairs else None,
        "worst_lines": worst,
    }

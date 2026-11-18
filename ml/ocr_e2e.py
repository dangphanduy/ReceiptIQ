"""End-to-end OCR metric: word-level precision / recall / F1, SROIE Task 2 style.

Per image, every OCR line text and every GT `words` string goes through
ml.text_norm.normalize_text and is split on whitespace; the two token multisets are matched
regardless of position or line. Pure Python: never import paddle, cv2 or datasets here.
"""

from collections import Counter
from collections.abc import Iterable, Mapping

from ml.text_norm import normalize_text


def word_tokens(texts: Iterable[str]) -> Counter[str]:
    """Multiset of the whitespace tokens of every text after normalize_text.

    Punctuation stays attached ("Date: 12/3/56" -> "DATE:", "12/3/56").
    """
    return Counter(tok for text in texts for tok in normalize_text(text).split())


def word_counts(pred_texts: Iterable[str], gt_texts: Iterable[str]) -> dict:
    """{"tp", "fp", "fn"} for one image, from a multiset (Counter) intersection.

    tp = sum over tokens of min(pred count, GT count); fp = pred tokens - tp;
    fn = GT tokens - tp. Pred "A A B" vs GT "A B B" gives tp 2, fp 1, fn 1.
    """
    pred, gt = word_tokens(pred_texts), word_tokens(gt_texts)
    tp = (pred & gt).total()
    return {"tp": tp, "fp": pred.total() - tp, "fn": gt.total() - tp}


def prf1(tp: int, fp: int, fn: int) -> dict:
    """P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).

    Return None for each ratio with a zero denominator, as in ml.ocr_eval.scores.
    No predicted tokens gives P = None; no GT tokens gives R = None; both empty gives
    P = R = F1 = None. Counts stay unchanged.
    """

    def ratio(num: int, den: int) -> float | None:
        return num / den if den else None

    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def micro_prf1(per_image: Iterable[Mapping]) -> dict:
    """tp/fp/fn summed over all images first, then prf1 (not a mean of per-image F1).

    No images gives all counts 0 and P = R = F1 = None (see prf1).
    """
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for counts in per_image:
        for k in totals:
            totals[k] += counts[k]
    return {**totals, **prf1(**totals)}

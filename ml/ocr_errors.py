"""Heuristic OCR error categories for wrong line pairs (RIQ-23).

A pair is a ml.ocr_recognition.line_pairs record; it is wrong when pair["exact"] is False.
GT artefacts are excluded first (comma cuts, known GT typos), then each remaining pair gets one
category from a character diff of the normalized texts. Pure Python: never import paddle, cv2,
datasets or PIL here.
"""

import difflib
from collections import Counter
from collections.abc import Mapping, Sequence

from ml.ocr_confidence import is_comma_cut

# GT tokens checked against the receipt image: the image shows what OCR read, GT is wrong.
# Keyed by SROIE key so a token is only excluded on the receipt where it was checked.
KNOWN_GT_TYPOS: dict[str, frozenset[str]] = {
    "X00016469670": frozenset({"PERSON"}),  # receipt prints "Sales Persor"
    "X00016469671": frozenset({"PERSON"}),
    "X51005200931": frozenset({"JM0326955-V"}),  # prints JM0325955-V
    "X51005230621": frozenset({"INVOLCE", "INCLUSLVE", "AMONUT"}),
    "X51005230648": frozenset({"INVOLCE"}),
    "X51005442322": frozenset({"ROASTEO", "RECRUITEMNT@TONYROMAS.COM.MY"}),
    "X51005442366": frozenset({"FUJ1500ML"}),
    "X51005442375": frozenset({"FRLED"}),
    "X51005715007": frozenset({"CAHS"}),
}


def _alnum(s: str) -> str:
    return "".join(c for c in s if c.isalnum())


def is_comma_cut_variant(pair: Mapping) -> bool:
    """GT cut at the first comma, but with a spacing or punctuation difference before the comma
    that ml.ocr_confidence.is_comma_cut does not allow (GT `KG BATU 30`, OCR `KG BATU30,ULU`).

    Compares letters and digits only, so it is looser than is_comma_cut. Something must have
    been cut: the OCR text after the first comma needs at least one letter or digit, so GT
    `SETAPAK.` vs OCR `SETAPAK,` is not a variant.
    """
    pred = pair["pred_norm"]
    if "," not in pred:
        return False
    before, after = pred.split(",", 1)
    return bool(_alnum(after)) and _alnum(before) == _alnum(pair["gt_norm"])


def is_known_gt_typo(pair: Mapping) -> bool:
    """The normalized GT line contains a token listed in KNOWN_GT_TYPOS for this key."""
    typos = KNOWN_GT_TYPOS.get(pair["key"], frozenset())
    return any(t in typos for t in pair["gt_norm"].split())


def _op_category(tag: str, gt: str, pred: str, at_edge: bool) -> str:
    chars = gt + pred
    if not any(c.isalnum() for c in chars):
        return "punctuation"
    if tag == "replace":
        if not all(c.isalnum() for c in chars):
            return "symbol"  # a symbol read as a letter/digit or the other way ($ -> S, @ -> 0)
        if (gt.isdigit() and pred.isalpha()) or (gt.isalpha() and pred.isdigit()):
            return "digit_letter"
        return "char_substitution"
    if at_edge:
        return "extra_text" if tag == "insert" else "missing_text"
    return "inner_insert_delete"


def classify(gt_norm: str, pred_norm: str) -> str:
    """One category for a wrong pair of normalized texts.

    `spacing` if the texts only differ in spaces. Otherwise spaces are removed, the texts are
    diffed (difflib, autojunk off) and every non-equal op gets a category; a pair whose ops
    disagree is `mixed`. Raises ValueError for equal texts.
    """
    if gt_norm == pred_norm:
        raise ValueError("pair is not wrong: texts are equal")
    gt, pred = gt_norm.replace(" ", ""), pred_norm.replace(" ", "")
    if gt == pred:
        return "spacing"
    matcher = difflib.SequenceMatcher(None, gt, pred, autojunk=False)
    cats = {
        _op_category(tag, gt[i1:i2], pred[j1:j2], i1 == 0 or i2 == len(gt))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    }
    return cats.pop() if len(cats) == 1 else "mixed"


def error_report(pairs: Sequence[Mapping], n_examples: int = 3) -> dict:
    """Exclusion counts and categories of the wrong pairs.

    Exclusions, first match wins: is_comma_cut, is_comma_cut_variant, is_known_gt_typo.
    Categories sort by count desc, then name; `share` is count / classified. Examples are the
    first pairs in input order with distinct keys, at most n_examples per category.
    Raises ValueError for n_examples < 0.
    """
    if n_examples < 0:
        raise ValueError(f"n_examples must be >= 0, got {n_examples}")
    wrong = [p for p in pairs if not p["exact"]]
    excluded = Counter()
    by_cat: dict[str, list[Mapping]] = {}
    for p in wrong:
        if is_comma_cut(p):
            excluded["comma_cut"] += 1
        elif is_comma_cut_variant(p):
            excluded["comma_cut_variant"] += 1
        elif is_known_gt_typo(p):
            excluded["known_gt_typo"] += 1
        else:
            by_cat.setdefault(classify(p["gt_norm"], p["pred_norm"]), []).append(p)
    classified = sum(len(v) for v in by_cat.values())

    categories = []
    for cat, items in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        examples, seen = [], set()
        for p in items:
            if len(examples) == n_examples:
                break
            if p["key"] not in seen:
                seen.add(p["key"])
                examples.append({k: p[k] for k in ("key", "gt_idx", "gt_text", "pred_text")})
        categories.append(
            {
                "category": cat,
                "count": len(items),
                "share": len(items) / classified,
                "examples": examples,
            }
        )
    return {
        "counts": {
            "pairs": len(pairs),
            "wrong": len(wrong),
            "excluded_comma_cut": excluded["comma_cut"],
            "excluded_comma_cut_variant": excluded["comma_cut_variant"],
            "excluded_known_gt_typo": excluded["known_gt_typo"],
            "classified": classified,
        },
        "categories": categories,
    }

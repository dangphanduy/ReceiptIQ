"""OCR error categories for wrong line pairs, on hand-made pairs (no model, no dataset)."""

import pytest

from ml.ocr_errors import (
    KNOWN_GT_TYPOS,
    classify,
    error_report,
    is_comma_cut_variant,
    is_known_gt_typo,
)
from ml.text_norm import normalize_text


@pytest.mark.parametrize(
    ("gt", "pred", "category"),
    [
        ("GST NO : 000800689824", "GST NO: 000800689824", "spacing"),
        ("SANYU STATIONERY SHOP", "SANYUSTATIONERYSHOP", "spacing"),
        ("CHANGE :", "CHANGE", "punctuation"),
        ("0.41", "041", "punctuation"),  # dropped decimal point
        ("--GST SUMMARY--", "GST SUMMARY", "punctuation"),
        ("$7.10", "S7.10", "symbol"),
        ("SR @6%", "SR 06%", "symbol"),
        ("TEO HENG & BOOKS", "TEO HENG S BOOKS", "symbol"),
        ("SUB-TOTAL", "8UB-TOTAL", "digit_letter"),
        ("INVOICE NO", "INV0ICE NO", "digit_letter"),
        ("SALES PERSON", "SALES PERSOR", "char_substitution"),
        ("67.92", "67.82", "char_substitution"),  # digit read as another digit
        ("TOTAL ITEM(S)", "TOTAL ITERN(S)", "char_substitution"),  # M -> RN
        ("193.00", "1 193.00", "extra_text"),
        ("6.36", "6.36 SR", "extra_text"),
        ("DATE", "ATE", "missing_text"),
        ("SERVICE CHARG", "SERVICE CHAR", "missing_text"),
        ("QTY", "QY", "inner_insert_delete"),
        ("RM4.10", "RN4,10", "mixed"),
        ("GST NO : 1", "GST NO: 7", "char_substitution"),  # spaces ignored next to a real error
    ],
)
def test_classify(gt, pred, category):
    assert classify(gt, pred) == category


def test_classify_refuses_a_correct_pair():
    with pytest.raises(ValueError, match="not wrong"):
        classify("TOTAL", "TOTAL")


@pytest.mark.parametrize(
    ("gt", "pred", "expected"),
    [
        ("KG BATU 30", "KG BATU30,ULUYAM LAMA", True),  # space differs before the cut comma
        ("SETAPAK.", "SETAPAK,", True),
        ("9", "9, JALAN SUBANG", True),
        ("KG BATU 30", "KG BATU 31,ULUYAM", False),
        ("TOTAL", "TOTAL.", False),  # no comma
    ],
)
def test_is_comma_cut_variant(gt, pred, expected):
    assert is_comma_cut_variant({"gt_norm": gt, "pred_norm": pred}) is expected


def test_known_gt_typo_is_per_key_and_per_token():
    key, typos = next(iter(KNOWN_GT_TYPOS.items()))
    token = next(iter(typos))
    other_key = "NOT-" + key
    assert is_known_gt_typo({"key": key, "gt_norm": f"TAX {token} NO"})
    assert not is_known_gt_typo({"key": other_key, "gt_norm": f"TAX {token} NO"})
    assert not is_known_gt_typo({"key": key, "gt_norm": f"TAX {token}X NO"})


def test_known_gt_typos_are_normalized_tokens():
    for key, typos in KNOWN_GT_TYPOS.items():
        assert key.startswith("X") and typos
        for t in typos:
            assert t == normalize_text(t) and " " not in t


def pair(key, gt, pred, gt_idx=0):
    gn, pn = normalize_text(gt), normalize_text(pred)
    return {
        "key": key, "gt_idx": gt_idx, "gt_text": gt, "pred_text": pred,
        "gt_norm": gn, "pred_norm": pn, "exact": gn == pn,
    }  # fmt: skip


def test_error_report_counts_exclusions_in_order_and_sorts_categories():
    typo_key, typos = next(iter(KNOWN_GT_TYPOS.items()))
    typo = next(iter(typos))
    pairs = [
        pair("K1", "TOTAL", "total"),  # correct: ignored
        pair("K1", "9", "9, JALAN", 1),  # comma cut (is_comma_cut)
        pair("K1", "KG BATU 30", "KG BATU30,ULUYAM", 2),  # comma-cut variant
        pair(typo_key, typo, typo + "Z"),  # known GT typo
        pair("K1", "TEL : 1", "TEL:1", 3),
        pair("K2", "TEL : 2", "TEL:2"),
        pair("K2", "CHANGE :", "CHANGE", 1),
        pair("K3", "QTY", "QY"),
        pair("K3", "TEL : 3", "TEL:3", 1),
    ]
    report = error_report(pairs, n_examples=2)
    assert report["counts"] == {
        "pairs": 9, "wrong": 8, "excluded_comma_cut": 1, "excluded_comma_cut_variant": 1,
        "excluded_known_gt_typo": 1, "classified": 5,
    }  # fmt: skip
    cats = report["categories"]
    # count desc, then name: inner_insert_delete before punctuation at count 1
    assert [(c["category"], c["count"]) for c in cats] == [
        ("spacing", 3), ("inner_insert_delete", 1), ("punctuation", 1)
    ]  # fmt: skip
    assert cats[0]["share"] == pytest.approx(0.6)
    # examples: first pairs in input order with distinct keys, at most n_examples
    assert cats[0]["examples"] == [
        {"key": "K1", "gt_idx": 3, "gt_text": "TEL : 1", "pred_text": "TEL:1"},
        {"key": "K2", "gt_idx": 0, "gt_text": "TEL : 2", "pred_text": "TEL:2"},
    ]


def test_error_report_examples_skip_repeated_keys():
    pairs = [pair("K1", "TEL : 1", "TEL:1"), pair("K1", "TEL : 2", "TEL:2", 1)]
    pairs.append(pair("K2", "TEL : 3", "TEL:3"))
    examples = error_report(pairs, n_examples=3)["categories"][0]["examples"]
    assert [e["key"] for e in examples] == ["K1", "K2"]


def test_error_report_with_no_wrong_pairs():
    report = error_report([pair("K1", "TOTAL", "TOTAL")])
    assert report["counts"]["classified"] == 0
    assert report["categories"] == []

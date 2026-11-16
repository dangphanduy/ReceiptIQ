import random

import pytest

from ml.ocr_recognition import edit_distance, line_cer, line_pairs, summarize

A, B, C = [0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("café", "cafe", 1),  # é vs e: one substitution
        ("é", "é", 2),  # decomposed form is two code points; no NFC on purpose
        ("a😀b", "ab", 1),  # non-BMP char is one code point
        ("😀", "😃", 1),
    ],
)
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b) == expected


def test_edit_distance_is_symmetric():
    rng = random.Random(0)
    for _ in range(200):
        a = "".join(rng.choice("ab c😀") for _ in range(rng.randrange(8)))
        b = "".join(rng.choice("ab c😀") for _ in range(rng.randrange(8)))
        assert edit_distance(a, b) == edit_distance(b, a)
        assert edit_distance(a, b) <= max(len(a), len(b))


def test_line_cer_normalizes_both_sides():
    assert line_cer(" total\tRM ", "TOTAL RM") == 0.0
    assert line_cer("TOTAL RN", "total rm") == pytest.approx(1 / 8)


def test_line_cer_is_none_for_empty_gt():
    assert line_cer("abc", "") is None
    assert line_cer("", " \t ") is None


def test_line_cer_can_exceed_one():
    assert line_cer("ABCDEFGH", "AB") == 3.0


def test_line_cer_sharp_s_counts_after_uppercase():
    # normalize_text("ß") == "SS", so "STRASSE" vs "straße" is an exact match
    assert line_cer("STRASSE", "straße") == 0.0


def test_line_pairs_uses_detection_matches():
    preds = [(B, "world"), (A, "HELL0"), ([90, 90, 99, 99], "noise")]
    gts = [(A, "hello"), (B, "World"), (C, "missed")]

    pairs = line_pairs("K1", preds, gts, 0.5)

    assert [(p["pred_idx"], p["gt_idx"]) for p in pairs] == [(0, 1), (1, 0)]
    first = pairs[0]
    assert first == {
        "key": "K1", "pred_idx": 0, "gt_idx": 1, "iou": 1.0,
        "gt_text": "World", "pred_text": "world", "gt_norm": "WORLD", "pred_norm": "WORLD",
        "edit_distance": 0, "cer": 0.0, "exact": True,
    }  # fmt: skip
    assert pairs[1]["edit_distance"] == 1
    assert pairs[1]["cer"] == pytest.approx(0.2)
    assert pairs[1]["exact"] is False


def test_line_pairs_empty_gt_has_no_cer_but_scores_exact():
    pairs = line_pairs("K1", [(A, " "), (B, "x")], [(A, ""), (B, "  ")], 0.5)
    assert [p["cer"] for p in pairs] == [None, None]
    assert [p["exact"] for p in pairs] == [True, False]


def pair(key, gt_idx, cer, exact=False):
    return {"key": key, "gt_idx": gt_idx, "cer": cer, "exact": exact}


def test_summarize_mean_vs_median_on_uneven_list():
    pairs = [pair("K", i, c) for i, c in enumerate([0.0, 0.0, 0.0, 0.1, 5.0])]
    s = summarize(pairs)
    assert s["cer"]["mean"] == pytest.approx(1.02)
    assert s["cer"]["median"] == 0.0


def test_summarize_skips_empty_gt_for_cer_but_counts_exact():
    pairs = [pair("K", 0, 0.0, True), pair("K", 1, None, True), pair("K", 2, 1.0)]
    s = summarize(pairs)
    assert s["counts"] == {"pairs": 3, "cer_pairs": 2, "skipped_empty_gt": 1}
    assert s["cer"] == {"mean": 0.5, "median": 0.5}
    assert s["exact_match_rate"] == pytest.approx(2 / 3)
    assert [p["gt_idx"] for p in s["worst_lines"]] == [2, 0]


def test_summarize_nothing_is_null():
    assert summarize([]) == {
        "counts": {"pairs": 0, "cer_pairs": 0, "skipped_empty_gt": 0},
        "cer": {"mean": None, "median": None},
        "exact_match_rate": None,
        "worst_lines": [],
    }


def test_summarize_only_empty_gt_has_null_cer_but_exact_rate():
    s = summarize([pair("K", 0, None, True)])
    assert s["cer"] == {"mean": None, "median": None}
    assert s["exact_match_rate"] == 1.0
    assert s["worst_lines"] == []


def test_worst_lines_order_and_truncation():
    pairs = [pair(f"K{i % 3}", i, (i % 5) / 4) for i in range(30)]
    pairs.append(pair("K0", 99, None))
    rng = random.Random(1)
    rng.shuffle(pairs)

    worst = summarize(pairs, worst_n=20)["worst_lines"]

    assert len(worst) == 20
    keys = [(-p["cer"], p["key"], p["gt_idx"]) for p in worst]
    assert keys == sorted(keys)
    assert worst[0]["cer"] == 1.0
    assert all(p["cer"] is not None for p in worst)
    # ties on CER broken by (key, gt_idx), deterministic regardless of input order
    assert [(p["key"], p["gt_idx"]) for p in worst[:6]] == [
        ("K0", 9), ("K0", 24), ("K1", 4), ("K1", 19), ("K2", 14), ("K2", 29)
    ]  # fmt: skip

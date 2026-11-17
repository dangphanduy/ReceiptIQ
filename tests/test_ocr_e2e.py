"""End-to-end word metrics on hand-made texts; no model, no dataset."""

import statistics

import pytest

from ml.ocr_e2e import micro_prf1, prf1, word_counts, word_tokens


def test_tokens_are_normalized_whitespace_split():
    assert word_tokens([" total\tRM ", "rm 1.00\n"]) == {"TOTAL": 1, "RM": 2, "1.00": 1}
    assert word_tokens(["", "  \t "]) == {}


def test_punctuation_stays_attached_to_tokens():
    # SROIE Task 2 example: "Date: 12/3/56" -> "Date:", "12/3/56"
    assert word_tokens(["Date: 12/3/56"]) == {"DATE:": 1, "12/3/56": 1}


def test_duplicate_tokens_match_as_multiset():
    assert word_counts(["A A B"], ["A B B"]) == {"tp": 2, "fp": 1, "fn": 1}


def test_token_order_and_line_split_do_not_matter():
    assert word_counts(["B", "A C"], ["C B A"]) == {"tp": 3, "fp": 0, "fn": 0}


def test_case_and_whitespace_are_normalized():
    assert word_counts(["total  rm"], ["TOTAL RM"]) == {"tp": 2, "fp": 0, "fn": 0}


def test_exact_match():
    counts = word_counts(["TOTAL 9.00"], ["TOTAL 9.00"])
    assert counts == {"tp": 2, "fp": 0, "fn": 0}
    assert prf1(**counts) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_empty_prediction_scores_zero():
    counts = word_counts([], ["TOTAL 9.00"])
    assert counts == {"tp": 0, "fp": 0, "fn": 2}
    assert prf1(**counts) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_empty_gt_scores_zero():
    counts = word_counts(["TOTAL 9.00"], [" "])
    assert counts == {"tp": 0, "fp": 2, "fn": 0}
    assert prf1(**counts) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_both_empty_scores_zero():
    assert prf1(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_prf1_values():
    assert prf1(2, 1, 1) == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3})
    assert prf1(1, 3, 0) == {"precision": 0.25, "recall": 1.0, "f1": 0.4}


def test_micro_sums_counts_before_ratios():
    per_image = [
        {"key": "K1", **word_counts(["A"], ["A"])},  # F1 1.0
        {"key": "K2", **word_counts(["X Y Z"], ["B"])},  # F1 0.0
    ]

    result = micro_prf1(per_image)

    assert result == {"tp": 1, "fp": 3, "fn": 1, "precision": 0.25, "recall": 0.5, "f1": 1 / 3}
    mean_f1 = statistics.fmean(prf1(r["tp"], r["fp"], r["fn"])["f1"] for r in per_image)
    assert mean_f1 == 0.5
    assert result["f1"] != pytest.approx(mean_f1)


def test_micro_of_no_images_scores_zero():
    assert micro_prf1([]) == {
        "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }  # fmt: skip

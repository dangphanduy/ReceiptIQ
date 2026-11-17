"""Latency stats, comma-cut filter, histogram and threshold sweep on hand-made data."""

import subprocess
import sys

import pytest

from ml.ocr_confidence import (
    THRESHOLDS,
    histogram,
    is_comma_cut,
    latency_stats,
    percentile,
    pick_budget,
    pick_threshold,
    threshold_sweep,
)


def test_latency_stats_empty_is_all_none():
    assert latency_stats([]) == {
        "n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None
    }  # fmt: skip


def test_latency_stats_one_item():
    assert latency_stats([2.5]) == {
        "n": 1, "mean": 2.5, "median": 2.5, "p95": 2.5, "min": 2.5, "max": 2.5
    }  # fmt: skip


def test_latency_stats_odd_n():
    assert latency_stats([3.0, 1.0, 2.0]) == {
        "n": 3, "mean": 2.0, "median": 2.0, "p95": 3.0, "min": 1.0, "max": 3.0
    }  # fmt: skip


def test_latency_stats_even_n_median_is_midpoint():
    stats = latency_stats([4.0, 1.0, 3.0, 2.0])
    assert stats["median"] == 2.5
    assert stats["p95"] == 4.0


@pytest.mark.parametrize(("n", "expected"), [(20, 19), (21, 20), (100, 95), (101, 96)])
def test_p95_is_nearest_rank_not_interpolated(n, expected):
    # nearest rank: the ceil(0.95 * n)-th smallest value; interpolation would give 19.05 for n=20
    assert latency_stats(list(range(1, n + 1)))["p95"] == expected


def test_percentile_p05_and_empty():
    assert percentile(list(range(1, 21)), 5) == 1
    assert percentile(list(range(1, 22)), 5) == 2
    assert percentile([], 5) is None


@pytest.mark.parametrize(
    ("pred_norm", "gt_norm", "expected"),
    [
        ("9, JALAN SUBANG JASA 3,", "9", True),
        ("1,007.50", "1", True),
        ("NO 2 & 4 , JALAN BAYU 4,", "NO 2 & 4", True),  # prefix is stripped
        ("10, JALAN", "9", False),  # comma, but a different prefix
        ("9 JALAN", "9", False),  # no comma
        ("9", "9", False),  # exact match
        ("JALAN, 9", "JALAN, 9", False),  # GT itself has the comma: not a cut
        ("ABC, D", "", False),  # empty GT, non-empty prefix
        (", D", "", True),  # empty GT, empty prefix: same rule
    ],
)
def test_is_comma_cut(pred_norm, gt_norm, expected):
    assert is_comma_cut({"pred_norm": pred_norm, "gt_norm": gt_norm}) is expected


def test_histogram_bins_lower_inclusive_last_bin_closed():
    edges = [0.0, 0.8, 0.9, 1.0]
    values = [0.0, 0.5, 0.8, 0.85, 0.9, 1.0]
    assert histogram(values, edges) == [2, 2, 2]


def test_histogram_first_bin_is_underflow():
    assert histogram([0.01, 0.79], [0.0, 0.8, 1.0]) == [2, 0]


def test_histogram_empty_values():
    assert histogram([], [0.0, 0.5, 1.0]) == [0, 0]


@pytest.mark.parametrize("value", [-0.01, 1.0001])
def test_histogram_refuses_value_out_of_range(value):
    with pytest.raises(ValueError, match="outside"):
        histogram([value], [0.0, 0.5, 1.0])


@pytest.mark.parametrize("edges", [[0.5], [0.0, 0.5, 0.5], [1.0, 0.0]])
def test_histogram_refuses_bad_edges(edges):
    with pytest.raises(ValueError, match="edges"):
        histogram([0.5], edges)


def test_threshold_sweep_counts():
    rows = threshold_sweep([0.5, 0.7, 0.9, 0.95], [True, False, True, False], [0.5, 0.8, 1.0])
    assert rows == [
        {"t": 0.5, "flagged": 0, "tp": 0, "fp": 0, "fn": 2,
         "precision": None, "recall": 0.0, "f1": 0.0},
        {"t": 0.8, "flagged": 2, "tp": 1, "fp": 1, "fn": 1,
         "precision": 0.5, "recall": 0.5, "f1": 0.5},
        {"t": 1.0, "flagged": 4, "tp": 2, "fp": 2, "fn": 0,
         "precision": 0.5, "recall": 1.0, "f1": 2 / 3},
    ]  # fmt: skip


def test_threshold_sweep_none_when_no_wrong_lines():
    (row,) = threshold_sweep([0.9], [False], [0.5])
    assert row == {
        "t": 0.5, "flagged": 0, "tp": 0, "fp": 0, "fn": 0,
        "precision": None, "recall": None, "f1": None,
    }  # fmt: skip


def test_threshold_sweep_equal_confidence_is_not_flagged_and_higher_threshold_flags_all():
    rows = threshold_sweep([0.8, 0.9], [True, False], [0.8, 1.0])

    assert rows[0]["flagged"] == 0
    assert rows[0]["precision"] is None
    assert rows[0]["recall"] == 0.0
    assert rows[1]["flagged"] == 2
    assert rows[1]["tp"] == 1 and rows[1]["fp"] == 1 and rows[1]["fn"] == 0


def test_threshold_sweep_all_correct_and_unflagged_has_undefined_ratios():
    (row,) = threshold_sweep([0.9, 1.0], [False, False], [0.5])

    assert row["precision"] is None
    assert row["recall"] is None
    assert row["f1"] is None


def test_threshold_sweep_refuses_length_mismatch():
    with pytest.raises(ValueError):
        threshold_sweep([0.9, 0.8], [True], [0.5])


def test_pick_threshold_max_f1_ties_to_lower_t():
    rows = [
        {"t": 0.9, "f1": None},
        {"t": 0.8, "f1": 0.4},
        {"t": 0.7, "f1": 0.5},
        {"t": 0.6, "f1": 0.5},
    ]
    assert pick_threshold(rows)["t"] == 0.6


def test_pick_threshold_all_none():
    assert pick_threshold([{"t": 0.5, "f1": None}]) is None
    assert pick_threshold([]) is None


def test_pick_budget_highest_t_within_flagged_rate():
    rows = [{"t": t, "flagged": f} for t, f in [(0.5, 0), (0.6, 1), (0.7, 3), (0.8, 4)]]
    assert pick_budget(rows, total=20, max_flagged_rate=0.15)["t"] == 0.7  # 3/20 = 0.15, inclusive
    assert pick_budget(rows, total=20, max_flagged_rate=0.1)["t"] == 0.6


def test_pick_budget_none_when_no_row_fits():
    assert pick_budget([{"t": 0.5, "flagged": 5}], total=10, max_flagged_rate=0.15) is None
    assert pick_budget([{"t": 0.5, "flagged": 0}], total=0, max_flagged_rate=0.15) is None
    assert pick_budget([], total=10, max_flagged_rate=0.15) is None


def test_thresholds_grid():
    assert len(THRESHOLDS) == 59  # 0.50..0.99 (50) + 0.991..0.999 (9); 0.99/0.995/0.999 deduped
    assert len(set(THRESHOLDS)) == len(THRESHOLDS)
    assert THRESHOLDS[:2] == (0.5, 0.51)
    assert THRESHOLDS[-11:] == (
        0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999
    )  # fmt: skip
    assert 0.8 in THRESHOLDS and 0.57 in THRESHOLDS  # rounded, no 0.5700000000000001
    assert list(THRESHOLDS) == sorted(THRESHOLDS)


def test_import_leaves_heavy_libraries_out():
    code = (
        "import sys, ml.ocr_confidence; "
        "loaded = [m for m in ('paddle', 'paddleocr', 'cv2', 'datasets', 'PIL') "
        "if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

"""Detection metrics on hand-made boxes. Boxes are [x0, y0, x1, y1] pixels."""

import math
import random
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from ml.ocr_eval import detection_counts, iou, match_greedy, micro, scores

A = [0, 0, 10, 10]  # area 100


def test_identical_boxes_have_iou_one():
    assert iou(A, A) == 1.0


def test_disjoint_and_touching_boxes_have_iou_zero():
    assert iou(A, [20, 20, 30, 30]) == 0.0
    assert iou(A, [10, 0, 20, 10]) == 0.0  # shares an edge, no area


def test_partial_overlap_iou():
    # overlap 5x10 = 50, union 150
    assert iou(A, [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_fractional_and_integer_coordinates_have_hand_computed_iou():
    # Each area is 16; intersection is 2.5 * 2.5 = 6.25; union is 25.75.
    fractional, integer = [0.5, 0.5, 4.5, 4.5], [2, 2, 6, 6]
    assert iou(fractional, integer) == pytest.approx(25 / 103)
    assert iou(integer, fractional) == pytest.approx(25 / 103)
    assert 0 <= iou(fractional, integer) <= 1
    assert iou(fractional, fractional) == 1.0


@pytest.mark.parametrize("m", [1e154, 1e308])
def test_identical_huge_finite_box_has_iou_one(m):
    # Unscaled, area (1e308^2) or union (2 * 1e154^2 - inter) overflows.
    box = [0.0, 0.0, m, m]
    assert iou(box, box) == 1.0


def test_coordinates_spanning_the_float_range_do_not_overflow():
    # Unscaled, x1 - x0 = 2e308 is already inf before any multiplication.
    big = [-1e308, -1e308, 1e308, 1e308]
    half = [0.0, -1e308, 1e308, 1e308]
    assert iou(big, big) == 1.0
    assert iou(big, half) == pytest.approx(0.5)


def test_iou_properties_on_random_boxes_of_any_magnitude():
    rng = random.Random(0)

    def box(scale):
        x0, x1 = sorted(rng.uniform(-scale, scale) for _ in range(2))
        y0, y1 = sorted(rng.uniform(-scale, scale) for _ in range(2))
        assert x1 > x0 and y1 > y0
        return [x0, y0, x1, y1]

    overlapping = 0
    for _ in range(500):
        scale = 10.0 ** rng.uniform(-3, 300)
        a, b = box(scale), box(scale)
        value = iou(a, b)
        assert 0.0 <= value <= 1.0  # also false for NaN
        assert iou(b, a) == value
        assert iou(a, a) == 1.0
        for k in (-40, -1, 1, 13):
            scaled = ([math.ldexp(v, k) for v in a], [math.ldexp(v, k) for v in b])
            assert iou(*scaled) == value
        overlapping += value > 0
    assert overlapping > 100  # the overlap branch is exercised, not only disjoint pairs


def test_thin_box_spanning_the_float_range():
    # One shared scale for both axes underflowed the 1e-100 height to 0.
    a = [0, 0, 1e308, 1e-100]
    assert iou(a, a) == 1.0
    assert iou(a, [0, 0, 1e308, 0.5e-100]) == 0.5
    assert iou([0, -1e-100, 1e308, 0], [0, -1e-100, 1e308, 0]) == 1.0


def test_true_iou_just_below_threshold_never_matches_even_if_float_rounds_to_it():
    gt = [0, 0, 1, 2**60 + 1]  # ints: exact, unlike float(2**60 + 1)
    pred = [0, 0, 1, 2**59]  # inside gt; IoU = 2**59 / (2**60 + 1), just below 0.5
    assert iou(pred, gt) == 0.5  # the correctly rounded float
    assert detection_counts([pred], [gt]) == {"tp": 0, "fp": 1, "fn": 1}


@pytest.mark.parametrize("bad", [[0, 0, True, 10], [False, 0, 10, 10]])
def test_bool_coordinates_are_invalid(bad):
    assert iou(A, bad) == 0.0


try:
    import numpy as np
except ImportError:  # numpy rows are skipped, the rest of the matrix still runs
    np = None
needs_numpy = pytest.mark.skipif(np is None, reason="numpy not installed")


class _RaisingSequence:
    """A 'box' whose item access blows up: iou() must still return 0.0."""

    def __len__(self):
        return 4

    def __getitem__(self, i):
        raise RuntimeError("boom")


def _np(kind, values):  # build numpy scalars lazily so collection works without numpy
    return lambda: [getattr(np, kind)(v) for v in values]


BIG = 10**400  # an int far beyond float range: valid, never converted to float
VALID = [
    pytest.param(lambda: (A, [5, 0, 15, 10]), 1 / 3, id="ints"),
    pytest.param(lambda: ([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 5.0]), 0.5, id="floats"),
    pytest.param(lambda: (A, [0.0, 0.0, 10.0, 5.0]), 0.5, id="int-float-mix"),
    pytest.param(lambda: ([0, 0, BIG, BIG], [0, 0, BIG, BIG]), 1.0, id="10**400-identical"),
    pytest.param(lambda: ([0, 0, BIG, BIG], [0, 0, BIG, BIG // 2]), 0.5, id="10**400-half"),
    pytest.param(lambda: ([0, 0, 1e308, 1e-100],) * 2, 1.0, id="thin-identical"),
    pytest.param(lambda: ([0, 0, 1e308, 1e-100], [0, 0, 1e308, 0.5e-100]), 0.5, id="thin-half"),
    pytest.param(lambda: ([-10, -10, 0, 0], [-10, -10, 0, -5]), 0.5, id="negative"),
    pytest.param(
        lambda: (_np("float64", [0, 0, 10, 10])(), A), 1.0, id="numpy-float64", marks=needs_numpy
    ),
]
INVALID = [
    pytest.param(lambda: [0, 0, True, 10], id="bool-True"),
    pytest.param(lambda: [False, 0, 10, 10], id="bool-False"),
    pytest.param(lambda: [0, 0, Fraction(BIG, 3), 10], id="Fraction-10**400/3"),
    pytest.param(lambda: [0, 0, Fraction(1, 2), 10], id="Fraction-1/2"),
    pytest.param(lambda: [0, 0, Decimal("1"), 10], id="Decimal"),
    pytest.param(lambda: [0, 0, complex(1, 0), 10], id="complex"),
    pytest.param(lambda: [0, 0, "1", 10], id="str"),
    pytest.param(lambda: [0, 0, None, 10], id="None-coord"),
    pytest.param(lambda: None, id="None-box"),
    pytest.param(lambda: [0, 0, math.nan, 10], id="nan"),
    pytest.param(lambda: [0, 0, math.inf, 10], id="inf"),
    pytest.param(lambda: [-math.inf, 0, 10, 10], id="-inf"),
    pytest.param(_np("int64", [0, 0, 10, 10]), id="numpy-int64", marks=needs_numpy),
    pytest.param(_np("float32", [0, 0, 10, 10]), id="numpy-float32", marks=needs_numpy),
    pytest.param(lambda: [0, 0, 10], id="3-values"),
    pytest.param(lambda: [0, 0, 10, 10, 1], id="5-values"),
    pytest.param(lambda: object(), id="object"),
    pytest.param(lambda: 5, id="int-not-a-box"),
    pytest.param(lambda: _RaisingSequence(), id="raising-sequence"),
    pytest.param(lambda: [10, 0, 0, 10], id="inverted"),
    pytest.param(lambda: [5, 0, 5, 10], id="zero-width"),
]


@pytest.mark.parametrize(("make", "expected"), VALID)
def test_iou_contract_valid_boxes_give_exact_result(make, expected):
    a, b = make()
    assert iou(a, b) == expected
    assert iou(b, a) == expected


@pytest.mark.parametrize("make", INVALID)
def test_iou_contract_invalid_box_gives_zero_as_a_or_b_and_never_raises(make):
    bad = make()
    assert iou(A, bad) == 0.0
    assert iou(bad, A) == 0.0
    assert iou(bad, bad) == 0.0
    assert detection_counts([bad], [A]) == {"tp": 0, "fp": 1, "fn": 1}


def _reference_iou(a, b):
    """Independent exact IoU straight from the definition."""
    ax0, ay0, ax1, ay1 = map(Fraction, a)
    bx0, by0, bx1, by1 = map(Fraction, b)
    w = max(Fraction(0), min(ax1, bx1) - max(ax0, bx0))
    h = max(Fraction(0), min(ay1, by1) - max(ay0, by0))
    inter = w * h
    return float(inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter))


def test_iou_equals_exact_reference_at_mixed_magnitudes_per_axis():
    rng = random.Random(0)

    def span(scale):
        lo, hi = sorted(rng.uniform(-scale, scale) for _ in range(2))
        assert hi > lo
        return lo, hi

    overlapping = 0
    for _ in range(500):
        sx, sy = 10.0 ** rng.uniform(-300, 300), 10.0 ** rng.uniform(-300, 300)
        (ax0, ax1), (ay0, ay1) = span(sx), span(sy)
        (bx0, bx1), (by0, by1) = span(sx), span(sy)
        a, b = [ax0, ay0, ax1, ay1], [bx0, by0, bx1, by1]
        value = iou(a, b)
        assert value == _reference_iou(a, b)
        assert 0.0 <= value <= 1.0  # also false for NaN
        assert iou(b, a) == value
        assert iou(a, a) == 1.0
        overlapping += value > 0
    assert overlapping > 100  # the exact-arithmetic branch is exercised, not only fast rejects


@pytest.mark.parametrize(
    "bad",
    [[5, 5, 5, 9], [5, 5, 9, 5], [9, 0, 1, 10], [0, 0, 10], [0, 0, float("nan"), 10], []],
    ids=["zero-width", "zero-height", "inverted", "three-values", "nan", "empty"],
)
def test_invalid_or_degenerate_box_gives_zero_not_crash(bad):
    assert iou(A, bad) == 0.0
    assert iou(bad, A) == 0.0


@pytest.mark.parametrize("bad", [[0, 0, "bad", 10], None])
def test_nonnumeric_or_nonsequence_box_gives_zero_not_crash(bad):
    assert iou(A, bad) == 0.0
    assert iou(bad, A) == 0.0


def test_iou_exactly_threshold_matches():
    # pred is the top half of the GT: overlap 50, union 100, IoU exactly 0.5
    pred = [0, 0, 10, 5]
    assert iou(pred, A) == 0.5
    assert detection_counts([pred], [A]) == {"tp": 1, "fp": 0, "fn": 0}


def test_overlap_above_and_below_threshold():
    above = [0, 0, 10, 6]  # IoU 0.6
    below = [0, 0, 10, 4]  # IoU 0.4
    assert detection_counts([above], [A]) == {"tp": 1, "fp": 0, "fn": 0}
    assert detection_counts([below], [A]) == {"tp": 0, "fp": 1, "fn": 1}


def test_no_overlap_is_one_fp_and_one_fn():
    assert detection_counts([[50, 50, 60, 60]], [A]) == {"tp": 0, "fp": 1, "fn": 1}


def test_one_gt_two_preds_gives_one_tp_one_fp():
    assert detection_counts([A, [0, 0, 10, 9]], [A]) == {"tp": 1, "fp": 1, "fn": 0}


def test_many_predictions_compete_for_one_gt():
    preds = [A, A, A, A, A]
    assert match_greedy(preds, [A]) == [(0, 0, 1.0)]
    assert detection_counts(preds, [A]) == {"tp": 1, "fp": 4, "fn": 0}


def test_one_pred_over_two_gts_gives_one_tp_one_fn():
    # A merged OCR line covering two GT boxes matches at most one of them.
    left, right = [0, 0, 10, 10], [10, 0, 20, 10]
    merged = [0, 0, 18, 10]  # IoU 10/18 with left, 8/20 with right
    assert detection_counts([merged], [left, right]) == {"tp": 1, "fp": 0, "fn": 1}


def test_greedy_takes_highest_iou_pair_first():
    preds = [[0, 0, 10, 7], [0, 0, 10, 9]]  # IoU 0.7, 0.9
    assert match_greedy(preds, [A]) == [(1, 0, pytest.approx(0.9))]


def test_ties_broken_by_pred_then_gt_index():
    assert [(p, g) for p, g, _ in match_greedy([A, A], [A, A])] == [(0, 0), (1, 1)]


def test_threshold_is_configurable():
    pred = [0, 0, 10, 6]  # IoU 0.6
    assert detection_counts([pred], [A], threshold=0.7) == {"tp": 0, "fp": 1, "fn": 1}


def test_empty_preds():
    counts = detection_counts([], [A, A])
    assert counts == {"tp": 0, "fp": 0, "fn": 2}
    assert scores(**counts) == {"precision": None, "recall": 0.0, "f1": 0.0}


def test_empty_gt():
    counts = detection_counts([A], [])
    assert counts == {"tp": 0, "fp": 1, "fn": 0}
    assert scores(**counts) == {"precision": 0.0, "recall": None, "f1": 0.0}


def test_both_empty_gives_null_scores():
    counts = detection_counts([], [])
    assert counts == {"tp": 0, "fp": 0, "fn": 0}
    assert scores(**counts) == {"precision": None, "recall": None, "f1": None}


def test_zero_precision_and_recall_gives_f1_zero():
    assert scores(tp=0, fp=3, fn=2) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_scores_values():
    s = scores(tp=3, fp=1, fn=2)
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_degenerate_gt_box_is_a_miss():
    assert detection_counts([A], [A, [3, 3, 3, 3]]) == {"tp": 1, "fp": 0, "fn": 1}


def test_micro_sums_counts_and_differs_from_mean_of_per_image_scores():
    per_image = [{"tp": 1, "fp": 0, "fn": 0}, {"tp": 1, "fp": 9, "fn": 0}]
    m = micro(per_image)

    assert {k: m[k] for k in ("tp", "fp", "fn")} == {"tp": 2, "fp": 9, "fn": 0}
    assert m["precision"] == pytest.approx(2 / 11)
    mean_precision = (1.0 + 0.1) / 2
    assert m["precision"] != pytest.approx(mean_precision)


def test_micro_of_nothing():
    assert micro([]) == {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}


def test_import_does_not_load_paddle_cv2_or_datasets():
    code = (
        "import sys, ml.ocr_eval; "
        "loaded = [m for m in ('paddle', 'paddleocr', 'cv2', 'datasets') if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

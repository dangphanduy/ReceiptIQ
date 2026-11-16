"""Detection metrics on hand-made boxes. Boxes are [x0, y0, x1, y1] pixels."""

import subprocess
import sys

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

"""Reading-order sorting and row grouping on synthetic OcrLines. Boxes are [x0, y0, x1, y1]."""

import dataclasses
import random
import subprocess
import sys

import pytest

from ml.ocr_schema import OcrLine
from ml.reading_order import ROW_OVERLAP_MIN, OcrRow, group_rows


def line(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.9) -> OcrLine:
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return OcrLine(text, conf, polygon, [x0, y0, x1, y1])


# Two-column receipt block, amounts drift a few px lower to the right (slight skew).
RECEIPT = [
    line("ITEM A", 10, 100, 80, 120, 0.95),
    line("5.00", 300, 103, 340, 123, 0.91),
    line("ITEM B", 10, 130, 80, 150, 0.93),
    line("7.50", 300, 134, 340, 154, 0.88),
    line("TOTAL", 10, 160, 90, 180, 0.99),
    line("12.50", 300, 165, 350, 185, 0.97),
]


def test_empty_input_gives_no_rows():
    assert group_rows([]) == []


def test_single_line_is_one_row():
    rows = group_rows([line("SHOP SDN BHD", 10, 5, 200, 25, 0.8)])
    assert rows == [OcrRow("SHOP SDN BHD", [10, 5, 200, 25], (0,), (0.8,))]


def test_label_and_amount_on_one_row_read_left_to_right():
    amount, label = line("12.50", 300, 100, 350, 120, 0.7), line("TOTAL", 10, 100, 90, 120, 0.6)
    rows = group_rows([amount, label])
    assert rows == [OcrRow("TOTAL 12.50", [10, 100, 350, 120], (1, 0), (0.6, 0.7))]


def test_two_separate_rows_do_not_merge_and_read_top_to_bottom():
    rows = group_rows([line("SECOND", 10, 25, 90, 45), line("FIRST", 10, 0, 90, 20)])
    assert [r.text for r in rows] == ["FIRST", "SECOND"]
    assert [r.source_indices for r in rows] == [(1,), (0,)]


def test_slight_skew_still_groups_columns_per_row():
    rows = group_rows(RECEIPT)
    assert [r.text for r in rows] == ["ITEM A 5.00", "ITEM B 7.50", "TOTAL 12.50"]
    assert rows[2].bbox == [10, 160, 350, 185]
    assert rows[2].confidences == (0.99, 0.97)


def test_overlap_threshold_is_inclusive_at_half_the_smaller_height():
    assert ROW_OVERLAP_MIN == 0.5
    # heights 20, overlap 10 -> 0.5 -> same row; overlap 9 -> 0.45 -> separate rows
    assert len(group_rows([line("A", 0, 0, 50, 20), line("B", 100, 10, 150, 30)])) == 1
    assert len(group_rows([line("A", 0, 0, 50, 20), line("B", 100, 11, 150, 31)])) == 2


def test_rows_with_different_heights():
    big_total = line("TOTAL", 10, 100, 120, 140)  # large font, height 40
    small_rm = line("RM 12.50", 200, 110, 260, 125)  # height 15, fully inside
    next_row = line("CASH", 10, 145, 60, 160)
    rows = group_rows([next_row, small_rm, big_total])
    assert [r.text for r in rows] == ["TOTAL RM 12.50", "CASH"]
    assert [r.source_indices for r in rows] == [(2, 1), (0,)]


def test_tall_box_joins_one_row_and_does_not_chain_the_rows_it_spans():
    # Shape seen in X51006388089: a stamp box spanning three printed rows on the right.
    lines = [
        line("1082", 450, 0, 670, 146),
        line("TAX", 55, 11, 115, 46),
        line("0.00", 212, 11, 277, 49),
        line("ROUNDING", 55, 43, 219, 81),
        line("0.01", 272, 46, 335, 81),
        line("TOTAL", 54, 77, 151, 113),
        line("103.00", 214, 78, 309, 117),
    ]
    rows = group_rows(lines)
    assert [r.text for r in rows] == ["TAX 0.00 1082", "ROUNDING 0.01", "TOTAL 103.00"]


def test_shuffled_input_gives_the_same_rows():
    expected = group_rows(RECEIPT)
    for seed in range(5):
        shuffled = RECEIPT[:]
        random.Random(seed).shuffle(shuffled)
        rows = group_rows(shuffled)
        assert [(r.text, r.bbox, r.confidences) for r in rows] == [
            (r.text, r.bbox, r.confidences) for r in expected
        ]
        # indices point into the shuffled input and select the same source lines
        assert [[shuffled[i] for i in r.source_indices] for r in rows] == [
            [RECEIPT[i] for i in r.source_indices] for r in expected
        ]


def test_every_source_index_appears_exactly_once():
    rows = group_rows(RECEIPT)
    flat = [i for r in rows for i in r.source_indices]
    assert sorted(flat) == list(range(len(RECEIPT)))
    for r in rows:
        assert len(r.source_indices) == len(r.confidences)
        assert r.text == " ".join(RECEIPT[i].text for i in r.source_indices)


def test_input_is_not_mutated():
    lines = RECEIPT[:]
    snapshot = [(ln.text, ln.confidence, list(ln.bbox), [p[:] for p in ln.polygon]) for ln in lines]
    group_rows(lines)
    assert lines == RECEIPT
    assert [
        (ln.text, ln.confidence, list(ln.bbox), [p[:] for p in ln.polygon]) for ln in lines
    ] == snapshot


def test_raw_text_is_kept_not_normalized():
    rows = group_rows([line("Total :", 10, 0, 60, 20), line("rm12.50", 100, 0, 160, 20)])
    assert rows[0].text == "Total : rm12.50"


def test_zero_height_box_is_its_own_row_without_crashing():
    rows = group_rows([line("A", 0, 10, 50, 10), line("B", 100, 0, 150, 20)])
    assert [r.text for r in rows] == ["B", "A"]


def test_row_is_frozen():
    row = group_rows([line("A", 0, 0, 10, 10)])[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.text = "B"


def test_import_does_not_load_heavy_modules():
    code = (
        "import sys, ml.reading_order; "
        "loaded = [m for m in ('paddle', 'paddleocr', 'cv2', 'datasets', 'PIL', 'numpy') "
        "if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

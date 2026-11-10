import json

import numpy as np
import pytest

from ml.ocr_schema import SCHEMA_VERSION, build_ocr_document, normalize_paddle_result


def paddle_like(texts, scores, polys):
    """Mimic the dict-like PaddleOCR 3.x result: lists of str/float, numpy polygons."""
    return {
        "rec_texts": texts,
        "rec_scores": np.array(scores, dtype=np.float32),
        "rec_polys": [np.array(p, dtype=np.int16) for p in polys],
    }


SQUARE = [[10, 20], [110, 20], [110, 40], [10, 40]]


def test_normalizes_numpy_values_to_plain_python():
    lines = normalize_paddle_result(paddle_like(["  TOTAL 31.00 "], [0.98], [SQUARE]))

    assert len(lines) == 1
    line = lines[0]
    assert line.text == "TOTAL 31.00"
    assert line.confidence == pytest.approx(0.98)
    assert type(line.confidence) is float
    assert line.polygon == [[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]]
    assert line.bbox == [10.0, 20.0, 110.0, 40.0]


def test_bbox_covers_rotated_polygon():
    tilted = [[12, 18], [110, 22], [108, 44], [10, 40]]
    (line,) = normalize_paddle_result(paddle_like(["x"], [0.5], [tilted]))
    assert line.bbox == [10.0, 18.0, 110.0, 44.0]


def test_drops_blank_text():
    lines = normalize_paddle_result(paddle_like(["a", "   ", ""], [0.9, 0.9, 0.9], [SQUARE] * 3))
    assert [line.text for line in lines] == ["a"]


def test_empty_result_is_ok():
    assert normalize_paddle_result({"rec_texts": [], "rec_scores": [], "rec_polys": []}) == []


@pytest.mark.parametrize(
    "raw",
    [
        paddle_like(["a", "b"], [0.9], [SQUARE, SQUARE]),  # length mismatch
        paddle_like(["a"], [1.5], [SQUARE]),  # confidence > 1
        paddle_like(["a"], [float("nan")], [SQUARE]),  # NaN confidence
        paddle_like(["a"], [0.9], [[[0, 0], [1, 1]]]),  # too few points
    ],
)
def test_rejects_malformed_results(raw):
    with pytest.raises(ValueError):
        normalize_paddle_result(raw)


def test_document_is_json_serializable_and_complete():
    lines = normalize_paddle_result(paddle_like(["A", "B"], [0.8, 0.6], [SQUARE, SQUARE]))
    doc = build_ocr_document("r.jpg", 463, 605, lines, {"name": "paddleocr"})

    roundtrip = json.loads(json.dumps(doc))
    assert roundtrip["schema_version"] == SCHEMA_VERSION
    assert roundtrip["image"] == {"path": "r.jpg", "width": 463, "height": 605}
    assert roundtrip["num_lines"] == 2
    assert roundtrip["mean_confidence"] == pytest.approx(0.7)
    assert set(roundtrip["lines"][0]) == {"text", "confidence", "polygon", "bbox"}


def test_document_without_lines_has_null_mean():
    doc = build_ocr_document("r.jpg", 1, 1, [], {})
    assert doc["num_lines"] == 0
    assert doc["mean_confidence"] is None

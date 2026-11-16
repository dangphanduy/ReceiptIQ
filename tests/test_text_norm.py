import subprocess
import sys

import pytest

from ml.text_norm import normalize_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("   ", ""),
        ("total", "TOTAL"),
        ("  Total  Rm  ", "TOTAL RM"),
        ("a\tb\nc\r\nd", "A B C D"),
        ("12 345", "12 345"),  # non-breaking space is whitespace
        ("a \t\n  b", "A B"),
        ("café", "CAFÉ"),
        ("straße", "STRASSE"),  # "ß".upper() == "SS": expected, length grows by one
        ("𝐀b", "𝐀B"),  # non-BMP char kept as is
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_is_idempotent():
    s = " Gst  Reg  No:\t001 "
    assert normalize_text(normalize_text(s)) == normalize_text(s)


def test_import_does_not_load_paddle_cv2_or_datasets():
    code = (
        "import sys, ml.text_norm, ml.ocr_recognition; "
        "loaded = [m for m in ('paddle', 'paddleocr', 'cv2', 'datasets') if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

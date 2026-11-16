"""The one text normalization used to compare OCR text with ground truth.

Pure Python: never import paddle, cv2 or datasets here.
"""


def normalize_text(s: str) -> str:
    """Uppercase, collapse every whitespace run to one space, strip both ends.

    Whitespace is what str.split() splits on (tabs, newlines, U+00A0 non-breaking space, ...).
    Uppercasing is Unicode-aware and may change length: "ß".upper() == "SS". No Unicode
    normalization (NFC/NFKC) and no punctuation removal.

    This is the single normalization for ReceiptIQ text comparisons: OCR recognition metrics
    (ml.ocr_recognition, RIQ-20) use it, and field evaluation (E3 / RIQ-29) must import this
    same function instead of writing its own copy.
    """
    return " ".join(s.upper().split())

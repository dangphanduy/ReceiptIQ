"""The one place that configures PaddleOCR and turns image bytes into an OCR document.

Shared by scripts/run_ocr_baseline.py and scripts/run_ocr_batch.py so they cannot drift.
paddleocr is imported only inside `create_engine`, so `run_ocr` stays testable with a fake.
"""

import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Protocol

import cv2  # opencv-contrib-python, installed by paddleocr -> paddlex; never add another wheel
import numpy as np

from ml.ocr_schema import build_ocr_document, normalize_paddle_result

# Scanned receipts are upright and flat; disable the extra doc-preprocessing models.
OCR_CONFIG = {
    "lang": "en",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    # ponytail: paddlepaddle 3.3.1 oneDNN CPU kernels crash on PIR models on Windows
    # (ConvertPirAttribute2RuntimeAttribute). Slower without it; retry True on upgrades.
    "enable_mkldnn": False,
}


class OcrEngine(Protocol):
    """What run_ocr needs from an engine: PaddleOCR 3.x, or a fake in tests."""

    def predict(self, input: np.ndarray) -> Sequence[Mapping]:
        """BGR uint8 HxWx3 image in; one dict-like result per image out."""
        ...


def engine_versions() -> dict:
    return {
        "paddleocr_version": version("paddleocr"),
        "paddlepaddle_version": version("paddlepaddle"),
    }


def create_engine() -> OcrEngine:
    """Build the PaddleOCR pipeline (slow: loads models). Build once, reuse per image."""
    from paddleocr import PaddleOCR

    return PaddleOCR(**OCR_CONFIG)


def run_ocr(engine: OcrEngine, image_bytes: bytes, image_path: str) -> dict:
    """OCR one encoded image (JPEG/PNG bytes) with `engine`; return the versioned OCR document.

    `image_path` is only recorded in the document, never opened.
    """
    # Decode exactly as PaddleX does for a file path (readers.py: cv2.imdecode, BGR).
    # PIL decodes JPEG differently, which would change OCR output vs. passing the file.
    # IMREAD_COLOR applies EXIF orientation; width/height below match the boxes on purpose.
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image {image_path}")
    height, width = bgr.shape[:2]
    start = time.perf_counter()
    results = engine.predict(bgr)
    elapsed = time.perf_counter() - start

    engine_meta = {
        "name": "paddleocr",
        **engine_versions(),
        "lang": OCR_CONFIG["lang"],
        "device": "cpu",
        "inference_seconds": round(elapsed, 3),
    }
    lines = normalize_paddle_result(results[0])
    return build_ocr_document(image_path, width, height, lines, engine_meta)

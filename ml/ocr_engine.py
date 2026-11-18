"""The one place that configures PaddleOCR and turns image bytes into an OCR document.

Shared by scripts/run_ocr_baseline.py and scripts/run_ocr_batch.py so they cannot drift.
paddleocr is imported only inside `create_engine`, so `run_ocr` stays testable with a fake.
"""

import argparse
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
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

DEVICE = re.compile(r"cpu|gpu:\d+", re.ASCII)  # passed to PaddleOCR(device=...) as-is
PADDLE_DISTRIBUTIONS = ("paddlepaddle", "paddlepaddle-gpu")


class OcrEngine(Protocol):
    """What run_ocr needs from an engine: PaddleOCR 3.x, or a fake in tests."""

    def predict(self, input: np.ndarray) -> Sequence[Mapping]:
        """BGR uint8 HxWx3 image in; one dict-like result per image out."""
        ...


def parse_device(value: str) -> str:
    """argparse `type=` for --device: "cpu" or "gpu:<int>"."""
    if not DEVICE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"expected 'cpu' or 'gpu:<int>', got {value!r}")
    return value


def engine_versions() -> dict:
    """paddleocr version plus whichever paddle distribution (CPU or GPU build) is installed."""
    found = {}
    for dist in PADDLE_DISTRIBUTIONS:
        with suppress(PackageNotFoundError):
            found[dist] = version(dist)
    if len(found) != 1:  # both share the `paddle` package; which one is live is unknowable
        raise RuntimeError(
            f"expected exactly one of {PADDLE_DISTRIBUTIONS} installed, found {sorted(found)}"
        )
    ((dist, paddle_version),) = found.items()
    return {
        "paddleocr_version": version("paddleocr"),
        "paddlepaddle_distribution": dist,
        "paddlepaddle_version": paddle_version,
    }


def create_engine(device: str = "cpu") -> OcrEngine:
    """Build the PaddleOCR pipeline on `device` (slow: loads models). Build once, reuse.

    Refuses a GPU that this paddle build cannot use, so the device recorded is the one used.
    """
    if not DEVICE.fullmatch(device):
        raise ValueError(f"device must be 'cpu' or 'gpu:<int>', got {device!r}")
    # Keep this before `import paddle`: paddlex loads torch (via modelscope), and torch's
    # cuDNN DLLs fail to load (WinError 127) once paddlepaddle-gpu has loaded its own.
    from paddleocr import PaddleOCR

    if device != "cpu":
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError(f"{device} needs paddlepaddle-gpu; this env has the CPU build")
        if int(device.split(":")[1]) >= paddle.device.cuda.device_count():
            raise RuntimeError(f"{device} not found: {paddle.device.cuda.device_count()} GPU(s)")
    return PaddleOCR(**OCR_CONFIG, device=device)


def run_ocr(engine: OcrEngine, image_bytes: bytes, image_path: str, *, device: str) -> dict:
    """OCR one encoded image (JPEG/PNG bytes) with `engine`; return the versioned OCR document.

    `image_path` is only recorded in the document, never opened. `device` is the one
    `engine` was built on (create_engine); it is recorded as engine.device.
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
        "device": device,
        "inference_seconds": round(elapsed, 3),
    }
    lines = normalize_paddle_result(results[0])
    return build_ocr_document(image_path, width, height, lines, engine_meta)

"""Engine-independent OCR result schema and PaddleOCR 3.x normalization.

Kept free of paddle imports so it can be unit-tested without the OCR model.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float  # recognition score in [0, 1]
    polygon: list[list[float]]  # [[x, y], ...] clockwise from top-left, pixel coords
    bbox: list[float]  # [x_min, y_min, x_max, y_max] (same convention as SROIE `bboxes`)


def normalize_paddle_result(raw: Mapping) -> list[OcrLine]:
    """Convert one PaddleOCR 3.x `predict()` result (dict-like) into OcrLines.

    Uses `rec_texts`, `rec_scores`, `rec_polys`; values may be numpy arrays.
    Lines whose text is empty after stripping are dropped.
    """
    texts, scores, polys = raw["rec_texts"], raw["rec_scores"], raw["rec_polys"]
    if not (len(texts) == len(scores) == len(polys)):
        raise ValueError(
            f"length mismatch: {len(texts)} texts, {len(scores)} scores, {len(polys)} polys"
        )

    lines = []
    for text, score, poly in zip(texts, scores, polys, strict=True):
        text = str(text).strip()
        if not text:
            continue
        confidence = float(score)
        if math.isnan(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence out of [0, 1]: {score!r} for {text!r}")
        polygon = _to_polygon(poly)
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        lines.append(OcrLine(text, confidence, polygon, [min(xs), min(ys), max(xs), max(ys)]))
    return lines


def _to_polygon(poly: Sequence) -> list[list[float]]:
    points = [[float(p[0]), float(p[1])] for p in poly]
    if len(points) < 4:
        raise ValueError(f"polygon needs >= 4 points, got {len(points)}")
    return points


def build_ocr_document(
    image_path: str, width: int, height: int, lines: list[OcrLine], engine: Mapping
) -> dict:
    """Assemble the JSON-serializable OCR output for one image."""
    confidences = [line.confidence for line in lines]
    return {
        "schema_version": SCHEMA_VERSION,
        "image": {"path": image_path, "width": width, "height": height},
        "engine": dict(engine),
        "num_lines": len(lines),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "lines": [asdict(line) for line in lines],
    }

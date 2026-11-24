"""Reading order for OCR lines: group boxes on the same visual row, top to bottom, left to right.

A receipt row such as `TOTAL ... 12.50` often comes out of OCR as two boxes (label, amount).
`group_rows` merges them into one `OcrRow` and keeps the indices of the source lines, so later
field extraction can point back to the original OCR evidence. Text stays raw (no
`normalize_text`); normalization belongs to evaluation.

Boxes are `OcrLine.bbox` = [x_min, y_min, x_max, y_max] pixels.
Pure Python on purpose: never import paddle, cv2, datasets, PIL or numpy here.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ml.ocr_schema import OcrLine

# Half the smaller box height: tolerates a few px of skew between columns, while two stacked
# rows of similar height overlap only marginally and stay apart.
ROW_OVERLAP_MIN = 0.5


@dataclass(frozen=True)
class OcrRow:
    text: str  # source texts in x order, joined by one space
    bbox: tuple[float, float, float, float]  # union of the source boxes, (x0, y0, x1, y1)
    source_indices: tuple[int, ...]  # indices into the input sequence, in x order
    confidences: tuple[float, ...]  # source confidences, same order as source_indices


def _same_row(a: Sequence[float], b: Sequence[float]) -> bool:
    """Vertical overlap of a and b >= ROW_OVERLAP_MIN of the smaller height; zero height: never."""
    smaller = min(a[3] - a[1], b[3] - b[1])
    if smaller <= 0:
        return False
    return min(a[3], b[3]) - max(a[1], b[1]) >= ROW_OVERLAP_MIN * smaller


def group_rows(lines: Sequence[OcrLine]) -> list[OcrRow]:
    """Group lines into visual rows, ordered top to bottom; lines in a row go left to right.

    Lines are visited by (y_min, x_min, index). Each joins the current row if it is `_same_row`
    with every line already in it, otherwise it starts a new row. Testing every member, not the
    growing union box, keeps a tall box (stamp) or a skewed scan from chaining rows together.
    The input is not modified.
    """
    order = sorted(range(len(lines)), key=lambda i: (lines[i].bbox[1], lines[i].bbox[0], i))
    groups: list[tuple[list[int], list[float]]] = []  # (member indices, union bbox)
    for i in order:
        box = lines[i].bbox
        if groups and all(_same_row(lines[j].bbox, box) for j in groups[-1][0]):
            members, union = groups[-1]
            members.append(i)
            union[:] = [*map(min, union[:2], box[:2]), *map(max, union[2:], box[2:])]
        else:
            groups.append(([i], list(box)))

    rows = []
    for members, union in groups:
        members.sort(key=lambda i: (lines[i].bbox[0], i))
        rows.append(
            OcrRow(
                text=" ".join(lines[i].text for i in members),
                bbox=tuple(union),
                source_indices=tuple(members),
                confidences=tuple(lines[i].confidence for i in members),
            )
        )
    return rows

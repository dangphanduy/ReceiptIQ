"""OCR latency and line confidence on a SROIE split: can confidence flag wrong lines for review?

Usage:
    python scripts/eval_ocr_confidence.py --split test
    python scripts/eval_ocr_confidence.py --split test --ocr-dir outputs/ocr/test-gpu \
        --latency-dir outputs/ocr/test-cpu-latency --out outputs/eval/confidence_test.json \
        --chart outputs/eval/confidence_hist_test.png --iou 0.5

A line is a ml.ocr_recognition.line_pairs pair (IoU >= --iou); its confidence is the OCR
line's, and it is wrong when its normalized texts differ. Flag rule: confidence < t. Two views:
all pairs, and "clean" = all minus suspected GT comma-cuts (ml.ocr_confidence.is_comma_cut).
Latency is engine.inference_seconds (predict() only) from --ocr-dir and from --latency-dir
(a CPU sample; the dir is optional). Same input contract as eval_ocr_detection.py: GT columns
`key`, `words`, `bboxes`, `image_size` only; keys without OCR JSON are listed, never scored.
"""

import argparse
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from eval_ocr_detection import (
    ROOT,
    load_gt_rows,
    load_ocr,
    parse_iou_threshold,
    resolve_device,
    write_json_atomic,
)
from PIL import Image, ImageDraw, ImageFont

from ml.ocr_confidence import (
    THRESHOLDS,
    histogram,
    is_comma_cut,
    latency_stats,
    percentile,
    pick_budget,
    pick_threshold,
    threshold_sweep,
)
from ml.ocr_recognition import line_pairs

GT_COLUMNS = ["key", "words", "bboxes", "image_size"]
# one underflow bin [0, 0.80), then 0.01 bins over [0.80, 1.00] (last bin includes 1.0)
EDGES = [0.0] + [round(0.8 + i / 100, 2) for i in range(21)]
BUDGET_FLAGGED_RATE = 0.15  # "budget" point: highest t that flags at most this share of lines
CREDIT = (
    "Data: ICDAR 2019 SROIE (Huang et al., 2019) via Hugging Face jsdnrs/ICDAR2019-SROIE, "
    "CC BY 4.0 (creativecommons.org/licenses/by/4.0). Processed by ReceiptIQ OCR."
)


def _seconds(doc: dict) -> float:
    s = doc["engine"]["inference_seconds"]
    # TypeError, not ValueError: load_ocr turns TypeError into an error naming the file
    if type(s) not in (int, float) or not 0 <= s < math.inf:
        raise TypeError(f"engine.inference_seconds must be a finite number >= 0, got {s!r}")
    return float(s)


def _ocr_doc(doc: dict) -> dict:
    lines = []
    for ln in doc["lines"]:
        text, conf = ln["text"], ln["confidence"]
        if type(text) is not str:
            raise TypeError(f"line text must be str, got {type(text).__name__}")
        if type(conf) not in (int, float) or not 0 <= conf <= 1:
            raise TypeError(f"line confidence must be a number in [0, 1], got {conf!r}")
        lines.append((ln["bbox"], text, float(conf)))
    return {"lines": lines, "seconds": _seconds(doc)}


def _latency(directory: Path, device: str | None, seconds: list[float]) -> dict:
    """Stats in split order; the first image of a fresh run includes warm-up."""
    return {
        "dir": str(directory),
        "device": device,
        "stats": latency_stats(seconds),
        "first_image_seconds": seconds[0] if seconds else None,
        "stats_excluding_first": latency_stats(seconds[1:]),
    }


def _summary(values: Sequence[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p05": percentile(values, 5),
    }


def _view(pairs: Sequence[dict], confs: Sequence[float]) -> dict:
    wrong = [not p["exact"] for p in pairs]
    correct_c = [c for c, w in zip(confs, wrong, strict=True) if not w]
    wrong_c = [c for c, w in zip(confs, wrong, strict=True) if w]
    sweep = threshold_sweep(confs, wrong, THRESHOLDS)

    def point(row):
        return row and {
            **{k: row[k] for k in ("t", "precision", "recall", "f1")},
            "flagged_rate": row["flagged"] / len(pairs),
        }

    budget = point(pick_budget(sweep, len(pairs), BUDGET_FLAGGED_RATE))
    return {
        "correct": _summary(correct_c),
        "wrong": _summary(wrong_c),
        "histogram": {
            "edges": EDGES,
            "correct": histogram(correct_c, EDGES),
            "wrong": histogram(wrong_c, EDGES),
        },
        "sweep": sweep,
        "candidate": point(pick_threshold(sweep)),
        "budget": budget and {**budget, "max_flagged_rate": BUDGET_FLAGGED_RATE},
    }


def evaluate(
    gt_rows: Iterable[Mapping], ocr_dir: Path, latency_dir: Path, iou_threshold: float
) -> dict:
    """Latency (both dirs) and confidence views over all keys with OCR JSON in ocr_dir.

    latency.cpu is None when latency_dir has no OCR JSON. Raises ValueError as
    eval_ocr_detection.load_ocr does (also for bad confidence / inference_seconds), or for a GT
    row with a non-str word or words and bboxes of different length.
    """
    gt_rows = list(gt_rows)  # iterated once per dir
    loaded = load_ocr(gt_rows, ocr_dir, _ocr_doc)
    pairs, confs = [], []
    for row, doc in loaded["images"]:
        words, bboxes = row["words"], row["bboxes"]
        if len(words) != len(bboxes):
            raise ValueError(f"GT {row['key']}: {len(words)} words but {len(bboxes)} bboxes")
        for i, w in enumerate(words):
            if type(w) is not str:
                raise ValueError(f"GT {row['key']}: words[{i}] is {type(w).__name__}, not str")
        lines = doc["lines"]
        preds = [(b, t) for b, t, _ in lines]
        for p in line_pairs(
            row["key"], preds, list(zip(bboxes, words, strict=True)), iou_threshold
        ):
            pairs.append(p)
            confs.append(lines[p["pred_idx"]][2])
    keep = [i for i, p in enumerate(pairs) if not is_comma_cut(p)]
    device = resolve_device(ocr_dir, loaded["ocr_devices"])

    cpu = None
    if any(p.name != "manifest.json" for p in latency_dir.glob("*.json")):
        lat = load_ocr(gt_rows, latency_dir, _seconds)
        lat_device = resolve_device(latency_dir, lat["ocr_devices"])
        cpu = _latency(latency_dir, lat_device, [s for _, s in lat["images"]])

    n_correct = sum(p["exact"] for p in pairs)
    return {
        "images_evaluated": len(loaded["images"]),
        "ocr_device": device,
        "latency": {
            "cpu": cpu,
            "gpu": _latency(ocr_dir, device, [doc["seconds"] for _, doc in loaded["images"]]),
        },
        "counts": {
            "pairs": len(pairs),
            "correct": n_correct,
            "wrong": len(pairs) - n_correct,
            "comma_cut_removed": len(pairs) - len(keep),
        },
        "confidence": {
            "all": _view(pairs, confs),
            "clean": _view([pairs[i] for i in keep], [confs[i] for i in keep]),
        },
        "missing_keys": loaded["missing_keys"],
    }


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def draw_chart(path: Path, confidence: Mapping, counts: Mapping, subtitle: str) -> None:
    """Two panels (all / clean): share of correct vs wrong lines per confidence bin,
    candidate threshold as a dashed line. Bins: "<0.80", then 0.01 wide up to 1.00."""
    blue, orange, ink, grid = (31, 119, 180), (230, 97, 1), (40, 40, 40), (225, 225, 225)
    width, panel_h, header_h, footer_h = 1400, 470, 50, 40
    img = Image.new("RGB", (width, header_h + 2 * panel_h + footer_h), "white")
    d = ImageDraw.Draw(img)
    f_title, f_text, f_small = _font(22), _font(16), _font(13)
    d.text((20, 14), "OCR line confidence: correct vs wrong lines", fill=ink, font=f_title)
    d.text((width - 20, 20), subtitle, fill=ink, font=f_small, anchor="ra")

    n_bins = len(EDGES) - 1
    slots = n_bins + 1  # one empty slot between the underflow bin and 0.80
    titles = {
        "all": f"All matched pairs (n = {counts['pairs']})",
        "clean": f"Clean: minus {counts['comma_cut_removed']} suspected GT comma-cuts",
    }
    for row, view_name in enumerate(("all", "clean")):
        view = confidence[view_name]
        top = header_h + row * panel_h
        x0, x1, y0, y1 = 90, width - 30, top + 50, top + panel_h - 75
        slot_w = (x1 - x0) / slots
        hist = view["histogram"]
        n_c, n_w = sum(hist["correct"]), sum(hist["wrong"])
        share_c = [c / n_c if n_c else 0.0 for c in hist["correct"]]
        share_w = [c / n_w if n_w else 0.0 for c in hist["wrong"]]
        y_max = max(0.1, math.ceil(max(share_c + share_w) * 10) / 10)

        def y_of(v, y0=y0, y1=y1, y_max=y_max):
            return y1 - v / y_max * (y1 - y0)

        for k in range(round(y_max * 10) + 1):  # y grid and ticks every 0.1
            v = k / 10
            d.line([(x0, y_of(v)), (x1, y_of(v))], fill=grid)
            d.text((x0 - 8, y_of(v)), f"{v:.2f}", fill=ink, font=f_small, anchor="rm")
        for i in range(n_bins):
            slot = i if i == 0 else i + 1
            left = x0 + slot * slot_w
            bar = slot_w * 0.42
            for j, (share, colour) in enumerate(((share_c[i], blue), (share_w[i], orange))):
                bx = left + slot_w * 0.08 + j * bar
                if share > 0:
                    d.rectangle([bx, y_of(share), bx + bar - 1, y1], fill=colour)
        d.line([(x0, y1), (x1, y1)], fill=ink, width=2)
        d.line([(x0, y0), (x0, y1)], fill=ink, width=2)
        d.text((x0 + slot_w / 2, y1 + 8), "< 0.80", fill=ink, font=f_small, anchor="ma")
        for k in range(0, 21, 5):  # boundaries 0.80, 0.85, ..., 1.00
            x = x0 + (2 + k) * slot_w
            d.line([(x, y1), (x, y1 + 5)], fill=ink, width=2)
            d.text((x, y1 + 8), f"{0.8 + k / 100:.2f}", fill=ink, font=f_small, anchor="ma")
        d.text(((x0 + x1) / 2, y1 + 32), "OCR line confidence", fill=ink, font=f_text, anchor="ma")
        y_label = Image.new("RGB", (220, 24), "white")  # Pillow cannot draw rotated text directly
        ImageDraw.Draw(y_label).text(
            (110, 12), "share of class", fill=ink, font=f_text, anchor="mm"
        )
        img.paste(y_label.rotate(90, expand=True), (12, int((y0 + y1) / 2 - 110)))
        d.text((x0, top + 14), titles[view_name], fill=ink, font=f_text)

        cand = view["candidate"]
        label = [(blue, f"correct (n = {n_c})"), (orange, f"wrong (n = {n_w})")]
        if cand:
            t = cand["t"]
            x = x0 + (2 + (t - 0.8) * 100) * slot_w if t >= 0.8 else x0 + slot_w
            for yy in range(int(y0), int(y1), 12):
                d.line([(x, yy), (x, min(yy + 7, y1))], fill=ink, width=2)
            label.append(
                (
                    ink,
                    f"flag if conf < {t:g}: P {cand['precision']:.2f} R {cand['recall']:.2f} "
                    f"F1 {cand['f1']:.2f}, flags {cand['flagged_rate']:.1%}",
                )
            )
        ly = y0 + 4
        for colour, text in label:  # legend, top left of the plot (bars sit at the right)
            d.rectangle([x0 + slot_w * 2 + 10, ly, x0 + slot_w * 2 + 26, ly + 14], fill=colour)
            d.text((x0 + slot_w * 2 + 34, ly + 7), text, fill=ink, font=f_small, anchor="lm")
            ly += 22
    d.text((20, img.height - footer_h / 2), CREDIT, fill=(90, 90, 90), font=f_small, anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        img.save(tmp, format="PNG")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _fmt(v, spec=".4f"):
    return "null" if v is None else format(v, spec)


def _print_summary(report: dict, images: int, out: Path, chart: Path) -> None:
    c = report["counts"]
    print(
        f"evaluated {images}, missing {len(report['missing_keys'])} | IoU >= "
        f"{report['iou_threshold']}: pairs {c['pairs']} (correct {c['correct']}, wrong "
        f"{c['wrong']}), comma-cut removed for clean view {c['comma_cut_removed']}"
    )
    for name, lat in report["latency"].items():
        if lat is None:
            print(f"latency {name}: no latency JSON in --latency-dir; skipped")
            continue
        s, ex = lat["stats"], lat["stats_excluding_first"]
        stats = " ".join(
            f"{k} {_fmt(s[k], '.2f')}" for k in ("mean", "median", "p95", "min", "max")
        )
        print(
            f"latency {name} ({lat['device']}, {lat['dir']}): n {s['n']} {stats} s | first "
            f"{_fmt(lat['first_image_seconds'], '.2f')} s, mean without first "
            f"{_fmt(ex['mean'], '.2f')} s"
        )

    def point(label, p):
        p = p or {}
        return (
            f"{label} t {_fmt(p.get('t'), 'g')} P {_fmt(p.get('precision'))} "
            f"R {_fmt(p.get('recall'))} F1 {_fmt(p.get('f1'))} "
            f"flagged {_fmt(p.get('flagged_rate'), '.2%')}"
        )

    for name, view in report["confidence"].items():
        print(
            f"{name}: conf median correct {_fmt(view['correct']['median'])} / wrong "
            f"{_fmt(view['wrong']['median'])}, p05 correct {_fmt(view['correct']['p05'])} / "
            f"wrong {_fmt(view['wrong']['p05'])} | {point('candidate', view['candidate'])} | "
            f"{point(f'budget <= {BUDGET_FLAGGED_RATE:.0%}', view['budget'])}"
        )
    print(f"-> {out}, {chart}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--ocr-dir", type=Path, help="default: outputs/ocr/<split>-gpu/")
    parser.add_argument(
        "--latency-dir", type=Path, help="CPU sample; default: outputs/ocr/<split>-cpu-latency/"
    )
    parser.add_argument("--out", type=Path, help="default: outputs/eval/confidence_<split>.json")
    parser.add_argument(
        "--chart", type=Path, help="default: outputs/eval/confidence_hist_<split>.png"
    )
    parser.add_argument("--iou", type=parse_iou_threshold, default=0.5, help="pair if IoU >= this")
    args = parser.parse_args()
    ocr_dir = args.ocr_dir or ROOT / "outputs" / "ocr" / f"{args.split}-gpu"
    latency_dir = args.latency_dir or ROOT / "outputs" / "ocr" / f"{args.split}-cpu-latency"
    out = args.out or ROOT / "outputs" / "eval" / f"confidence_{args.split}.json"
    chart = args.chart or ROOT / "outputs" / "eval" / f"confidence_hist_{args.split}.png"

    rows, revision = load_gt_rows(args.split, GT_COLUMNS, ocr_dir)
    try:
        result = evaluate(rows, ocr_dir, latency_dir, args.iou)
    except ValueError as exc:
        sys.exit(str(exc))
    images = result.pop("images_evaluated")
    if images == 0:
        sys.exit(f"no OCR JSON for any {args.split} key in --ocr-dir {ocr_dir}; no report written")

    report = {
        "split": args.split,
        "sroie_revision": revision,
        "iou_threshold": args.iou,
        "ocr_dir": str(ocr_dir),
        **result,
    }
    write_json_atomic(out, report)
    draw_chart(
        chart,
        report["confidence"],
        report["counts"],
        f"SROIE {args.split}, OCR {report['ocr_device']}, IoU >= {args.iou}",
    )
    _print_summary(report, images, out, chart)


if __name__ == "__main__":
    main()

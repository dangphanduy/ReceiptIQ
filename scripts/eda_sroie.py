"""EDA on the SROIE train split: dataset stats plus a fixed-seed sample for manual review.

Usage:
    python scripts/eda_sroie.py               # seed 42, 20 receipts
    python scripts/eda_sroie.py --seed 7 --n 30

Reads data/sroie (run download_sroie.py first). Uses the train split only, so the
test split stays clean for evaluation. Writes (git-ignored):
    outputs/eda/stats.json             stats over the whole train split + sampled keys
    outputs/eda/samples/<key>.jpg|json sampled images + ground-truth entities and words
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_from_disk
from download_sroie import DATA_DIR, ENTITY_FIELDS, ROOT

TOTAL_WORDS = re.compile(r"\b(SUB\s*TOTAL|TOTAL|ROUNDING|CASH|CHANGE)\b", re.IGNORECASE)


def shape(s: str) -> str:
    """Format shape of a field value: digits -> 9, letters -> A, everything else kept."""
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", s))


def summary(values: list[int]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def compute_stats(train) -> dict:
    sizes = train["image_size"]
    # Some GT fields are None (e.g. X51005663280 address); count them as empty.
    entities = [{f: e[f] or "" for f in ENTITY_FIELDS} for e in train["entities"]]
    words = train["words"]
    keys = train["key"]

    total_lines = [sum(bool(TOTAL_WORDS.search(w)) for w in ws) for ws in words]
    # GT value not found verbatim in any text region: GT typo, reformatted value, or split line.
    not_verbatim = {
        f: [
            k
            for k, e, ws in zip(keys, entities, words, strict=True)
            if not any(e[f] in w for w in ws)
        ]
        for f in ("date", "total")
    }
    return {
        "n_receipts": len(train),
        "width": summary([s["width"] for s in sizes]),
        "height": summary([s["height"] for s in sizes]),
        "lines_per_receipt": summary([len(ws) for ws in words]),
        "field_length_chars": {f: summary([len(e[f]) for e in entities]) for f in ENTITY_FIELDS},
        "empty_fields": {f: sum(not e[f].strip() for e in entities) for f in ENTITY_FIELDS},
        "date_shapes": Counter(shape(e["date"]) for e in entities).most_common(10),
        "total_shapes": Counter(shape(e["total"]) for e in entities).most_common(10),
        "company_all_upper_ratio": round(
            sum(e["company"] == e["company"].upper() for e in entities) / len(entities), 3
        ),
        "total_keyword_lines": summary(total_lines),
        "receipts_with_3plus_total_lines": sum(n >= 3 for n in total_lines),
        "not_verbatim_in_words": {
            f: {"count": len(v), "examples": v[:10]} for f, v in not_verbatim.items()
        },
    }


def export_sample(train, seed: int, n: int, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    for i in sorted(random.Random(seed).sample(range(len(train)), n)):
        row = train[i]
        key = re.sub(r"[^\w.-]", "_", row["key"])  # same filename rule as download_sroie.py
        row["image"].convert("RGB").save(out_dir / f"{key}.jpg", quality=95)
        record = {k: row[k] for k in ("key", "image_size", "entities", "words")}
        (out_dir / f"{key}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        keys.append(row["key"])
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    train = load_from_disk(str(DATA_DIR / "sroie"))["train"]
    out = ROOT / "outputs" / "eda"
    stats = compute_stats(train)
    keys = export_sample(train, args.seed, args.n, out / "samples")
    stats["sample"] = {"seed": args.seed, "keys": keys}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {out / 'stats.json'} and {len(keys)} samples to {out / 'samples'}")


if __name__ == "__main__":
    main()

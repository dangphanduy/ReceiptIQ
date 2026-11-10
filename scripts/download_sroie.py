"""Download ICDAR 2019 SROIE from Hugging Face at a pinned revision and verify it.

Usage:
    python scripts/download_sroie.py                      # pinned revision, 3 sample images
    python scripts/download_sroie.py --export-samples 10
    python scripts/download_sroie.py --revision main      # newer revision (count check skipped)

Writes (all under data/, which is git-ignored):
    data/hf_cache/                 raw Hugging Face download cache
    data/sroie/                    `datasets.save_to_disk` copy (load with `load_from_disk`)
    data/sroie/manifest.json       repo id, revision, commit sha, split sizes, license
    data/samples/<key>.jpg|.json   a few test images + ground-truth entities for the baseline
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

REPO_ID = "jsdnrs/ICDAR2019-SROIE"
# Commit checked on 2026-11-10. Bump deliberately and update docs/dataset.md.
PINNED_REVISION = "bffe40c26759f3376ec2b3ae9031dbba54cd587c"
EXPECTED_SPLIT_SIZES = {"train": 626, "test": 361}  # measured at PINNED_REVISION
REQUIRED_COLUMNS = {"image", "key", "image_size", "entities", "words", "bboxes"}
ENTITY_FIELDS = ("company", "date", "address", "total")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def verify(ds: DatasetDict, check_sizes: bool) -> None:
    """Raise if the download is missing splits, columns, or examples."""
    missing_splits = set(EXPECTED_SPLIT_SIZES) - set(ds)
    if missing_splits:
        raise RuntimeError(f"missing splits: {sorted(missing_splits)}")
    for name, split in ds.items():
        missing = REQUIRED_COLUMNS - set(split.column_names)
        if missing:
            raise RuntimeError(f"split {name!r} missing columns: {sorted(missing)}")
        if len(split) == 0:
            raise RuntimeError(f"split {name!r} is empty")
        if (
            check_sizes
            and name in EXPECTED_SPLIT_SIZES
            and len(split) != EXPECTED_SPLIT_SIZES[name]
        ):
            raise RuntimeError(
                f"split {name!r} has {len(split)} rows, expected {EXPECTED_SPLIT_SIZES[name]}"
            )
        first = split[0]
        if set(first["entities"]) < set(ENTITY_FIELDS):
            raise RuntimeError(f"split {name!r} entities lack {ENTITY_FIELDS}")
        if len(first["words"]) != len(first["bboxes"]):
            raise RuntimeError(f"split {name!r}: words/bboxes length mismatch in first row")
        first["image"].load()  # proves image bytes decode


def export_samples(ds: DatasetDict, n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in ds["test"].select(range(min(n, len(ds["test"])))):
        key = re.sub(r"[^\w.-]", "_", row["key"])  # key comes from remote data; keep it a filename
        row["image"].convert("RGB").save(out_dir / f"{key}.jpg", quality=95)
        (out_dir / f"{key}.json").write_text(
            json.dumps({"key": row["key"], "entities": row["entities"]}, indent=2),
            encoding="utf-8",
        )
    print(f"exported {min(n, len(ds['test']))} test samples to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--export-samples", type=int, default=3, metavar="N")
    args = parser.parse_args()

    commit_sha = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    print(f"{REPO_ID} @ {args.revision} -> commit {commit_sha}")

    ds = load_dataset(REPO_ID, revision=commit_sha, cache_dir=str(DATA_DIR / "hf_cache"))
    verify(ds, check_sizes=commit_sha == PINNED_REVISION)
    print("verified:", {name: len(split) for name, split in ds.items()})

    out = DATA_DIR / "sroie"
    ds.save_to_disk(str(out))
    manifest = {
        "repo_id": REPO_ID,
        "requested_revision": args.revision,
        "commit_sha": commit_sha,
        "splits": {name: len(split) for name, split in ds.items()},
        "license": "CC BY 4.0",
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "original_source": "https://rrc.cvc.uab.es/?ch=13",
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved to {out}")

    if args.export_samples > 0:
        export_samples(ds, args.export_samples, DATA_DIR / "samples")


if __name__ == "__main__":
    main()

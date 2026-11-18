# Dataset: ICDAR 2019 SROIE

## Source

| | |
|---|---|
| Used copy | Hugging Face [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) |
| Pinned revision | `bffe40c26759f3376ec2b3ae9031dbba54cd587c` (checked 2026-11-10) |
| Original source | ICDAR 2019 Robust Reading Challenge on SROIE, <https://rrc.cvc.uab.es/?ch=13> |
| Paper | Huang et al., *ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction*, ICDAR 2019 ([arXiv:2103.10213](https://arxiv.org/abs/2103.10213)) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The Hugging Face copy is an extended, corrected version of the original. Per its
dataset card it adds 14 Task 3 test annotations missing from the original release
and adds image metadata. Changes are listed at <https://github.com/jsdnrs/ICDAR2019-SROIE>.
The repo is pinned to that copy because it loads with `datasets` without manual
download steps. The official RRC page needs an account to download.

## Download

```bash
python scripts/download_sroie.py                    # pinned revision + 3 sample images
python scripts/download_sroie.py --export-samples 10
```

The script resolves the revision to a commit SHA, downloads through `datasets`,
checks splits, columns, row counts, entity fields and that an image decodes, then writes:

- `data/sroie/`: `load_from_disk("data/sroie")` copy, plus `manifest.json` (commit SHA, split sizes, timestamp)
- `data/samples/<key>.jpg` and `<key>.json`: first N **test** images with ground-truth entities
- `data/hf_cache/`: raw download cache

`data/` is git-ignored. Never commit dataset files or receipt images.

## Contents (at the pinned revision)

| Split | Rows |
|---|---|
| train | 626 |
| test | 361 |

Columns: `image` (PIL), `key`, `image_size {width, height}`,
`entities {company, date, address, total}` (strings), `words` (text regions,
usually one line each), and `bboxes` (`[x0, y0, x1, y1]` pixel coordinates, top-left/bottom-right).
The original annotations were 4-point quadrilaterals. The HF copy keeps only two corners.

Note: the card's split table says 347 test images, but the data at the pinned
revision has 361. That is 347 plus the 14 added Task 3 annotations, and
626 + 361 = 987 matches the card's summary. The download script checks the measured counts.

## Attribution (CC BY 4.0)

Anything that redistributes or shows derived material (screenshots, sample
outputs, a demo with SROIE images) must:

1. Credit the original authors: Zheng Huang, Kai Chen, Jianhua He, Xiang Bai,
   Dimosthenis Karatzas, Shijian Lu, C. V. Jawahar (ICDAR 2019 SROIE).
2. Link the license: <https://creativecommons.org/licenses/by/4.0/>.
3. State that changes were made, e.g. "images processed by ReceiptIQ OCR".
   Also mention that the HF copy itself modifies the original.
4. Not suggest that the authors endorse ReceiptIQ.

Suggested credit line:

> Receipt images and annotations: ICDAR 2019 SROIE (Huang et al., 2019), via
> Hugging Face `jsdnrs/ICDAR2019-SROIE`, licensed CC BY 4.0. Processed by ReceiptIQ.

BibTeX:

```bibtex
@inproceedings{Huang_2019,
  author    = {Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and
               Karatzas, Dimosthenis and Lu, Shijian and Jawahar, C. V.},
  title     = {ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
  booktitle = {2019 International Conference on Document Analysis and Recognition (ICDAR)},
  publisher = {IEEE},
  year      = {2019}
}
```

## Limitations

- **Small**: about 1k images. That is enough for evaluating a baseline, but small for training.
- **Narrow domain**: English receipts, mostly from Malaysia, 2016–2018. Layouts,
  `RM` currency and tax/rounding conventions are regional. Results do not
  transfer to other countries, languages (including Vietnamese) or invoice types.
- **Scans, not phone photos**: mostly flat and upright scans. Real uploads
  (skewed, shadowed, crumpled photos) will perform worse.
- **Privacy**: some personal fields are blurred, but company names and addresses remain.
- **Annotation quirks**: field strings follow SROIE conventions (e.g. `total` usually
  without currency, though ~15% keep `$` or `RM`; multi-line addresses joined). Evaluation must normalize
  consistently, see [baseline.md](baseline.md). Hard cases and GT errors found on the
  train split are listed in [eda.md](eda.md).
- **Unknown split criterion**: the card does not say how train/test were split.

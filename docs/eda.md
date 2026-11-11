# EDA: SROIE train split (data card)

Exploratory look at the SROIE **train** split to find what makes the four fields
(`company`, `date`, `address`, `total`) hard to extract. The test split is not used here,
so it stays clean for evaluation. Dataset details and the CC BY 4.0 attribution are in
[dataset.md](dataset.md).

## Reproduce

```bash
python scripts/eda_sroie.py            # seed 42, 20 receipts; needs data/sroie from download_sroie.py
```

The script writes `outputs/eda/stats.json` (all numbers below) and
`outputs/eda/samples/<key>.jpg|.json` (the 20 sampled receipts with GT entities and `words`).
`outputs/` is git-ignored. The same seed gives byte-identical output.
Hard cases were found by viewing the 20 sampled images against their GT. Cases outside
the sample come from the script's automatic signals and are marked *(signal)*.

Sample (seed 42): `X51005361900 X51005361946 X51005433492 X51005433514 X51005577192
X51005605333 X51005663280 X51005676538 X51005711447 X51006008091 X51006311767 X51006329399
X51006350763 X51006414392 X51006913031 X51007419197 X51008099044 X51008114260 X51008142060
X51008164525`

## Stats (train, 626 receipts)

| | min | median | max |
|---|---|---|---|
| image width (px) | 436 | 812 | 4961 |
| image height (px) | 605 | 1682.5 | 7016 |
| text regions (`words`) per receipt | 18 | 50 | 153 |
| lines matching `SUBTOTAL\|TOTAL\|ROUNDING\|CASH\|CHANGE` | 0 | 5 | 10 |
| `company` length (chars) | 7 | 24 | 49 |
| `date` length | 8 | 10 | 12 |
| `address` length | 0 | 72 | 135 |
| `total` length | 0 | 5 | 8 |

- 595 / 626 receipts (95%) have ≥ 3 total-like lines.
- Empty GT fields: `address` 1 (`X51005663280`, stored as `None`), `total` 1 (`X51005433522`).
- GT value not found verbatim in any `words` region: `date` 5, `total` 2 (see hard case 10).
- `company` GT is all upper case in 100% of receipts, even when the receipt prints mixed case.

### Field formats (shape: digit → `9`, letter → `A`)

| `date` shape | n | | `total` shape | n |
|---|---|---|---|---|
| `99/99/9999` | 330 | | `99.99` | 325 |
| `99-99-99` | 82 | | `9.99` | 125 |
| `99-99-9999` | 70 | | `999.99` | 84 |
| `99/99/99` | 61 | | `$9.99` | 28 |
| `99 AAA 9999` | 47 | | `AA9.99` (`RM`) | 17 |
| `9/9/9999` | 6 | | `AA99.99` | 17 |
| `99 AAA 99` | 5 | | `AA 99.99` | 14 |
| `99-AAA-9999` | 5 | | `AA 9.99` | 4 |
| `99/AAA/9999` | 5 | | `$99.99` | 4 |
| `9999-99-99` | 4 | | `AA999.99` | 3 |

Dates keep the receipt's own format (day-first, 2- or 4-digit year, month names).
Totals are usually bare numbers, but about 15% keep a `$` or `RM` prefix, sometimes with a space.

## Hard cases

| # | Type | What happens | Example keys |
|---|---|---|---|
| 1 | Many total-like lines | SUB-TOTAL, TOTAL TAX, ROUNDING, TOTAL, CASH, CHANGE, and a GST-summary `TOTAL:` *below* the real total showing the pre-tax amount (16.98 vs GT 18.00) | `X51005361900`, `X51006350763` |
| 2 | Rounding | GT `total` is the rounded amount, while the line labelled `Total` holds the unrounded one | `X51005663280` (102.39 → 102.40), `X51005605333` (31.47 → 31.45), `X51006414392` |
| 3 | Varied total labels | `TOTAL`, `Total Incl. GST@6%`, `Nett Total`, `GRAND TOTAL`, `Total Amt Rounded`, `Total Sales (Inclusive of GST)`, `Total RM` | `X51005361946`, `X51005433492`, `X51007419197`, `X51006350763`, `X51008142060`, `X51005605333` |
| 4 | Stamps and handwriting over text | A red `PAID` stamp over the totals, or over the company line; handwritten numbers and dates (`A03018`, `30/12/17`, `102.40`) | `X51005361900`, `X51005361946`, `X51005433492`, `X51005663280` |
| 5 | Scan noise, faded print | Halftone dot pattern over the whole page; faint dot-matrix glyphs | `X51005361946`, `X51006350763` |
| 6 | Coloured ink / dot-matrix | Header and TOTAL/CASH/CHANGE printed in pink dot-matrix | `X51005577192`, `X51005605333` |
| 7 | Scale and margins | Small receipt in the middle of an A4 scan at 4945×6981, vs a tight 612×984 crop | `X51006414392`, `X51006913031`, `X51007419197` |
| 8 | Company not on the first line | Brand name first and legal entity below (`Chili's` → GT `T.A.S LEISURE SDN BHD`); company line after the address; a second company in the "Bill To" block | `X51005663280`, `X51007419197`, `X51008142060` |
| 9 | Several dates / formats | Paper date vs stamp date; one receipt prints both `2018-04-06` and `06/04/18`; month names | `X51005361946`, `X51006414519` *(signal, image checked)*, `X51005663280` |
| 10 | GT errors and inconsistencies | Company typo `TAII ORING` (receipt: `TAILORING`); GT date `20180304` in a format the receipt never prints; empty `total` while the receipt shows `Nett Total: $8.20`; `None` address; total `1,007.50` with a thousands comma | `X51006350763`, `X51005447850` *(signal, image checked)*, `X51005433522` *(signal, image checked)*, `X51005663280`, `X51006556830` *(signal, image checked)* |
| 11 | Annotation errors in `words` | `words` reads `28-11-18` where the image and GT both say `28-01-18` | `X51008142038` *(signal, image checked)* |
| 12 | GT normalisation drift | Same store, two receipts, different address punctuation (`TAMPOI,81200` vs `TAMPOI 81200`); currency prefix kept in some totals (`$8.20`, `RM33.90`) but dropped in others (`RM 37.10` on paper → `37.10`) | `X51005433492` / `X51005433514`, `X51006733494` *(signal)*, `X51005361946` |

## Hypotheses for extraction rules (E3)

1. **Total**: take the amount from the *last* line labelled TOTAL / GRAND TOTAL / NETT TOTAL /
   Total Incl. / Total Amt Rounded that sits *above* CASH/CHANGE. Skip lines with QTY, TAX/GST,
   SUB, EXCL, and anything under a "GST Summary" header. Prefer the rounded value when a ROUNDING line exists.
2. **Total check**: if CASH − CHANGE equals a candidate, that candidate is very likely the total.
3. **Total normalisation**: compare totals after removing `RM`, `$`, spaces and thousands commas.
   GT is inconsistent here (hard case 12), so evaluation must normalise both sides the same way.
4. **Date**: regex over day-first patterns (`dd/mm/yyyy`, `dd-mm-yy`, `dd MMM yyyy`, `yyyy-mm-dd`).
   Prefer a match next to `Date`, or the first printed date. Ignore stamp dates. Evaluate on the parsed date, not the raw string.
5. **Company**: look in the top ~5 lines for a line ending in `SDN BHD`, `S/B`, `SB`, `ENTERPRISE`,
   `RESTAURANT`, `TRADING`. Otherwise use the first upper-case line that is not an address, phone or GST ID.
   Skip anything after `Bill To`. Upper-case the output.
6. **Address**: join the lines between the company and the first `TEL` / `GST` / `Company Reg` line, up to
   the line holding the 5-digit postcode and state. Upper-case and normalise commas and spaces before comparing.
7. **Upstream**: OCR quality matters most for faded dot-matrix and stamped regions. Before tuning any
   rule, check the OCR baseline on hard cases 4–6.

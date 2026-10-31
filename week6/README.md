# Week 6 — Class Task: the integration checkpoint

One unchecked dataset for the National Grid Authority, made trustworthy and turned into one figure for the board: a three-file audit with a 17-row decisions table, three charts answering three kinds of question, a normalized reliability figure with its counterfactual, and a real-data bonus on Pakistan's solar electricity.

**CLO1 + CLO2 — everything from Weeks 1-5 at once.**

← [Back to the course index](../README.md)

## Run it

```bash
cd week6
python selfcheck.py                    # course self-check: toolkits + data load
jupyter notebook                       # Run All in partA, then partB, then partC
```

**Run order matters:** Task A writes `submission/clean/`, and Tasks B and C read from it.

Or run the hand-in scripts directly. They work from any folder:

```bash
python submission/partA_process.py     # audit -> submission/clean/, partA_audit.txt, partA_decisions.md
python submission/partB_charts.py      # partB_1.png, partB_2.png, partB_3.png
python submission/partC_figure.py      # partC_figure.png
```

Requires `pandas`, `numpy`, `matplotlib`, `jupyter`, `ipykernel` (versions in `submission/requirements.txt`). Nothing new this week: `vizlib.py`, `tslib.py` and `geolib.py` are the Week 3-5 toolkits, unchanged. The raw files in `data/` are never written to; Task A re-reads them at the end and checks they're unchanged.

## Contents

| Path | What it is |
|---|---|
| `TASKS.md` | The brief and marking guide |
| `vizlib.py` · `tslib.py` · `geolib.py` · `selfcheck.py` | Course toolkits and self-check, unchanged |
| `data/` | `district_grid.csv`, `grid_daily.csv`, `generation_mix.csv`, `regions.geojson`, `DATA_DICTIONARY.md` (raw, untouched) |
| `partA_process.ipynb` | Task A: the audit of all four files, the fixes, the decisions table, the clean files |
| `partB_charts.ipynb` | Task B: trend (daily energy), composition (generation mix), relationship (solar vs income) |
| `partC_figure.ipynb` | Task C: the board figure, suppression and funnel check, class breaks, the counterfactual |
| `partC_note.md` | Task C: the half-page note |
| `charts/` | Every chart (300 dpi PNG), including `partC_counterfactual.png` |
| `submission/` | The hand-in, exactly the list in TASKS.md |
| `partD_bonus/` | **Bonus, kept separate:** Task D2 on real Our World in Data electricity data for Pakistan |

### `submission/` — the hand-in

| File | Task |
|---|---|
| `README.md` · `requirements.txt` | Setup: what to run, in what order, and the library versions |
| `partA_process.py` · `partA_audit.txt` · `partA_decisions.md` · `clean/` | A: script, its printed audit, the decisions table, six cleaned CSVs written by the script |
| `partB_charts.py` · `partB_1.png` · `partB_2.png` · `partB_3.png` | B: script and the three charts |
| `partC_figure.py` · `partC_figure.png` · `partC_note.md` | C: script, the 300 dpi figure, the note |

## Self-checks

Every number in the brief is an `assert`, so a notebook stops if the cleaning is off.

| Check | Target | Mine |
|---|---|---|
| Register rows / population / households | 24 / 57,857,000 / 8,908,512 | yes |
| Persons per household | 5.70 – 7.36 | yes (D21 repaired 182 → 182,000) |
| `electrification_pct` | 66.6 – 90.3, 3 missing | yes |
| Usable `outage_minutes` | 22, mean 77 – 214 min | yes |
| Interruptions | 388,879 | yes (kept `Kotla`, dropped `Kotla City`) |
| Zone-days kept / span / never reported | 3,046 / 529 × 6 = 3,174 / 128 | yes |
| Unusable `peak_load_mw` | 46 | yes (46 codes; I also blanked 3 conflicting double filings) |
| Meter reading falls | 1 | yes (P4, 17 Feb 2026, a meter swap) |
| P1 June 2026, differenced vs summed | 1,200,239 vs 574,845,315 (479×) | yes |
| Part rows / parts vs total / all rows | 144 / within 2 MWh / 200.0% | yes |
| GeoJSON join | 24 / 0 | yes |
| Generation / solar | −5.0% / +82.1%, 4.1% → 7.9% | yes |
| r(count, households) / r(rate, households) | +0.991 / +0.175 | yes |
| Top 3 by count / by rate | D17, D13, D24 / D12, D11, D13 | yes, D13 on both |
| National pooled rate | 43.7 per 1,000 households | yes |
| 5 most populous | 12.6% of land, 59.9% of people | yes |

**Energy convention.** The brief's P1 June figure is `reading on 1 July − reading on 1 June`, so it treats a reading as the meter at the start of its day. I use that everywhere. With the dictionary's "end of day" wording P1 June would be 1,199,207 MWh (0.09% lower); no chart changes.

## Chart analysis

**Task A — the eleven defects.** Register: D11 keyed twice, `electrification_pct` as text (`88.1%`) with 3 blanks, `outage_minutes = -1` on D09 and D20, and D21's households in thousands (5,940 people per household). Daily file: two date formats (every November 2025 row is DD/MM/YYYY), 15 double filings (12 identical, 3 with the same meter reading but different peak and minutes), 128 unreported zone-days (P5 36, P1 12), 46 `-1` peaks, one meter swap, and a cumulative counter that is 479× wrong if summed; plus the partial September 2026. Generation file: the ministry's Total row inside the parts, so every row sums to 200%. Each one has a rule and a count in `partA_decisions.md`; nothing missing became a zero.

**Task B — three charts.**

1. **Trend:** every zone delivered 5.9-7.5% more energy per day in summer 2026 than summer 2025. Plan summer 2027 on that growth. A `sum()` over the gappy summers would have shown P1 and P2 falling.
2. **Composition:** solar +82% (458 → 833 GWh a month) while total generation fell 5%. Same month a year apart, solar is up 35-62%, so the rise is real; only about 0.2 of the 3.8 share points comes from the shrinking total.
3. **Relationship:** rooftop solar per household follows income (r = +0.93), from 2.1-2.3 kW in the two richest districts to 0.19-0.48 in the two poorest. The solar line shouldn't be sold as help for poor districts.

**Task C — the board figure.** Map (Jenks, 5 classes, equal-area) beside a ranked bar chart, both in interruptions per 1,000 billed households. Lashkar (D12), Kotla (D11) and Mehrabad (D13) are worst, about 50 per 1,000 against 43.7 nationally. The count version would point at Qilabad (19th per household) and Zhobara (12th), both below the national rate, and take money away from Lashkar and Kotla. Mehrabad is on both lists because it is poor (joint-lowest income, r(rate, income) = −0.71), not because it is big. Nothing suppressed; 8 districts are above the 95% funnel, 14 below, 2 inside.

**Bonus (Task D2, separate folder).** Pakistan's solar electricity grew 14-fold from 2019 to 2025 (2.6 → 36.6 TWh, 1.9% → 18.8%) while total generation grew 44%. Named defect: `other_renewable_electricity` already includes `biofuel_electricity`, so drawing both double-counts up to 2.15 TWh a year.

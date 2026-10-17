# Week 4 — Time-Series, Proportions & Hierarchies

Nine charts that go wrong before anything is drawn, three smoothing decisions computed, a part-to-whole legality check with a pie-ranking experiment, and the 2×2 e-commerce review figure.

**L7 — design time-series charts that expose trend, seasonality and cycles. · L8 — compare part-to-whole strategies and build hierarchical charts. (CLO2)**

← [Back to the course index](../README.md)

## Run it

```bash
cd week4
python tslib.py                        # self-check that the toolkit works
jupyter notebook                       # open any partX notebook and Run All
```

Run Part D **before** Part B: Part D writes the cleaned daily series (`data/clean/sessions_daily_clean.csv`) that Part B works on.

Or run the hand-in scripts directly. They work from any folder:

```bash
python submission/partA_charts.py
python submission/partD_figure.py
python submission/partB_windows.py
python submission/partC_shares.py
python submission/partC_shares.py --run "Reader name"    # time one real reader on the pie and the bars
```

Requires `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `squarify`, `jupyter`, `ipykernel`. All Part A-E data is already in `data/`.

## Contents

| Path | What it is |
|---|---|
| `TASKS.md` | The brief and marking guide |
| `vizlib.py` | Week 3's toolkit, unchanged |
| `tslib.py` | Week 4's toolkit: `audit_index`, `window_report`, `lag_at_turn`, `banked_figsize`, `exclusivity_check`, `rollup_other`, `treemap`, `cycle_plot` |
| `data/` | Twelve synthetic datasets + `DATA_DICTIONARY.md`; `data/clean/` is written by Part D |
| `partA_nine_charts.ipynb` | Part A: nine rebuilds with asserted self-checks, diagnoses, redesigns |
| `partB_windows.ipynb` | Part B: window report, three smoothers vs Black Friday, decomposition, lag, 60 window framings |
| `partC_shares.ipynb` | Part C: three legality tests, the two stimuli, the timing harness and scoring |
| `partD_ecommerce_figure.ipynb` | Part D: inspection, every fault with a rule and a count, the 2×2 figure, a 28-day appendix |
| `partE_justification.md` | Part E: the one-page written justification |
| `charts/` | Every chart (300 dpi PNG) |
| `submission/` | The hand-in files from TASKS.md |
| `partF_bonus/` | **Bonus, kept separate:** Week 1's open question settled on real PBS data |

### `submission/` — the hand-in

| File | Part |
|---|---|
| `partA_charts.py` · `partA_diagnosis.pdf` | A: rebuilds all nine and plots the redesigns; one page per case with both figures |
| `partB_windows.py` · `partB_output.txt` | B: script + its printed output |
| `partC_shares.py` · `partC_stimulus_pie.png` · `partC_stimulus_bars.png` | C: script, both stimuli; `pie_ranking_results.csv` appears once a reader has been run |
| `partD_figure.py` · `partD_figure.png` · `partD_cleaning_log.csv` | D: script, 300 dpi figure, cleaning log |
| `partE_justification.md` | E: the one-page justification |

## Self-checks

Every Part A self-check is an `assert`, so the notebook stops if a rebuild is not faithful.

| Case | Target | Mine |
|---|---|---|
| 1 | 24 points, drops of 958 and 978 | yes |
| 2 | 69 points, 207 → 253, +22% | yes |
| 3 | June ~8,800, July ~12,500, +43% | 8,782 / 12,543, +42.8% |
| 4 | trailing mean bottoms on 2025-09-07 | yes |
| 5 | 7 points, 158.8 → 92.2, −41.9% | yes |
| 6 | Sep 10,805, Oct 4,041 | yes |
| 7 | envelope ~548 to ~709 | 548.3 to 708.7 |
| 8 | 42.6 / 32.3 / 27.5 / 28.6 | yes |
| 9 | 24.0% → 38.0% | yes |
| D | eight shares within 0.3 pts of `shares_eight.csv` | max 0.23 pts |

## Chart analysis

**Part A — nine charts.** None of the numbers are fiddled. Each chart goes wrong upstream of the pixels:

1. **Orders:** `YYYY-M` text sorted alphabetically puts Oct-Dec before February. In real order all 23 monthly changes are positive (+52.5% overall).
2. **Shop:** 69 rows over 90 days. The 21-day closure (3-23 Feb) vanishes on a row-number axis; both sides are flat and the "+22%" is one step at reopening.
3. **Bandwidth:** a level summed to daily. June logged only 16.4 hours a day against 24 in July, which explains the whole +43%. The daily mean actually fell 2.5%.
4. **Checkout:** a 31-day trailing mean lags 15 days. The real turn is 23-28 August, not 7 September.
5. **Revenue:** May is the seasonal peak and November the trough; May→Nov falls ~42% every year, and 2025 is up 9.9% on 2024.
6. **October:** an incomplete month, 11 of 31 days. Orders per day were a record.
7. **Product D:** the steadiest line (SD 3.6). It inherits product B's volatility (SD 35.6) through the stack's moving baseline.
8. **Search:** non-exclusive parts summing to 131%. 27.6% of customers touched 2+ channels; search alone is 25.9%.
9. **Premium:** a rising share of a falling total. Premium revenue fell 5% while total revenue fell 40%.

**Part B — windows.** I use a 7-day window because the cycle being cancelled is weekly. On Black Friday the 14-day mean reads 33,468, the median 24,780 and the EWMA 38,029. The weekly cycle swings 7,584 sessions/day and the largest residual is Black Friday. `period=365` can't run on 529 days. A 31-day trailing mean lags 15 days. Across 60 start × window framings, 34 show a rising trend and 26 a falling one.

**Part C — shares.** Case 8 can't be drawn as any share chart, `shares_eight` is a legal pie but better as bars, and premium-vs-other 2025Q4 is a legal two-slice pie that answers the wrong question. The stimuli and harness are ready. *The reader run still has to be done* — no results are invented.

**Part D — the figure.** Seven data problems handled, each with a rule and a count (two timestamp formats, unsorted rows, device casing, 287 `-1` sentinels, 286 duplicates, 13 unrecorded days, a partial final month), plus a duplicated SKU and 7 negative-net SKUs in the catalogue. Traffic grew through 2024, hit a spring low of ~22.6k/day in May 2025 and is up 37% since. Weekend evenings are the peak, and Phones lead the catalogue at 424m PKR.

**Part E — justification.** One page: each panel's question, chart type and channel rule; every time decision; every row removed with before → after; the roll-ups; and the window clause.

**Bonus (Part F, separate folder).** On real PBS diabetes-drug data (1991-2008), the seasonal swing grew ×3.9, but only because the business did. The multiplicative residual SD stays flat (×0.97) while the additive one grows ×1.55.

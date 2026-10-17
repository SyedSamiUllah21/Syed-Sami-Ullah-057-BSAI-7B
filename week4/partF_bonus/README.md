# Week 4 — Part F (bonus): is the seasonality growing, or is the business?

Kept separate from Parts A-E. Marked out of 5 and not added into the 100.

Week 1 Task 5 (`flights.csv`) asked: the seasonal swing gets bigger every year, so is the seasonality itself growing or is the business growing? Here it's settled on real data.

| File | What it is |
|---|---|
| `partF_cycle.ipynb` | The notebook: audit, three panels, the answer |
| `partF_cycle.py` | Same code as a script (`python partF_cycle.py`) |
| `partF_cycle.png` | The 300 dpi figure: banked series, cycle plot, additive vs multiplicative residuals |
| `get_data.py` | Re-creates the CSV from the original file (needs `pyreadr`) |
| `data/PBS.rda` | The original download |
| `data/pbs_alimentary_monthly.csv` | 13 drug classes (ATC2, the category column) × 204 months |

**Source.** Pharmaceutical Benefits Scheme (PBS) monthly scripts and cost, Medicare Australia, July 1991 - June 2008, as published in Rob Hyndman's `tsibbledata` R package: <https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data/PBS.rda>. Downloaded 17 October 2026.

**Answer.** For A10 (diabetes drugs) the within-year swing grew from 3.0m to 11.6m AUD (×3.9), but only because the business grew about 6×. As a share of each year's level the swing never trends up, and the multiplicative residual SD stays flat (4.9% → 4.8%, ×0.97) while the additive one grows ×1.55. So the seasonality is multiplicative, the same answer the textbook gives for `flights.csv`.

Needs `tslib.py` / `vizlib.py` from the `week4/` folder above (the code adds it to the path).

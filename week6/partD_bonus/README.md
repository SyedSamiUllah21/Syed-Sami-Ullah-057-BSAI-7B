# Week 6 - Task D2 bonus: the same pipeline on real data

Is Pakistan's shift to solar electricity real, or arithmetic? Audit, clean, decisions table and one figure, on a real file.

**Source.** Our World in Data, energy dataset (`owid-energy-data.csv`, built on Ember yearly electricity data and the Energy Institute Statistical Review), https://github.com/owid/energy-data. **Downloaded 31 October 2026.** The SHA-256 of the file is in `data/SOURCE.txt`.

## Run it

```bash
cd week6/partD_bonus
python get_data.py           # re-downloads and writes data/owid_energy_extract.csv (all entities, 1985+, untouched)
python partD_real_data.py    # audit -> clean/ -> partD_decisions.md -> partD_figure.png
```

Or open `partD_real_data.ipynb` and Run All.

| File | What it is |
|---|---|
| `get_data.py` | Downloads the file and saves the extract; no cleaning |
| `data/owid_energy_extract.csv` · `data/SOURCE.txt` | The raw extract and where it came from |
| `partD_real_data.py` / `.ipynb` | Audit, cleaning, decisions table, figure |
| `partD_decisions.md` | Four decisions: what, how detected, rule, rows affected |
| `clean/pakistan_electricity_2000_on.csv` | Written by the script |
| `partD_figure.png` | The figure (300 dpi) |

## The named defect

**`other_renewable_electricity` already contains `biofuel_electricity`.** A stack that draws both as separate parts double-counts bioenergy, by up to 2.15 TWh a year for Pakistan (2015). The 8 exclusive parts (coal, gas, oil, nuclear, hydro, solar, wind, other renewables) match the published total exactly in every year from 2000. The same file also mixes 87 aggregates into the `country` column (continents, income groups, the same region from several sources), so summing every row for 2023 gives 8.7 times the World total. And before 2000 Pakistan's fossil split is missing while the total is not, so the window starts in 2000.

## What the figure shows

Solar went from 2.6 TWh in 2019 (1.9% of generation) to 36.6 TWh in 2025 (18.8%), 14-fold, while total generation grew 44%. Because the total grew too, the rise in share is all real output: with solar flat at its 2019 level the share would have fallen to 1.3%. This is the opposite of the synthetic grid, where the total fell and a small part of the share gain was arithmetic.

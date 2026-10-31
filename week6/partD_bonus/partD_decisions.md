# Task D2 - decisions table (real data)

Written by `partD_real_data.py`.

| # | What | How I detected it | Rule applied | Rows affected |
|---|---|---|---|---|
| 1 | `country` mixes countries with 87 aggregate rows (continents, income groups, and the same region from several sources: 'Africa (EI)', 'Africa (Ember)'...) | Entities with no ISO code; every row summed for 2023 = 8.7x the World row | A country is a row with an ISO-3 code. The 7 former / partly recognised countries are named, not counted as aggregates. Nothing is ever summed across all rows | 3,175 aggregate rows excluded from any country total |
| 2 | Columns that look like parts are not exclusive: `biofuel` sits inside `other_renewable`, and `fossil`, `renewables`, `low_carbon` are totals of other columns | Exclusivity test on Pakistan: 8 parts match the total exactly; adding `biofuel` overshoots by up to 2.15 TWh; all 13 columns summed = 2.4x | Use exactly 8 exclusive parts: coal, gas, oil, nuclear, hydro, solar, wind, other renewables (incl. bioenergy). Shares are over their sum, checked against the published total | 5 columns not used as parts |
| 3 | Pakistan 1985-1999: total present, coal / gas / oil missing (NaN) | `isna()` per column; parts with NaN as 0 fall 32 TWh a year short of the total on average | Missing is not zero. The window starts in 2000, the first year every part is reported; fixed before drawing | 15 years left out of the chart |
| 4 | 2025 is the newest year in the file | Last year in the series; the most recent years of a release can be revised later | Kept and drawn like the other years, but labelled as the latest release in the caption | 1 year flagged in the caption |

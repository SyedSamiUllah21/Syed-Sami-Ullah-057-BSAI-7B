# Week 4 - Part E: Written justification

**Syed Sami Ullah · BSAI-7B · CDB601220 Data Visualization** · figure: `submission/partD_figure.png` · all data is synthetic course data.

## 1. The four panels

| Panel | Question it answers | Chart type, and why | Week 2 channel rule |
|---|---|---|---|
| 1 | Where is traffic heading once the weekend rhythm is removed? | Line over a real calendar: raw daily total in grey, 7-day centred mean on top | Position on a common scale. One pop-out colour (blue) for the signal, grey for context |
| 2 | When in the week do people visit? | Day × hour heatmap of mean sessions per hour, sequential `Blues` from 0 | Colour value for magnitude, with no midpoint so not diverging. Only used where position is already spent on two categorical axes |
| 3 | Where is the revenue mass, and how does it nest? | Treemap category → sub-category, one hue per parent, values printed | Area is 4th in the ranking (read to about ±20%), so every block carries its number |
| 4 | By *how much* do the sub-categories differ? | Sorted horizontal bars from zero, values printed | Length on a common baseline, 2nd in the ranking. This is the panel for "Phones are 10× Drones" |

Panels 3 and 4 show the same eight numbers on purpose: the treemap answers *where is the mass* (and the parent structure), the bars answer *by how much*.

## 2. Time decisions

- **Parse.** `ts` holds two formats: 9,019 ISO rows and 3,651 `DD/MM/YYYY HH:MM`. I parsed each with an explicit `format=`. The slash rows are day-first, because their first field goes above 12 in 2,151 rows and the second field never does. `format="mixed"` would have parsed every row with no error and put 1,382 rows on the wrong date (span Jan 2024 to Dec 2025). The correct span, 1 Apr 2024 to 11 Sep 2025, matches the ISO rows.
- **Resolution and aggregation.** Hourly → daily. `sessions` is a flow, so it is `sum()`, with `min_count=24` so a day exists only if all 24 hours do. The heatmap stays hourly (`mean`).
- **Window.** 7 days, because the cycle being removed is the weekly one (weekends run +35%), and a 7-day window holds each weekday exactly once (Part B.1). It is **centred** because this is a review of past weeks. A live dashboard would have to use trailing and say it lags 15 days at 31 (Part B.4). `min_periods=7`, so the trend breaks around each outage day.
- **Aspect ratio.** Panel 1 is drawn at h/w 0.60. `banked_figsize` on the trend suggests 0.75; the figure is a 2×2 grid, so I accepted the slightly flatter panel and say so.

**The window clause.** The window was fixed at 7 days before plotting, because the cycle being removed is weekly (Part B.1). The series is shown from its first recorded day; no start date was selected. A 28-day version is in the Part D appendix and does not change the direction (spring low on 25 May vs 14 May, rising since). Part B.5 shows the same series gives 34 rising and 26 falling framings, which is why none of these choices was made after looking. Symmetry test: if the 7-day line had hidden the June upturn, I would still have used it, because the reason for 7 is the weekly cycle and not the result.

## 3. Every row removed or changed

| Rule | Count | Moved from → to |
|---|---|---|
| Exact duplicate timestamps, keep first | 286 rows | 12,670 → 12,384 rows; mean 970.2 → 969.4 sessions/h |
| `sessions = -1` sentinel → NaN, filled from neighbouring hours (3 edge hours from same hour ±7 days), flagged | 287 hours on 222 days | mean 946.9 (with −1 counted) → 969.0 sessions/h |
| Unrecorded hours left as NaN, **not** filled | 312 hours = 13 whole days | 516 plotted days of 529, line breaks |
| `device` re-cased | 1,447 rows | 6 labels → 3 |
| Partial September 2025 | 11 days | marked on panel 1, never totalled as a month |
| SKU-0127 filed under Wearables as well as Audio; Wearables copy dropped | 1 row | Wearables 10.13% → 9.35% of net (self-check now within 0.23 pts) |
| Negative net revenue (returns > sales) kept out of area | 7 SKUs, −1.07 PKR m | net 1,711.38 → 1,712.45 PKR m drawn; the −1.07 is stated in the note |

Black Friday (113,710 sessions on 29 Nov 2024) is real, so it stays in and is annotated, not removed.

## 4. Roll-ups

Depth is cut at **sub-category**: 337 SKUs → 8 sub-categories → 4 categories, and children sum to parents (checked with an `assert`). `rollup_other(min_share_pct=1)` put **nothing** in `Other`, because the smallest sub-category, Drones, is 2.4% (41.3 PKR m), so there is no hidden bucket to declare. The top level has 4 categories, inside the 4 ± 1 limit. Device classes are summed together (panel 1 answers a whole-site question), and the heatmap rolls 12,097 recorded hours into 168 day × hour means.

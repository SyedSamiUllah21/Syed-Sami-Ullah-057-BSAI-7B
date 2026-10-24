# Week 5 — Multivariate Data & Geospatial Visualization

Nine charts broken by the encoding or the geography, five channel and spatial decisions computed, the two-map reader activity, and the three health figures: a four-variable bubble chart, a parallel coordinates plot and a normalized choropleth.

**L9 — map high-dimensional data with multivariate encodings; implement parallel coordinates. · L10 — build thematic maps without population bias, boundary distortion or bad colour scaling. (CLO2)**

← [Back to the course index](../README.md)

## Run it

```bash
cd week5
python geolib.py                       # toolkit self-check (see the note below on pandas 3)
jupyter notebook                       # open any partX notebook and Run All
```

The parts don't depend on each other, so any order works. Part D1 uses the size range chosen in Part B1, and the value is written into the script.

Or run the hand-in scripts directly. They work from any folder:

```bash
python submission/partA_charts.py
python submission/partB_decisions.py > submission/partB_output.txt
python submission/partC_maps.py
python submission/partC_maps.py --run "Reader name"                                        # one real reader, on screen
python submission/partC_maps.py --record "Reader name" --a "x, y, z" --b "x, y, z"         # one real reader, answers taken on paper
python submission/partD_bubble.py
python submission/partD_pcp.py
python submission/partD_choropleth.py
```

Requires `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `jupyter`, `ipykernel`. No geopandas: all maps, projections and class breaks come from `geolib.py`. All Part A-E data is already in `data/`.

**Toolkit note.** On pandas 3, `python geolib.py` stops at `corr_chain()`, because `np.fill_diagonal(df.corr().abs().values, 0)` writes into an array pandas now hands back read-only. My notebooks use the same greedy rule with a copied array (and fall back to `geolib.corr_chain` when it works), and they assert the brief's chain. `geolib.py` itself is unchanged.

## Contents

| Path | What it is |
|---|---|
| `TASKS.md` | The brief and marking guide |
| `vizlib.py` | Week 3's toolkit, unchanged |
| `geolib.py` | Week 5's toolkit, unchanged: `read_geojson`, `polygon_area_km2`, `equal_area_xy`, `breaks`, `gvf`, `choropleth`, `prop_symbols`, `size_legend`, `scale_frame`, `corr_chain`, `pcp`, `funnel_limits` |
| `data/` | Seven tables, three GeoJSON boundary sets and `DATA_DICTIONARY.md` |
| `partA_nine_charts.ipynb` | Part A: nine rebuilds with asserted self-checks, diagnoses, redesigns |
| `partB_decisions.ipynb` | Part B: size range, overplotting four ways, scaling + PCA, project before you measure, the funnel |
| `partC_maps.ipynb` | Part C: the two maps, the reader harness and the scoring cell |
| `partD_bubble.ipynb` · `partD_pcp.ipynb` · `partD_choropleth.ipynb` | Part D: the three health figures |
| `partE_justification.md` | Part E: the one-page written justification |
| `charts/` | Every chart (300 dpi PNG) |
| `submission/` | The hand-in files from TASKS.md |
| `partF_bonus/` | **Bonus, kept separate:** COVID-19 deaths per 100,000 by country, on real WHO / UN / Natural Earth data |

### `submission/` — the hand-in

| File | Part |
|---|---|
| `partA_charts.py` · `partA_diagnosis.pdf` | A: rebuilds all nine and plots the redesigns; one page per case with both figures |
| `partB_decisions.py` · `partB_output.txt` | B: script + its printed output |
| `partC_maps.py` · `partC_map_A.png` · `partC_map_B.png` · `partC_reader_results.csv` | C: script, the two stimuli, the results file (**header only until real readers are run**) |
| `partD_bubble.py` + `.png` · `partD_pcp.py` + `.png` · `partD_choropleth.py` + `.png` | D: three scripts, three 300 dpi figures |
| `partE_justification.md` | E: the one-page justification |

## Self-checks

Every self-check is an `assert`, so a notebook stops if a rebuild is not faithful.

| Check | Target | Mine |
|---|---|---|
| 1 | 60 marks, largest 589× the smallest, 9 countries under 2.0 M below a point | yes. 589× is the **diameter** ratio with `s=pop**2`; the area ratio is 346,921×. The 9 are under 2 pt across (s < 4 pt²) |
| 2 | 24 districts, 30,564 cases, 57,857,000 people, top 3 D17, D13, D09 | yes |
| 3 | 58 of 60 below 0.10 on min-max GDP, median 0.017 | yes |
| 4 | access and cases 3 axes apart, urban + age between | yes |
| 5 | rate 31.07 to 33.40, spread 7.2% | yes |
| 6 | pooled 8.72, U32 40.9 = 4.7× from 9 in 22,000; 10 units > 100k between 5.5 and 12.0 | yes |
| 7 | r = −0.308 | yes |
| 8 | top 5 by population: 12.6% of land, 59.9% of people | yes |
| 9 | means 62.3 / 62.0, areas 9,933 / 9,433 | yes |
| C | r = +0.935, top 3 by count and by rate, overlap 0, 7 move ≥3 places | yes |
| D2 | greedy chain access → cases → urban → age → GDP, r = −0.799 | yes (on raw columns) |
| D3 | 0 join failures, pooled 52.8 from 30,564, top 3 by rate D19, D20, D21, r(cases, pop) = +0.986 | yes |
| Zones | r by zone −0.889 | yes (mean of member incomes; population-weighted gives −0.746) |

## Chart analysis

**Part A — nine charts.** Every chart is arithmetically right. The lie is in a channel or a boundary:

1. **Bubbles:** `s=pop**2` puts population on the radius. The biggest mark is 589× across and 346,921× in area (Lie Factor 590), the two biggest hold 85% of the ink, and 57 of 60 centres sit under a bigger mark. Size and access are only weakly related (r = +0.21), and the real pattern is access vs log GDP (r = +0.83).
2. **Count map:** a population map, r(cases, pop) = +0.986. Per head the top three are D19-D21 (69-76 per 100k), and they share no district with the count top three.
3. **Flat GDP:** min-max with one $120,219 outlier. GDP actually varies the most of any column (425×, CV 2.65). log10 before min-max spreads it (median 0.35).
4. **Axis order:** access and cases were 3 axes apart. Side by side, 1,435 of 1,770 line pairs cross (r = −0.80).
5. **Five-level gradient:** quantile classes over a 2.3-point spread. Poisson noise alone is ±4.4, χ² = 2.30 on 23 df, so all 24 districts are within chance of the national rate.
6. **U32:** 9 cases in 22,000. One case moves its rate 4.5 per 100k, and only 1 of 40 units is outside the 95% funnel (2 expected by chance).
7. **r = −0.31:** the areal unit sets it. The same data in 6 zones gives −0.89 (MAUP, scale effect).
8. **"Rural":** area read as people. The five biggest districts are 12.6% of the land and 59.9% of the people, and rate vs density r = +0.00.
9. **Radar:** enclosed area depends on axis order. Across 60 orders A wins 39 and B wins 21, while the means differ by 0.33.

Cases 3 and 4 are the two fixed without changing chart type (scaling and axis order).

**Part B — decisions.** `sizes = (6, 3,534)` pt², area proportional to population, diameter ratio 24.3×. All four overplotting views show the cluster, but only alpha (0.04) and hexbin (gridsize 45) show it's denser than the core; I'd publish the hexbin. PCA: PC1 + PC2 = 83.7%, so a 2-axis view misses 16.3%. Degree-shoelace areas overstate the four northernmost districts by 5.5% (D21 worst), and Mercator inflates area 1.24× to 1.49× (geolib's self-check line uses sec instead of sec², so it understates this). Funnel: 1 of 40 outside, and telling 9 from 18 per 100k takes about 42,700 people (249,000 for non-overlapping bands).

**Part C — two maps.** Both maps are built with labelled counties. The four polygons with no county (D21-D24) are counted and hatched. The harness takes readers on screen or from paper. *The five-reader run still has to be done*, and no results are invented.

**Part D — three figures.** D1: access rises with GDP in every region (r = +0.83; +0.71 to +0.93 within regions), with population on area and a size legend, and Aldonia kept and labelled. D2: log10 GDP then min-max, greedy chain GDP → access → cases → urban → age, Borealis brushed: the access–cases X is the clearest pattern. D3: cases per 100,000 residents, Jenks 5 classes, equal-area, 0 join failures, nothing suppressed, with a sorted count bar chart beside the map.

**Part E — justification.** One page: channel budget, PCP scaling and order, denominator / scheme / projection, and the counterfactual (the count map points at D17, D13, D09, two of which are below the national rate).

**Bonus (Part F, separate folder).** COVID-19 deaths per 100,000 to the end of 2022 on WHO / UN / Natural Earth v5.1.2 data: Peru (640) and Eastern Europe are darkest per head, while a count map would have pointed at the US, Brazil and India.

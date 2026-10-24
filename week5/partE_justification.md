# Week 5 - Part E: Written justification

**Syed Sami Ullah · BSAI-7B · CDB601220 Data Visualization** · figures: `submission/partD_bubble.png`, `partD_pcp.png`, `partD_choropleth.png` · all data is synthetic course data. I used `geolib` for every map, class break and projection (no geopandas or mapclassify).

## 1. Which variable got which channel, and why

I spent the slide-8 budget in the order the reader has to compare things: two positions, then area, then hue.

| Rank | Channel | D1 bubble chart | Why it got this channel |
|---|---|---|---|
| 1 | y position | healthcare access (0-100) | the outcome. It needs the most precise comparison, and the axis has real bounds |
| 2 | x position | GDP per capita, **log** scale | the main driver, which needs precise reading too. Its spread is multiplicative (425×, Aldonia 9.2× the next), so a linear axis would crush 59 countries into a corner |
| 3 | area | population | only needed roughly ("big or small country"), and area is read worse than position. **Area, not radius**: `s = 6 × pop / 0.6 M` pt², so the area ratio is 589× and the diameter ratio √589 = 24×. A size legend (10 / 100 / 300 M) converts area back into people |
| 4 | hue | region (6) | identity, not amount. Six hues tested all-pairs: worst ΔE 8.4 (tritan), 10.3 deutan, all "ok". Legend outside the plot, key countries labelled directly |

In D3 the map spends both positions on geography before showing any number, so the rate goes on **lightness** (one sequential blue ramp, dark = more) and the count goes on **length** in the bar chart beside it.

## 2. The parallel coordinates decisions

- **Scaling: log10 on GDP, then min-max on all five axes.** Plain min-max let Aldonia flatten GDP (58 of 60 below 0.10, median 0.017, case 3). z-score leaves the same squeeze (Aldonia z = +7.5, the other 59 use 11% of the axis). Rank removes it but also throws away the gap sizes. After log10 the GDP median sits at 0.35 and the real min and max ($283, $120,219) are printed at the axis ends.
- **Axis order: the slide-24 greedy chain by |r|, computed on the columns as drawn, before drawing.** On raw data the chain is access → cases → urban → age → GDP (the brief's self-check, asserted). With log GDP, r(GDP, access) rises from 0.36 to 0.83, so the same rule puts GDP first: **GDP → access → cases → urban → age**. Access and cases are adjacent either way. (`geolib.corr_chain` crashes on pandas 3 because `DataFrame.values` is read-only there, so I reimplemented the same rule and assert it gives the brief's chain.)
- **Brushing:** everything grey, and one region in orange chosen by a rule fixed first: the highest median case rate. That's Borealis.
- **What it revealed that a scatter matrix wouldn't:** Borealis isn't a tidy cluster in any one pair. Its GDP and access lines spread over most of the range, so in a SPLOM its 10 points would just mix into all 10 panels. Followed as lines, they converge on the right-hand side: high case rates (median 119.6 vs 100.0) together with the least urban populations (31% vs 49%). That's one group's profile across five variables at once, which is what the PCP gives and pairwise panels don't.

## 3. The spatial decisions

- **Denominator: resident population, rate per 100,000.** Counts follow population at r = +0.986, so a count map is a population map. Per 100,000 is the unit the brief and the health data use, and it puts the national figure at a readable 52.8. The legend title says it in words: "Cases per 100,000 residents".
- **Classification: natural breaks (Jenks), 5 classes, breaks printed in the legend** (30.4 / 33.8 / 47.5 / 54.4 / 64.8 / 76.5). Highest GVF (0.969; quantile 0.908). The biggest gap in the sorted rates (33.8 → 42.2) becomes a class edge, where quantile would cut through clusters and invent steps (case 5 shows how bad that gets).
- **Projection: Lambert cylindrical equal-area, one reference latitude (30.5°N) for every layer.** Colour over an area makes area the encoding. Measured on these polygons, Mercator would inflate the north 1.49× and the south 1.24×, about 20% extra ink for the northern classes (Part B4). Areas were computed on the sphere, never in degrees (a degree shoelace overstates the north 4 districts by 5.5%).
- **Join and suppression:** 24 of 24 districts joined, 0 failures, counted both ways. Rule fixed before mapping: no rate published from fewer than 20 cases or 50,000 residents. **Nothing was suppressed** (smallest district: 186 cases, 468,000 people). The rule would bite at the small-area level: in `smallarea_cases.csv` it would hold back 37 of 40 units, U32 included, because at 8.7 per 100,000 a unit needs about 230,000 people to expect 20 cases. Aggregating those units up to districts is the honest move, even though it loses exactly the "interesting" spike case 6 was built on.

## 4. The counterfactual

My raw-count map would have coloured **D17, D13 and D09** darkest, the three districts with the most people (6.3-11.0 M each), and it would have looked like a capital-region outbreak. Two of those three (D17 at 51, D09 at 50 per 100,000) are actually *below* the national rate of 52.8. The districts where a resident is most at risk, **D19, D20 and D21** at 69-76 per 100,000, would have been pale mid-blues, and the response would have gone where the people are, not where the risk is. The count map points at the biggest cities, every week, whatever happens. Published often enough, that gives "Qilabad" a reputation for disease the data never supported, and the cost lands on the people who live there, not on whoever drew the map. Part C tests this on real readers: with the crime file the two maps' top threes share none of three counties, and *[mean overlap from `submission/partC_reader_results.csv` goes here once the five readers are done]*.

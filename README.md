# Data Visualization — BSAI-7B

Coursework for the Data Visualization course, organised by week.

| Week | Topic | Folder |
|---|---|---|
| 1 | Why we visualize, and choosing the right chart | [`week1/`](week1/) |
| 2 | Visual perception — preattentive attributes, Gestalt, cognitive load | [`week2/`](week2/) |
| 3 | Matplotlib & Seaborn — nine charts that are wrong, Lie Factor, CVD-safe palettes, the clinical review figure | [`week3/`](week3/) |
| 4 | Time series, proportions & hierarchies — nine charts broken upstream, windows and smoothers, pie vs bars, the e-commerce review figure | [`week4/`](week4/) |
| 5 | Multivariate data & maps — nine charts broken by the encoding or the geography, bubble chart, parallel coordinates, normalized choropleth | [`week5/`](week5/) |
| 6 | Class task: the integration checkpoint — audit and clean three grid files (eleven defects, decisions table), three charts, the normalized reliability figure for the board, real-data bonus | [`week6/`](week6/) |

Notebooks are committed with their outputs, so the charts are readable on GitHub without running anything.

## Running the notebooks

```bash
# Week 1
cd week1
python download_assignment_data.py
jupyter notebook

# Week 2
cd week2
python download_assignment_data.py
jupyter notebook

# Week 3 (data already included)
cd week3
python vizlib.py
jupyter notebook

# Week 4 (data already included; run Part D before Part B)
cd week4
python tslib.py
jupyter notebook

# Week 5 (data already included; no geopandas needed)
cd week5
python geolib.py
jupyter notebook

# Week 6 (data already included; run Part A before Parts B and C)
cd week6
python selfcheck.py
jupyter notebook
```

Requires `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `jupyter`, `ipykernel`. Week 4 also needs `statsmodels` and `squarify`. Week 5 also needs `scikit-learn`.

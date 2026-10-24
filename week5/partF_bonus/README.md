# Week 5 — Part F (bonus): a normalized map on real data

Kept separate from Parts A-E. Marked out of 5 and not added into the 100.

**COVID-19 deaths per 100,000 people, cumulative to 31 December 2022, by country**: a real count divided by a real population, on real boundaries, five classes, equal-area.

| File | What it is |
|---|---|
| `partF_map.ipynb` | The notebook: join (and what failed to join), exclusions with rule and count, scheme comparison, the map |
| `partF_map.py` | Same code as a script (`python partF_map.py`) |
| `partF_map.png` | The 300 dpi map |
| `get_data.py` | Re-creates both data files from the original sources |
| `data/covid_deaths_2022.csv` | 233 countries/territories: reported deaths to 2022-12-31, 2022 population |
| `data/ne_110m_admin_0_countries_v5.1.2.geojson` | The boundaries, as downloaded |

**Sources** (all downloaded 24 October 2026):

- **Deaths:** WHO COVID-19 dashboard counts, as republished by Our World in Data, <https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/cases_deaths/full_data.csv> (`total_deaths` on 2022-12-31).
- **Population:** UN World Population Prospects 2022, as packaged by Our World in Data, <https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv>.
- **Boundaries:** Natural Earth 1:110m Admin 0 – Countries, **release v5.1.2**, <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_admin_0_countries.geojson>. Natural Earth draws de facto boundaries (Kashmir is split on the Line of Control, which matches neither Pakistan's nor India's official map).

**Decisions.** Natural breaks, 5 classes (GVF 0.957 vs 0.925 equal interval and 0.887 quantile). Behrmann projection (cylindrical equal-area, true scale 30°). Antarctica dropped. The three countries reporting zero deaths for 2020-2022 (North Korea, Turkmenistan, Falkland Islands) are shown as "no usable report", not as the lowest class. Five polygons with no separate WHO figure (Taiwan, N. Cyprus, Somaliland, W. Sahara, Fr. S. Antarctic Lands) are hatched. 62 small territories have no polygon at 1:110m.

**What the raw-count version would have shown.** The US, Brazil, India, Russia and Mexico, the countries with the most reported deaths, would be darkest because they're among the most populous. India (530,693 deaths) is only 37 per 100,000, while on a five-class count map Peru, the worst per head at 640, would sit in the second-lightest class and Bulgaria (561) in the lightest. The two top-five lists share no country.

Needs `geolib.py` / `vizlib.py` from the `week5/` folder above (the code adds it to the path).

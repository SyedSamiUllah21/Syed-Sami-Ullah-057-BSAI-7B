"""
Part F data: download the real COVID-19 death counts, the populations and the country boundaries,
and write the two small files this bonus uses.

Sources (all downloaded 24 October 2026):
  Deaths      WHO COVID-19 dashboard counts as republished by Our World in Data (covid-19-data repository),
              public/data/cases_deaths/full_data.csv -> `total_deaths` on 2022-12-31 (cumulative confirmed deaths)
  Population  UN World Population Prospects 2022, as packaged by Our World in Data,
              scripts/input/un/population_latest.csv (mid-2022 population; 2021 for a few territories)
  Boundaries  Natural Earth 1:110m Cultural Vectors, Admin 0 - Countries, release v5.1.2
              (natural-earth-vector repository, geojson/ne_110m_admin_0_countries.geojson)

Only the standard library and pandas are needed. The two output files are already in data/,
so you only need this script to re-create them.
"""
from pathlib import Path
import io
import urllib.request

import pandas as pd

DEATHS = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/cases_deaths/full_data.csv"
POP = "https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv"
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_admin_0_countries.geojson"
CUTOFF = "2022-12-31"
HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "data" / "covid_deaths_2022.csv"
OUT_GEO = HERE / "data" / "ne_110m_admin_0_countries_v5.1.2.geojson"

# names that differ between the WHO/OWID death file and the UN population file
ALIAS = {"East Timor": "Timor"}


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


if __name__ == "__main__":
    OUT_CSV.parent.mkdir(exist_ok=True)
    deaths = pd.read_csv(io.BytesIO(fetch(DEATHS)), usecols=["date", "location", "total_deaths"])
    deaths = deaths[deaths.date == CUTOFF].drop(columns="date")
    pop = pd.read_csv(io.BytesIO(fetch(POP)))
    pop = pop[~pop.iso_code.str.startswith("OWID_") | (pop.iso_code == "OWID_KOS")]      # drop continents / income groups
    deaths["entity"] = deaths.location.replace(ALIAS)
    df = deaths.merge(pop[["entity", "iso_code", "year", "population"]], on="entity", how="inner")
    df["iso3"] = df.iso_code.replace({"OWID_KOS": "KOS"})                                 # Natural Earth's code for Kosovo
    df = df.rename(columns={"total_deaths": "deaths_to_2022_12_31", "year": "population_year"})
    df[["location", "iso3", "deaths_to_2022_12_31", "population", "population_year"]].sort_values("iso3").to_csv(OUT_CSV, index=False)
    OUT_GEO.write_bytes(fetch(NE))
    print(f"wrote {OUT_CSV.name}: {len(df)} countries/territories with deaths and population on {CUTOFF}")
    print(f"wrote {OUT_GEO.name}: {OUT_GEO.stat().st_size / 1e3:,.0f} kB")

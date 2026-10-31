"""
Week 6 Task D2 - download the real data and save the extract the bonus uses.

Source: Our World in Data, energy dataset (built on Ember yearly electricity
data and the Energy Institute Statistical Review), owid-energy-data.csv from
https://github.com/owid/energy-data . Downloaded 31 Oct 2026; the SHA-256 of
the file I downloaded is printed and written next to the extract.

The extract keeps EVERY entity (countries and aggregates alike) and every
electricity column, 1985 onwards, exactly as published. Nothing is cleaned
here: the audit and the cleaning happen in partD_real_data.py.

    python get_data.py
"""
import datetime
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "owid_energy_extract.csv"

COLS = ["country", "year", "iso_code", "population", "electricity_generation",
        "coal_electricity", "gas_electricity", "oil_electricity", "fossil_electricity",
        "hydro_electricity", "nuclear_electricity", "solar_electricity", "wind_electricity",
        "biofuel_electricity", "other_renewable_electricity", "other_renewable_exc_biofuel_electricity",
        "renewables_electricity", "low_carbon_electricity", "solar_share_elec"]

raw = urllib.request.urlopen(URL, timeout=120).read()
sha = hashlib.sha256(raw).hexdigest()
df = pd.read_csv(io.BytesIO(raw))
ext = df.loc[df.year >= 1985, COLS]
OUT.parent.mkdir(exist_ok=True)
ext.to_csv(OUT, index=False)
(OUT.parent / "SOURCE.txt").write_text(
    f"{URL}\ndownloaded {datetime.date.today().isoformat()}\nsha256 of full file: {sha}\n"
    f"full file: {df.shape[0]} rows x {df.shape[1]} columns; extract: year >= 1985, {len(COLS)} columns, {len(ext)} rows\n",
    encoding="utf-8")
print(f"full file {df.shape}, sha256 {sha[:16]}...  ->  {OUT.name}: {ext.shape}")

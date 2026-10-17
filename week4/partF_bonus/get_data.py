"""
Part F data: download the real PBS dataset and write the monthly table this bonus uses.

Source: Pharmaceutical Benefits Scheme (PBS) monthly script and cost data, Medicare Australia,
July 1991 - June 2008, as published in Rob Hyndman's `tsibbledata` R package (tidyverts).
File: https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data/PBS.rda
Downloaded: 17 October 2026.

Needs `pyreadr` (pip install pyreadr). The CSV it writes is already in data/, so you only need
this script to re-create it from the original file.
"""
from pathlib import Path
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data/PBS.rda"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "PBS.rda"
OUT = HERE / "data" / "pbs_alimentary_monthly.csv"

if __name__ == "__main__":
    import pyreadr

    RAW.parent.mkdir(exist_ok=True)
    if not RAW.exists():
        urllib.request.urlretrieve(URL, RAW)
    pbs = pyreadr.read_r(str(RAW))["PBS"]
    # R stores the yearmonth as days since 1970-01-01
    pbs["month"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(pbs["Month"], unit="D")
    alim = (pbs[pbs.ATC1 == "A"]
            .groupby(["month", "ATC2", "ATC2_desc"], as_index=False)[["Scripts", "Cost"]].sum()
            .rename(columns={"ATC2": "atc2", "ATC2_desc": "atc2_desc", "Scripts": "scripts", "Cost": "cost_aud"}))
    alim["month"] = alim.month.dt.strftime("%Y-%m")
    alim.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(alim):,} rows, {alim.atc2.nunique()} categories, {alim.month.min()} to {alim.month.max()}")

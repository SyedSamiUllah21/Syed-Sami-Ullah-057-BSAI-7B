# %% [markdown]
# # Week 6 - Task A: make the files trustworthy
#
# One script: raw CSVs + GeoJSON in, a printed audit, the fixes, a decisions table, and cleaned copies in `clean/`. Nothing is edited by hand and the files in `data/` are never written to.
#
# Everything the audit prints also goes to `partA_audit.txt`, and the decisions table is written to `partA_decisions.md` with the row counts filled in by the code (not typed).
#
# All data is synthetic course data for CDB601220 (Week 6).

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    HERE = Path(__file__).resolve().parent      # running as submission/partA_process.py
    IN_NOTEBOOK = False
except NameError:
    HERE = Path.cwd() / "submission"            # running the notebook from week6/
    IN_NOTEBOOK = True


def find_root(start):
    """The folder that holds data/ and the course toolkits (week6/, or the course's assignments/)."""
    for cand in (start.parent, start, start.parent / "assignments", start / "assignments"):
        if (cand / "data" / "district_grid.csv").exists() and (cand / "tslib.py").exists():
            return cand
    raise SystemExit("Can't find data/ and the toolkits. Put this folder next to them (see README).")


ROOT = find_root(HERE)
sys.path.insert(0, str(ROOT))
import geolib as gl
import tslib as tl

DATA = ROOT / "data"
OUT = HERE
CLEAN = OUT / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

LOG = []


def say(*parts):
    """print() that also keeps the line for partA_audit.txt"""
    line = " ".join(str(p) for p in parts)
    print(line)
    LOG.append(line)


def head(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


DECISIONS = []


def decide(file, what, detected, rule, rows):
    DECISIONS.append({"file": file, "what": what, "detected": detected, "rule": rule, "rows": rows})


SENTINELS = [-1, -9, -99, -999, 999, 9999]
pd.set_option("display.width", 140)

# %% [markdown]
# ## 1 · The district register (`district_grid.csv`)
#
# Read everything as it comes, then look: dtypes, the key, sentinels, the text column, and the range of anything I will divide by.

# %%
head("FILE 1  district_grid.csv  (the district register)")
reg_raw = pd.read_csv(DATA / "district_grid.csv")
reg = reg_raw.copy()
say(f"rows {len(reg)}, columns {reg.shape[1]}")
say("dtypes: " + ", ".join(f"{c}={t}" for c, t in reg.dtypes.astype(str).items()))

# key
dup_codes = reg[reg.code.duplicated(keep=False)]
say(f"key `code`: {reg.code.nunique()} unique of {len(reg)} rows -> duplicated: {sorted(dup_codes.code.unique())}")
say(dup_codes[["code", "district", "zone", "population", "households", "solar_kw_installed", "outages", "outage_minutes"]].to_string(index=False))

# sentinels in the numeric columns
num_cols = reg.select_dtypes("number").columns
for c in num_cols:
    hits = reg[reg[c].isin(SENTINELS)]
    neg = reg[reg[c] < 0]
    if len(hits) or len(neg):
        say(f"sentinel scan: {c} has {len(hits)} value(s) in {SENTINELS} -> {hits.code.tolist()} = {hits[c].tolist()}")

# the column that did not parse as a number
el = reg.electrification_pct
el_num = pd.to_numeric(el, errors="coerce")
el_text = reg[el.notna() & el_num.isna()]
say(f"electrification_pct dtype={el.dtype}: {el_num.notna().sum()} parse as numbers, "
    f"{len(el_text)} are text {el_text[['code', 'electrification_pct']].values.tolist()}, {el.isna().sum()} blank {reg.loc[el.isna(), 'code'].tolist()}")

# ranges of the divisors
pph = reg.population / reg.households
say("ranges of anything I divide by:")
for c in ["population", "households", "area_km2", "outages"]:
    say(f"   {c:<12} min {reg[c].min():>12,.1f} ({reg.loc[reg[c].idxmin(), 'code']})   max {reg[c].max():>12,.1f} ({reg.loc[reg[c].idxmax(), 'code']})")
say(f"   persons per household  min {pph.min():.2f}   median {pph.median():.2f}   max {pph.max():,.1f} ({reg.loc[pph.idxmax(), 'code']})")

# %% [markdown]
# Four things are wrong in the register, and each one has a different fix:
#
# - **D11 is in twice** (`Kotla` and `Kotla City`). Same population, households, area, point, income and electrification. The solar, outage and minutes columns differ by tiny amounts (12 kW, 3 interruptions, 410 minutes), so it's one district keyed twice, not two districts. I keep the row whose name matches the GeoJSON (`Kotla`).
# - **`electrification_pct` is text**: two values carry a `%` sign and three are blank. The `%` ones are repaired (strip and parse). The blanks stay missing, never zero.
# - **`outage_minutes = -1`** for D09 and D20 is the operator's not-available code. It becomes missing, so 22 districts have a usable duration.
# - **D21 has 182 households** for 1,081,000 people, which is 5,940 people per household. Every other district has a six- or seven-digit household count and 5.7-7.4 people per household. 182 is the count in thousands, so I repair it to 182,000 (5.94 per household, inside the others' range). Here I can state the true value and why, so it is a repair and not a drop.

# %%
# 1. duplicate key
geo_names = {f["code"]: f["name"] for f in gl.read_geojson(DATA / "regions.geojson")}
d11 = reg[reg.code == "D11"]
keep_mask = ~((reg.code.duplicated(keep=False)) & (reg.district != reg.code.map(geo_names)))
n_dup_dropped = int((~keep_mask).sum())
reg = reg[keep_mask].reset_index(drop=True)
diff_cols = [c for c in d11.columns if d11[c].nunique() > 1]
say(f"\nD11 rows differ only in: {diff_cols} -> kept '{geo_names['D11']}', dropped {n_dup_dropped} row ('Kotla City')")
decide("district_grid", "Duplicate key: D11 appears twice ('Kotla', 'Kotla City')",
       "`code.duplicated()`; the two rows agree on population, households, area, point and income",
       "Same code = same district. Keep the row whose name matches the GeoJSON `name` (Kotla), drop the other", f"{n_dup_dropped} dropped")

# 2-3. electrification as text
had_pct = reg.electrification_pct.astype("string").str.contains("%", na=False)
reg["electrification_pct"] = pd.to_numeric(reg.electrification_pct.astype("string").str.rstrip("%").str.strip(), errors="raise")
n_pct, n_el_missing = int(had_pct.sum()), int(reg.electrification_pct.isna().sum())
decide("district_grid", "`electrification_pct` stored as text, two values typed with a `%` sign",
       "dtype is str; `pd.to_numeric(errors='coerce')` fails on exactly these values",
       "Strip the `%` and parse. The number was always there, so this is a repair", f"{n_pct} repaired")
decide("district_grid", "`electrification_pct` blank",
       "`isna()` after parsing",
       "Left missing (NaN). Not zero and not imputed; anything that uses it says n = 21", f"{n_el_missing} left missing")

# 4. operator code in outage_minutes
code_rows = reg.outage_minutes.isin(SENTINELS)
n_om_code = int(code_rows.sum())
om_codes = reg.loc[code_rows, "code"].tolist()
reg["outage_minutes"] = reg.outage_minutes.where(~code_rows).astype("Int64")
decide("district_grid", "`outage_minutes = -1` (operator not-available code)",
       "Sentinel scan of every numeric column; -1 minutes is impossible",
       f"Set to missing. {' and '.join(om_codes)} keep their interruption count but have no duration", f"{n_om_code} set missing")

# 5. households in thousands
pph = reg.population / reg.households
bad_hh = reg[(pph < 3) | (pph > 12)]
assert bad_hh.code.tolist() == ["D21"], bad_hh
old_hh = int(bad_hh.households.iloc[0])
reg.loc[reg.code == "D21", "households"] = old_hh * 1000
decide("district_grid", f"D21 `households` = {old_hh} for {int(bad_hh.population.iloc[0]):,} people ({bad_hh.population.iloc[0] / old_hh:,.0f} per household)",
       "Range check on the divisor: persons per household must be 3-12; only D21 fails",
       "Recorded in thousands. Repair to x1000 = 182,000 (5.94 per household, inside the other 23's 5.70-7.36). Repair, not drop, because the true value can be stated", "1 repaired")

reg["persons_per_household"] = reg.population / reg.households
say(f"after cleaning: {len(reg)} rows, {reg.code.nunique()} codes, population {reg.population.sum():,}, households {reg.households.sum():,}")
say(f"persons per household {reg.persons_per_household.min():.2f} - {reg.persons_per_household.max():.2f}")
say(f"electrification {reg.electrification_pct.min()} - {reg.electrification_pct.max()} with {n_el_missing} missing")
mean_len = reg.outage_minutes / reg.outages
say(f"usable outage_minutes: {int(reg.outage_minutes.notna().sum())} of {len(reg)}, mean interruption {float(mean_len.min()):.0f} - {float(mean_len.max()):.0f} minutes")
say(f"interruptions, all districts: {reg.outages.sum():,}")

# the brief's confirmations
assert len(reg) == 24 and reg.code.is_unique
assert reg.population.sum() == 57_857_000 and reg.households.sum() == 8_908_512
assert round(reg.persons_per_household.min(), 2) == 5.70 and round(reg.persons_per_household.max(), 2) == 7.36
assert reg.electrification_pct.min() == 66.6 and reg.electrification_pct.max() == 90.3 and n_el_missing == 3
assert reg.outage_minutes.notna().sum() == 22 and round(float(mean_len.min())) == 77 and round(float(mean_len.max())) == 214
assert reg.outages.sum() == 388_879

# %% [markdown]
# ## 2 · The boundaries (`regions.geojson`) and the joins
#
# The join is counted both ways, and the shipped `area_km2` and the `lon`/`lat` points are checked against the geometry.

# %%
head("FILE 2  regions.geojson  (boundaries) and every join")
feats = gl.read_geojson(DATA / "regions.geojson")
geo_codes = [f["code"] for f in feats]
say(f"features {len(feats)}, unique codes {len(set(geo_codes))}, geometry type Polygon (read_geojson raises otherwise)")
matched = set(reg.code) & set(geo_codes)
miss_reg = sorted(set(reg.code) - set(geo_codes))
miss_geo = sorted(set(geo_codes) - set(reg.code))
say(f"join register <-> GeoJSON on code: {len(matched)} matched | register rows with no polygon: {len(miss_reg)} | polygons with no row: {len(miss_geo)}")
name_mismatch = [(c, reg.set_index('code').district[c], geo_names[c]) for c in matched if reg.set_index('code').district[c] != geo_names[c]]
say(f"name check after the D11 fix: {len(name_mismatch)} name mismatches")
geo_area = {f["code"]: gl.polygon_area_km2(f["ring"]) for f in feats}
dev = (reg.set_index("code").area_km2 / pd.Series(geo_area) - 1).abs()
say(f"area_km2 vs spherical polygon area (geolib.polygon_area_km2): max deviation {dev.max() * 100:.4f}%")
outside = [f["code"] for f in feats
           if not gl._point_in_ring(*reg.set_index("code").loc[f["code"], ["lon", "lat"]], f["ring"])]
say(f"lon/lat point inside its own polygon: {24 - len(outside)} of 24")
assert len(matched) == 24 and not miss_reg and not miss_geo and not name_mismatch and not outside
decide("regions.geojson", "Join of the register to the boundaries",
       "Set difference both ways on `code`; names, areas and points checked against the geometry",
       "No change needed: 24 matched, 0 unmatched either way, areas agree to < 0.001%", "0")

# %% [markdown]
# ## 3 · The daily returns (`grid_daily.csv`)
#
# The file is shuffled and three thousand rows long, so nothing here is found by reading. `tslib.audit_index` first runs on the file as it comes, to show it can't even be audited until the dates are dates.

# %%
head("FILE 3  grid_daily.csv  (six zones, daily returns)")
day_raw = pd.read_csv(DATA / "grid_daily.csv")
day = day_raw.copy()
say(f"rows {len(day)}, columns {day.shape[1]}")
say("dtypes: " + ", ".join(f"{c}={t}" for c, t in day.dtypes.astype(str).items()))
say(f"zones: {day.zone.value_counts().sort_index().to_dict()}  (joins to register zones: {sorted(set(day.zone)) == sorted(set(reg.zone))})")
a0 = tl.audit_index(day.set_index("date"), "D")
say(f"audit_index on the raw date column: dtype_ok={a0['dtype_ok']} monotonic={a0['monotonic']} -> nothing else can be checked yet")

shape = day.date.str.replace(r"\d", "9", regex=True)
say(f"date shapes: {shape.value_counts().to_dict()}")
slash = day.date.str.contains("/")
first, second = day.date[slash].str[:2].astype(int), day.date[slash].str[3:5].astype(int)
say(f"the 99/99/9999 rows: first field {first.min():02d}-{first.max():02d}, second field always {[int(x) for x in sorted(second.unique())]}, year {sorted(day.date[slash].str[-4:].unique())}"
    " -> day first (DD/MM/YYYY): a month can't be 30")

# %% [markdown]
# **Date rule.** Two shapes in one column. The 174 slash dates have a first field running 01-30 and a second field that is always 11, so they are DD/MM/YYYY (November 2025). Each shape is parsed with its own explicit format; nothing is guessed by `dayfirst` and nothing is sorted as text.

# %%
day["date"] = pd.to_datetime(day.date.where(~slash), format="%Y-%m-%d")
day.loc[slash, "date"] = pd.to_datetime(day_raw.date[slash], format="%d/%m/%Y")
day["date"] = pd.to_datetime(day.date)
n_slash = int(slash.sum())
iso_in_nov = int(((~slash) & (day.date.dt.to_period("M") == "2025-11")).sum())
say(f"parsed: {n_slash} DD/MM rows all land in {sorted(day.date[slash].dt.strftime('%Y-%m').unique())}; ISO rows in that month: {iso_in_nov}")
decide("grid_daily", "Two date formats in one column: 174 rows as DD/MM/YYYY (every November 2025 row)",
       "Shape of each string (digits -> 9); first field 01-30, second always 11",
       "Parse each shape with its own explicit format (`%Y-%m-%d`, `%d/%m/%Y`). No text sort, no format guessing", f"{n_slash} parsed")

monotonic_before = bool(day.date.is_monotonic_increasing)
day = day.sort_values(["zone", "date"], kind="stable").reset_index(drop=True)
say(f"file order: sorted by date before = {monotonic_before}; now sorted by zone, date")
decide("grid_daily", "Rows are in no order (shuffled)",
       "`is_monotonic_increasing` is False on the parsed dates",
       "Sort by zone, then date, after parsing. Never plot against row number", f"{len(day):,} reordered, 0 values changed")

# duplicates
key = ["zone", "date"]
dups = day[day.duplicated(key, keep=False)]
exact = day.duplicated(keep="first")
n_exact = int(exact.sum())
say(f"duplicate zone-days: {dups.drop_duplicates(key).shape[0]} keys on {len(dups)} rows; exact copies to drop: {n_exact}")
day = day[~exact].reset_index(drop=True)
conf = day[day.duplicated(key, keep=False)]
say("duplicates that disagree (same meter reading, different peak and minutes):")
say(conf.to_string(index=False))
assert (conf.groupby(key).cumulative_mwh.nunique() == 1).all()
conf_keys = conf.drop_duplicates(key)[key]
day = day.drop_duplicates(key, keep="first").reset_index(drop=True)
mark = day.set_index(key).index.isin(conf_keys.set_index(key).index)
day["filed_twice_conflict"] = mark
day.loc[mark, ["peak_load_mw", "outage_minutes"]] = np.nan
n_conf = int(mark.sum())
decide("grid_daily", "Zone-days filed twice, identical",
       "`duplicated(['zone','date'])` after parsing, then `duplicated()` on all columns",
       "Drop the copy, keep one", f"{n_exact} dropped")
decide("grid_daily", "Zone-days filed twice with the same meter reading but different peak and outage minutes",
       "Same key, `cumulative_mwh` identical, other fields differ",
       "Keep one row (the reading agrees). I can't tell which peak/minutes is right, so both fields become missing on that day", f"{n_conf} dropped, {n_conf} kept with 2 fields blanked")
say(f"zone-days kept: {len(day):,}")
assert len(day) == 3_046

# %% [markdown]
# Now the calendar can be checked, per zone, with `tslib.audit_index`.

# %%
span = pd.date_range(day.date.min(), day.date.max(), freq="D")
slots = len(span) * day.zone.nunique()
say(f"\nreporting span {span[0].date()} -> {span[-1].date()} = {len(span)} days x {day.zone.nunique()} zones = {slots:,} slots")
miss_rows = []
for z, g in day.groupby("zone"):
    a = tl.audit_index(g.set_index("date"), "D")
    say(f"   {z}: rows {a['n_rows']}  dtype_ok={a['dtype_ok']}  monotonic={a['monotonic']}  dupes={a['n_duplicated']}  "
        f"missing days={a['n_missing']}  partial_tail={a['partial_tail']} ({a['tail_period']}: {a['tail_count']} days)")
    for m in span.difference(g.date):
        miss_rows.append({"zone": z, "date": m})
miss = pd.DataFrame(miss_rows)
n_never = len(miss)
by_month = miss.groupby([miss.date.dt.strftime("%Y-%m"), "zone"]).size().sort_values(ascending=False)
say(f"zone-days never reported: {n_never} (= {slots} - {len(day)}); worst zone {miss.zone.value_counts().idxmax()} with {miss.zone.value_counts().max()}, "
    f"best {miss.zone.value_counts().idxmin()} with {miss.zone.value_counts().min()}; worst zone-month {by_month.index[0][1]} {by_month.index[0][0]} with {by_month.iloc[0]} days")
assert len(span) == 529 and slots == 3_174 and n_never == 128
decide("grid_daily", "Zone-days never reported (not evenly spread: P5 36, P1 12; P5 lost 9 days of December 2025)",
       "`tslib.audit_index(freq='D')` per zone against the full 529-day calendar",
       "Not filled. Each zone is reindexed to the full calendar so a gap is a NaN that breaks the line, never a zero and never a straight line across", f"{n_never} slots stay empty")

# sentinels in the daily file
for c in ["cumulative_mwh", "peak_load_mw", "outage_minutes"]:
    v = day[c]
    say(f"{c:<15} min {v.min():>14,.1f}  max {v.max():>14,.1f}  sentinel hits {int(v.isin(SENTINELS).sum())}  negatives {int((v < 0).sum())}  zeros {int((v == 0).sum())}")
pk_code = day.peak_load_mw.isin(SENTINELS)
n_pk = int(pk_code.sum())
say(f"peak_load_mw = -1 on {n_pk} zone-days: {day.loc[pk_code, 'zone'].value_counts().sort_index().to_dict()}")
day.loc[pk_code, "peak_load_mw"] = np.nan
assert n_pk == 46
decide("grid_daily", "`peak_load_mw = -1`, the SCADA not-available code",
       "Sentinel scan; a peak demand of -1 MW is impossible",
       "Set to missing. Peaks are summarised with mean/max over the readings that exist, and n is stated", f"{n_pk} set missing")
say(f"outage_minutes zeros: {int((day.outage_minutes == 0).sum())} spread over {day.loc[day.outage_minutes == 0, 'zone'].nunique()} zones; "
    f"max {day.outage_minutes.max():.0f} min < 1,440 in a day, so zeros are read as real no-interruption days")

# %% [markdown]
# **The meter.** `cumulative_mwh` is a reading on a counter, so the only meaningful energy is a *difference* between two readings. First check that it only goes up.

# %%
day["step"] = day.groupby("zone").cumulative_mwh.diff()
falls = day[day.step < 0]
say(f"\nzone-days where the meter reading goes DOWN: {len(falls)}")
for _, r in falls.iterrows():
    prev = day[(day.zone == r.zone) & (day.date < r.date)].iloc[-1]
    say(f"   {r.zone} {prev.date.date()} {prev.cumulative_mwh:,.1f} -> {r.date.date()} {r.cumulative_mwh:,.1f}  (then +{day[(day.zone == r.zone) & (day.date > r.date)].iloc[0].cumulative_mwh - r.cumulative_mwh:,.1f} next day)")
assert len(falls) == 1
RESET = falls.iloc[0]
day["meter_segment"] = day.groupby("zone").step.transform(lambda s: (s < 0).cumsum()).astype(int)
decide("grid_daily", f"Meter reading falls once: {RESET.zone} on {RESET.date.date()} (21,444,254.8 -> 1,180.0, then climbs ~42,000/day again)",
       "First difference of the reading within each zone, sorted by date; one negative step",
       "A meter replacement. Readings from that day on are meter segment 1. The difference across the swap is not energy, so it is missing (not repaired: the energy on the swap day is unknown)", "1 difference set missing")

# %% [markdown]
# **Energy convention.** Energy in a period = reading at the end of the period minus the reading at the start, inside one meter segment. The brief's monthly check is `reading on 1 July - reading on 1 June`, so it treats a reading as the meter at the start of its reporting day (like a billing read on the 1st). I use that one convention everywhere so my months match the published number: the energy of day *t* is `reading(t+1) - reading(t)`, and it exists only when day *t+1* was reported. With the dictionary's "end of day" wording every value moves by one day; for P1 in June 2026 that would give 1,199,207 MWh, 0.09% lower, and no chart changes.

# %%
frames = []
for z, g in day.groupby("zone"):
    full = g.set_index("date").reindex(span)
    full["zone"] = z
    nxt = full.cumulative_mwh.shift(-1)
    same_seg = full.meter_segment.shift(-1) == full.meter_segment
    full["energy_mwh"] = (nxt - full.cumulative_mwh).where(same_seg)
    full["reported"] = full.cumulative_mwh.notna()
    frames.append(full)
cal = pd.concat(frames).rename_axis("date").reset_index()
n_energy = int(cal.energy_mwh.notna().sum())
n_no_next = int((cal.reported & cal.energy_mwh.isna()).sum())
say(f"daily energy values: {n_energy:,} of {len(day):,} reported zone-days; {n_no_next} have none (next day unreported, the swap day, or the last day)")

# a month's energy: reading on the 1st of the next month minus the reading on the 1st.
# If a 1st is missing, the counter is interpolated in time between its two neighbours, for month totals only.
months = pd.period_range(span[0], span[-1], freq="M")
mrows, interp_edges = [], set()
for z, g in cal.groupby("zone"):
    s = g.set_index("date").cumulative_mwh
    seg = g.set_index("date").meter_segment
    for m in months:
        a, b = m.start_time.normalize(), min((m + 1).start_time.normalize(), span[-1])
        partial = (m + 1).start_time.normalize() > span[-1]
        vals, segs = [], []
        for t in (a, b):
            if pd.isna(s.get(t)):
                interp_edges.add((z, t))
                lo, hi = s[:t].last_valid_index(), s[t:].first_valid_index()
                w = (t - lo) / (hi - lo)
                vals.append(s[lo] + w * (s[hi] - s[lo]))
                segs.append(seg[lo] if seg[lo] == seg[hi] else -1)
            else:
                vals.append(s[t])
                segs.append(seg[t])
        ok = segs[0] == segs[1] and segs[0] >= 0
        mrows.append({"zone": z, "month": str(m), "energy_mwh": round(vals[1] - vals[0], 1) if ok else np.nan,
                      "days_covered": (b - a).days, "complete_month": not partial,
                      "note": "" if ok else "meter replaced in this month: not computed"})
monthly = pd.DataFrame(mrows)
n_interp = len(interp_edges)
say(f"monthly energy: {len(monthly)} zone-months; month-edge readings interpolated: {n_interp}; "
    f"not computed: {monthly.energy_mwh.isna().sum()} ({monthly.loc[monthly.energy_mwh.isna(), ['zone', 'month']].values.tolist()}); "
    f"partial final month: {months[-1]} ({monthly.loc[~monthly.complete_month, 'days_covered'].iloc[0]} day-steps, data end {span[-1].date()})")

p1_june = monthly.query("zone == 'P1' and month == '2026-06'").energy_mwh.iloc[0]
p1_sum = day[(day.zone == "P1") & (day.date.dt.to_period("M") == "2026-06")].cumulative_mwh.sum()
say(f"\nP1, June 2026 energy: differenced {p1_june:,.0f} MWh | the column summed {p1_sum:,.0f} MWh | {p1_sum / p1_june:,.0f}x too big")
assert round(p1_june) == 1_200_239 and round(p1_sum) == 574_845_315 and round(p1_sum / p1_june) == 479
decide("grid_daily", "`cumulative_mwh` is a counter (a level), not energy. Summing it for P1, June 2026 gives 574,845,315 MWh",
       "Data dictionary + the reading only ever rises; sum vs difference compared for one zone-month (479x)",
       "Energy = difference of readings inside one meter segment, never `sum()`. Day t = reading(t+1) - reading(t), only if t+1 was reported. Month = reading on the 1st of next month - reading on the 1st",
       f"{n_energy:,} daily values, {len(monthly)} zone-months")
decide("grid_daily", "Month-edge reading (the 1st) missing for some zone-months",
       "Reindexed calendar: NaN on a 1st",
       "For month totals only, interpolate the counter linearly in time between the two neighbouring readings. This splits a known change across days, it doesn't invent energy", f"{n_interp} edges")
decide("grid_daily", f"Final month {months[-1]} is partial (data ends {span[-1].date()})",
       "`audit_index` partial_tail; the calendar ends on the 11th",
       "Kept but flagged `complete_month = False`; drawn lighter and labelled, never compared with full months", f"{int((~monthly.complete_month).sum())} zone-months flagged")

# %% [markdown]
# ## 4 · The generation mix (`generation_mix.csv`)
#
# The Week 4 exclusivity test, written out: is every row a part, does each part belong to exactly one parent, and do the parts add up to the whole?

# %%
head("FILE 4  generation_mix.csv  (monthly generation by category)")
gen_raw = pd.read_csv(DATA / "generation_mix.csv")
gen = gen_raw.copy()
say(f"rows {len(gen)}, columns {gen.shape[1]}")
say("dtypes: " + ", ".join(f"{c}={t}" for c, t in gen.dtypes.astype(str).items()))
gm = pd.PeriodIndex(gen.month, freq="M")
say(f"months {gen.month.nunique()} ({gen.month.min()} -> {gen.month.max()}), calendar gaps {len(pd.period_range(gm.min(), gm.max(), freq='M').difference(gm.unique()))}, "
    f"rows per month {[int(x) for x in sorted(gen.groupby('month').size().unique())]}")
say(f"key month+source+subsource unique: {not gen.duplicated(['month', 'source', 'subsource']).any()}; mwh min {gen.mwh.min():,} max {gen.mwh.max():,}, sentinels {int(gen.mwh.isin(SENTINELS).sum())}")
say("categories: " + "; ".join(f"{s} > {', '.join(sorted(g.subsource.unique()))}" for s, g in gen.groupby("source")))
parents = gen.groupby("subsource").source.nunique()
say(f"each subsource under exactly one source: {bool((parents == 1).all())}")

tot_mask = gen.subsource.eq("Total") | gen.source.eq("All sources")
published = gen[tot_mask].set_index("month").mwh
parts = gen[~tot_mask].copy()
psum = parts.groupby("month").mwh.sum()
allsum = gen.groupby("month").mwh.sum()
say(f"rows that are totals, not parts: {int(tot_mask.sum())} ('All sources / Total', one per month)")
say(f"parts per month: {[int(x) for x in sorted(parts.groupby('month').size().unique())]} -> {len(parts)} part rows")
say(f"parts summed vs published total: max |difference| {int((psum - published).abs().max())} MWh in any month")
say(f"every row summed / published total: {[float(x) for x in sorted((allsum / published).round(4).unique())]} in every month -> {(allsum / published).mean() * 100:.1f}% of generation")
assert len(parts) == 144 and (psum - published).abs().max() <= 2 and round((allsum / published).mean() * 100, 1) == 200.0
decide("generation_mix", "The ministry's 'All sources / Total' row sits in the file as if it were a category",
       "Exclusivity test: every row summed = 200.0% of the total in every month; parts alone match the total within 2 MWh",
       "Move the total out of the parts table (kept separately for the reconciliation). Shares are computed over the 8 parts only", f"{int(tot_mask.sum())} moved out")

gen_total = psum.rename("generation_mwh")
sol = parts[parts.subsource == "Solar"].set_index("month").mwh
g_chg = gen_total.iloc[-1] / gen_total.iloc[0] - 1
s_chg = sol.iloc[-1] / sol.iloc[0] - 1
say(f"generation {gen_total.index[0]} -> {gen_total.index[-1]}: {g_chg * 100:+.1f}% | solar {s_chg * 100:+.1f}% in MWh, share {sol.iloc[0] / gen_total.iloc[0] * 100:.1f}% -> {sol.iloc[-1] / gen_total.iloc[-1] * 100:.1f}%")
assert round(g_chg * 100, 1) == -5.0 and round(s_chg * 100, 1) == 82.1
assert round(sol.iloc[0] / gen_total.iloc[0] * 100, 1) == 4.1 and round(sol.iloc[-1] / gen_total.iloc[-1] * 100, 1) == 7.9

# %% [markdown]
# ## 5 · Write the clean files, the audit and the decisions table

# %%
head("WRITE")
reg_out = reg.drop(columns=["persons_per_household"])
reg_out.to_csv(CLEAN / "district_grid_clean.csv", index=False)

day_out = cal[cal.reported].copy()
day_out = day_out[["date", "zone", "cumulative_mwh", "meter_segment", "energy_mwh", "peak_load_mw", "outage_minutes", "filed_twice_conflict"]]
day_out["filed_twice_conflict"] = day_out.filed_twice_conflict.astype(bool)
day_out["outage_minutes"] = day_out.outage_minutes.round().astype("Int64")
day_out["energy_mwh"] = day_out.energy_mwh.round(1)
day_out["meter_segment"] = day_out.meter_segment.astype(int)
day_out["date"] = day_out.date.dt.strftime("%Y-%m-%d")
day_out.to_csv(CLEAN / "grid_daily_clean.csv", index=False)
miss.assign(date=miss.date.dt.strftime("%Y-%m-%d")).to_csv(CLEAN / "grid_daily_unreported.csv", index=False)
monthly.to_csv(CLEAN / "grid_monthly_energy.csv", index=False)

parts.to_csv(CLEAN / "generation_parts.csv", index=False)
pd.DataFrame({"month": published.index, "published_total_mwh": published.values,
              "parts_sum_mwh": psum.reindex(published.index).values}).to_csv(CLEAN / "generation_totals.csv", index=False)

for f in sorted(CLEAN.glob("*.csv")):
    say(f"   clean/{f.name:<28} {sum(1 for _ in open(f)) - 1:>5} rows")
assert len(day_out) == 3_046 and len(reg_out) == 24 and len(parts) == 144

# raw files untouched
for name, raw in [("district_grid.csv", reg_raw), ("grid_daily.csv", day_raw), ("generation_mix.csv", gen_raw)]:
    assert raw.equals(pd.read_csv(DATA / name)), name
say("raw files in data/ re-read and compared: unchanged")

table = ["# Task A - decisions table", "",
         "Written by `partA_process.py`; the row counts come from the code. One row per decision.", "",
         "| # | File | What | How I detected it | Rule applied | Rows affected |",
         "|---|---|---|---|---|---|"]
for i, d in enumerate(DECISIONS, 1):
    table.append(f"| {i} | `{d['file']}` | {d['what']} | {d['detected']} | {d['rule']} | {d['rows']} |")
table += ["", f"Repaired = the true value can be stated (D21 households, the `%` values, date formats). "
          f"Dropped = a copy of another row ({n_dup_dropped + n_exact + n_conf} rows in all). "
          f"Set missing = an operator code or a value I can't recover ({n_om_code} outage durations, {n_el_missing} electrification, {n_pk} peaks, "
          f"{n_conf} conflicting peaks and minutes, 1 meter-swap difference). Nothing missing is ever turned into a zero.", ""]
(OUT / "partA_decisions.md").write_text("\n".join(table), encoding="utf-8")
say(f"wrote partA_decisions.md ({len(DECISIONS)} decisions)")
(OUT / "partA_audit.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print("wrote partA_audit.txt")

# %% [markdown]
# ## The decisions table

# %%
if IN_NOTEBOOK:
    from IPython.display import Markdown, display
    display(Markdown((OUT / "partA_decisions.md").read_text(encoding="utf-8")))

# %% [markdown]
# ## Self-check against the brief
#
# Every row of the brief's Task A table is an `assert` above, so if this cell runs they all passed.
#
# | Check | Brief | Mine |
# |---|---|---|
# | Rows in the register | 24 | 24 |
# | Population / households | 57,857,000 / 8,908,512 | same |
# | Persons per household | 5.70 - 7.36 | 5.70 - 7.36 |
# | Electrification | 66.6 - 90.3, 3 missing | same |
# | Usable `outage_minutes` | 22, mean 77 - 214 min | same |
# | Interruptions | 388,879 | 388,879 |
# | Zone-days kept | 3,046 | 3,046 |
# | Span / slots | 529 days, 3,174 | same |
# | Never reported | 128 | 128 |
# | `peak_load_mw` codes | 46 | 46 (plus 3 conflicting peaks I blanked myself) |
# | Meter falls | 1 | 1 (P4, 17 Feb 2026) |
# | P1 June 2026 | 1,200,239 vs 574,845,315 (479x) | same |
# | Part rows / parts vs total / all rows | 144 / within 2 MWh / 200.0% | same |
# | GeoJSON join | 24 / 0 | 24 / 0 |

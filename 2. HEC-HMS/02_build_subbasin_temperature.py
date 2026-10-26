# -*- coding: utf-8 -*-
"""
Step 2 - Sub-basin mean air temperature for the GCM-driven HEC-HMS runs.

HEC-HMS (Hargreaves ET) is driven by one temperature gage per sub-basin. For the
SSP runs these are built from station-wise (nearest-grid) NEX-GDDP-CMIP6 data of
the top-ranked GCM for each variable (Module 1, Table 4):
    tasmax : FGOALS-g3      tasmin : INM-CM4-8
    T_station = (tasmax + tasmin) / 2
    T_subbasin = sum_g  w_g * T_station,g
with the Thiessen weights in Data/Temperature_station_weights.csv. These are
the Met_1.met rain-gauge weights except for SUB3 (Sialsuk/Thingdawl swapped)
and SUB10 (Thingdawl 0.007127 instead of 0.07129); see the Note column.
The result is the "First" series (TASAVG_Stationwise_<ssp>_First.csv), i.e.
the series from the first-ranked GCMs.

Input  : Data/GCM_Stationwise/TASMAX_Stationwise_<ssp>_FGOALS-g3.csv
         Data/GCM_Stationwise/TASMIN_Stationwise_<ssp>_INM-CM4-8.csv
         Data/Temperature_station_weights.csv
Output : Data/GCM_Stationwise/reproduced/TASAVG_Stationwise_<ssp>_First.csv
         and a comparison with the file used in the paper.

Run    : python 02_build_subbasin_temperature.py
"""

import pandas as pd

from common import DATA_DIR, SW_DIR

TOP_GCM = {"tasmax": "FGOALS-g3", "tasmin": "INM-CM4-8"}
SSPS = ["ssp245", "ssp585"]


def subbasin_tavg(tmax: pd.DataFrame, tmin: pd.DataFrame) -> pd.DataFrame:
    station = (tmax + tmin) / 2.0
    w = pd.read_csv(DATA_DIR / "Temperature_station_weights.csv")
    out = {}
    for sub, g in w.groupby("Subbasin", sort=False):
        out[sub] = sum(station[r.Station] * r.Weight for r in g.itertuples())
    return pd.DataFrame(out)[[f"Sub{i}" for i in range(1, 13)]]


def main():
    out_dir = SW_DIR / "reproduced"
    out_dir.mkdir(exist_ok=True)
    for ssp in SSPS:
        tx = pd.read_csv(SW_DIR / f"TASMAX_Stationwise_{ssp}_{TOP_GCM['tasmax']}.csv",
                         parse_dates=["Date"]).set_index("Date")
        tn = pd.read_csv(SW_DIR / f"TASMIN_Stationwise_{ssp}_{TOP_GCM['tasmin']}.csv",
                         parse_dates=["Date"]).set_index("Date")
        tavg = subbasin_tavg(tx, tn)
        dst = out_dir / f"TASAVG_Stationwise_{ssp}_First.csv"
        tavg.reset_index().to_csv(dst, index=False)
        used = SW_DIR / f"TASAVG_Stationwise_{ssp}_First.csv"
        msg = ""
        if used.exists():
            u = pd.read_csv(used, parse_dates=["Date"]).set_index("Date")
            u.columns = [c.capitalize() for c in u.columns]
            diff = (tavg - u[tavg.columns]).abs().max().max()
            msg = f" | max |difference| vs file used in paper: {diff:.2e} degC"
        print(f"{ssp}: {len(tavg)} days, 12 sub-basins -> reproduced/{dst.name}{msg}")


if __name__ == "__main__":
    main()

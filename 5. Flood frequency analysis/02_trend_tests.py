# -*- coding: utf-8 -*-
"""
Step 2 - Trend tests and 30-year means of annual maximum discharge and stage
(Table 7, Section 4.4).

Mann-Kendall test (with tie correction, two-sided, alpha = 0.05) and Sen's
slope on the annual maxima of 1986-2014 (historical) and 2021-2099 (SSPs);
mean annual maxima for 2021-2047, 2048-2073 and 2074-2099.

Input  : Results/AMS_<stage>.csv (step 1)
Output : Results/Table7_trends_<stage>.csv

Run    : python 02_trend_tests.py [--stage hecssp|paper|observed]
"""

import argparse

import pandas as pd

from common import RESULTS_DIR, SCENARIOS, SUB_PERIODS, TREND_PERIOD, mann_kendall


def main(stage="paper"):
    ams = pd.read_csv(RESULTS_DIR / f"AMS_{stage}.csv")
    rows = []
    for scen in SCENARIOS:
        a, b = TREND_PERIOD[scen]
        d = ams[(ams.Scenario == scen) & ams.Year.between(a, b)]
        for var in ("Discharge_m3s", "Stage_m"):
            r = {"Scenario": scen, "Variable": var, "Trend_period": f"{a}-{b}", **mann_kendall(d[var].values)}
            r["Mean_annual_max"] = d[var].mean()
            if scen != "Historical":
                for label, p0, p1 in SUB_PERIODS:
                    r[f"Mean_{label}"] = ams[(ams.Scenario == scen) & ams.Year.between(p0, p1)][var].mean()
            rows.append(r)
    t7 = pd.DataFrame(rows)
    t7.round(4).to_csv(RESULTS_DIR / f"Table7_trends_{stage}.csv", index=False)
    cols = ["Scenario", "Variable", "Trend_period", "Z", "p", "Sen_slope", "Trend", "Mean_annual_max"]
    print(t7[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hecssp", "paper", "observed"], default="hecssp")
    main(ap.parse_args().stage)

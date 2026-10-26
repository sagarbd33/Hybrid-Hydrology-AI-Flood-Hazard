# -*- coding: utf-8 -*-
"""
Step 7 - Performance of the standalone HEC-HMS model (Table 5, HEC-HMS rows)
and summary of the long-term runs.

Calibration 2018-2020 and validation 2021 against CWC discharge at Bairabi,
using only days with an observation:
    NSE, RMSE, MAE, PBIAS (%), R2 (squared Pearson correlation),
    bias of the top-5 % observed flows (%).

Input  : Results/HMS_Outlet_Q_Observed_CalVal.csv  (from step 6)
         Data/Observed/Bairabi_Discharge_CWC.csv    (Date, Bairabi) - restricted,
                                                     not distributed
Output : Results/HMS_performance_CalVal.csv
         Results/HMS_annual_maxima_summary.csv      (all scenarios available)

Run    : python 07_evaluate.py
"""

import numpy as np
import pandas as pd

from common import CAL_PERIOD, FLOW_GAGE, OBS_FLOW_CSV, RESULTS_DIR, SCENARIOS, VAL_PERIOD


def scores(obs, sim) -> dict:
    ok = ~(np.isnan(obs) | np.isnan(sim))
    o, s = np.asarray(obs)[ok], np.asarray(sim)[ok]
    err = s - o
    top = o >= np.quantile(o, 0.95)
    return {
        "N_days": int(ok.sum()),
        "NSE": 1 - np.sum(err ** 2) / np.sum((o - o.mean()) ** 2),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "PBIAS_%": float(100 * np.sum(err) / np.sum(o)),
        "R2": float(np.corrcoef(o, s)[0, 1] ** 2),
        "Top5pct_bias_%": float(100 * np.sum(err[top]) / np.sum(o[top])),
    }


def evaluate_calval(sim_csv=None, obs_csv=OBS_FLOW_CSV, out_csv=None):
    sim_csv = sim_csv or RESULTS_DIR / "HMS_Outlet_Q_Observed_CalVal.csv"
    if not sim_csv.exists():
        print(f"[SKIP] {sim_csv.name} not found (steps 5-6 for Observed_CalVal)")
        return None
    if not obs_csv.exists():
        print(f"[SKIP] observed discharge not found: {obs_csv}\n"
              f"       (CWC data are restricted; place the file there to compute Table 5)")
        return None
    sim = pd.read_csv(sim_csv, parse_dates=["Date"]).set_index("Date")["Q_HMS_m3s"]
    obs = pd.read_csv(obs_csv, parse_dates=["Date"]).set_index("Date")[FLOW_GAGE]
    rows = []
    for name, (a, b) in (("Calibration", CAL_PERIOD), ("Validation", VAL_PERIOD)):
        j = pd.concat([obs.loc[a:b], sim.loc[a:b]], axis=1, keys=["obs", "sim"])
        rows.append({"Period": f"{name} {a[:4]}-{b[:4]}", **scores(j["obs"].values, j["sim"].values)})
    df = pd.DataFrame(rows)
    out_csv = out_csv or RESULTS_DIR / "HMS_performance_CalVal.csv"
    df.round(4).to_csv(out_csv, index=False)
    print(df.round(3).to_string(index=False))
    return df


def summarise_runs():
    rows = []
    for s in SCENARIOS:
        f = RESULTS_DIR / f"HMS_Outlet_Q_{s}.csv"
        if not f.exists():
            continue
        q = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Q_HMS_m3s"]
        amax = q.groupby(q.index.year).max()
        rows.append({"Scenario": s, "Start": q.index.min().date(), "End": q.index.max().date(),
                     "Mean_Q": q.mean(), "Mean_annual_max_Q": amax.mean(), "Max_Q": q.max()})
    if rows:
        df = pd.DataFrame(rows).round(3)
        df.to_csv(RESULTS_DIR / "HMS_annual_maxima_summary.csv", index=False)
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    evaluate_calval()
    summarise_runs()

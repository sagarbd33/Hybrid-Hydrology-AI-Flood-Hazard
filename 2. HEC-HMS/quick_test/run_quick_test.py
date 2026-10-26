# -*- coding: utf-8 -*-
"""
Quick test for Module 2 (about 5 s, no HEC-HMS installation needed).

1. Parameter table: re-extracts Table 2 from the calibrated .basin file and
   compares it with quick_test/expected/Table2_parameter_ranges.csv.
2. Project build: creates a short HEC-HMS project (Jun-Dec 2017, observed
   forcing) in quick_test/output/, then checks the control window, the gage
   time window and the rainfall written to HEC-DSS against the input CSV.
3. Station-wise forcing: rebuilds the 2015 SSP585 sub-basin temperature from
   tasmax/tasmin (step 2) and bias-corrects a 2015 sample of KACE-1-0-G
   rainfall at Lunglei with the saved XGBoost model (step 3), comparing with
   quick_test/expected/.
4. Result export and statistics: writes a known outlet hydrograph to a DSS
   file in the HEC-HMS result format, reads it back with the export function
   and computes the performance statistics against a synthetic observation,
   comparing with quick_test/expected/metrics.csv.

Run:  python quick_test/run_quick_test.py
Pass: prints "QUICK TEST PASSED".
"""

import importlib
import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from common import BASIN_FILE, MODELS_SW_DIR, PRECIP_GAGES, SW_DIR, TEMPLATE_DIR, read_forcing  # noqa: E402

params = importlib.import_module("01_parameter_table")
tavg = importlib.import_module("02_build_subbasin_temperature")
swbc = importlib.import_module("03_bias_correct_stationwise")
build = importlib.import_module("04_setup_projects")
export = importlib.import_module("06_export_results")
evaluate = importlib.import_module("07_evaluate")

OUT = HERE / "output"
CFG = dict(precip="Observed/ObservedPr.csv", temp="Observed/ObservedTemp.csv",
           start="2017-06-01", end="2017-12-31")


def check(name, ok, detail=""):
    print(f"   {'OK  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    ok = True

    print("1) Parameter table (Table 2)")
    df = params.parse_basin(TEMPLATE_DIR / BASIN_FILE)
    got = []
    for proc, label, key, etype in params.TABLE2:
        if key in df.columns:
            v = pd.to_numeric(df.loc[df.Type == etype, key], errors="coerce").dropna()
            if not v.empty:
                got.append((label, v.min(), v.max()))
    got = pd.DataFrame(got, columns=["Parameter", "Min", "Max"])
    exp = pd.read_csv(HERE / "expected" / "Table2_parameter_ranges.csv")
    m = got.merge(exp, on="Parameter", suffixes=("", "_exp"))
    ok &= check("ranges match", len(m) == len(exp) and np.allclose(m[["Min", "Max"]].values,
                                                                   m[["Min_exp", "Max_exp"]].values),
                f"({len(m)} parameters)")

    print("2) HEC-HMS project build (QuickTest, Jun-Dec 2017)")
    pdir = build.setup("QuickTest", cfg=CFG, projects_dir=OUT, verbose=False)
    ctrl = (pdir / "Calibration.control").read_text().replace("\r", "")
    ok &= check("control window", "Start Date: 31 May 2017" in ctrl and "End Date: 30 December 2017" in ctrl)
    gage = (pdir / "QuickTest.gage").read_text().replace("\r", "")
    ok &= check("gage window", "Start Time: 1 June 2017, 00:00" in gage and "End Time: 31 December 2017, 00:00" in gage)
    pr = read_forcing(CFG["precip"], CFG["start"], CFG["end"]).fillna(0)
    worst = 0.0
    for g, col in list(PRECIP_GAGES.items())[:4]:
        s = export.read_daily(pdir / "QuickTest.dss", f"//{g}/PRECIP-INC//1DAY/GAGE/", CFG["start"], CFG["end"])
        worst = max(worst, float(np.max(np.abs(s.values - pr[col].values))))
    ok &= check("DSS rainfall = CSV", worst < 1e-3, f"(max |diff| {worst:.1e} mm)")

    print("3) Station-wise forcing (2015, SSP585)")
    rd = lambda f: pd.read_csv(f, parse_dates=["Date"]).set_index("Date").loc["2015"]
    t = tavg.subbasin_tavg(rd(SW_DIR / "TASMAX_Stationwise_ssp585_FGOALS-g3.csv"),
                           rd(SW_DIR / "TASMIN_Stationwise_ssp585_INM-CM4-8.csv"))
    used = rd(SW_DIR / "TASAVG_Stationwise_ssp585_First.csv")
    used.columns = [c.capitalize() for c in used.columns]
    d = float((t - used[t.columns]).abs().max().max())
    ok &= check("sub-basin temperature", d < 0.02, f"(max |diff| {d:.3f} degC)")
    import joblib
    raw = pd.read_csv(HERE / "input" / "PR_Lunglei_ssp585_2015_raw.csv", parse_dates=["Date"]).set_index("Date")["Lunglei"]
    got = swbc.correct(raw, joblib.load(MODELS_SW_DIR / "PR_Stationwise_KACE-1-0-G_Lunglei_Tuned_XGBoost.pkl"), True)
    exp_bc = pd.read_csv(HERE / "expected" / "PR_Lunglei_ssp585_2015_BiasCorrected.csv", parse_dates=["Date"]).set_index("Date")["Lunglei"]
    d = float(np.max(np.abs(got.values - exp_bc.reindex(got.index).values)))
    ok &= check("rainfall bias correction", d < 1e-4, f"(max |diff| {d:.1e} mm)")

    print("4) Outlet export and statistics")
    from pydsstools.heclib.dss.HecDss import Open
    sample = pd.read_csv(HERE / "input" / "sample_outlet_flow.csv", parse_dates=["Date"]).set_index("Date")
    dss = OUT / "Cal.dss"
    with Open(str(dss), mode="rw") as fid:
        fid.put_ts("//Outlet/FLOW//1DAY/RUN:Cal/", values=sample["Q_sim"].values,
                   start_time=sample.index[0].strftime("%d%b%Y 00:00").upper(),
                   data_units="M3/S", data_type="PER-AVER")
    q = export.read_daily(dss, "//Outlet/FLOW//1DAY/RUN:Cal/", sample.index[0], sample.index[-1])
    ok &= check("export = written series", np.allclose(q.values, sample["Q_sim"].values, atol=1e-4))
    sc = evaluate.scores(sample["Q_obs"].values, q.values)
    exp_m = pd.read_csv(HERE / "expected" / "metrics.csv").iloc[0]
    good = all(abs(sc[k] - exp_m[k]) < 1e-3 for k in ["NSE", "RMSE", "MAE", "PBIAS_%", "R2"])
    ok &= check("NSE/RMSE/MAE/PBIAS/R2", good,
                f"(NSE {sc['NSE']:.3f}, RMSE {sc['RMSE']:.3f}, PBIAS {sc['PBIAS_%']:.2f} %)")

    print("\nQUICK TEST PASSED" if ok else "\nQUICK TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

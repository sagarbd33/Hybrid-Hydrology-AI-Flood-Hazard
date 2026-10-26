# -*- coding: utf-8 -*-
"""
Step 3 - XGBoost bias correction of the station-wise GCM forcing for HEC-HMS
(replaces BiasCorrectionForStationWiseTp.py and BiasCorrectionForStationWiseTemp.py).

Same model form as Module 1:
    corrected(t) = GCM(t) + bias_model[GCM(t), GCM(t-1), day-of-year]
  * precipitation: one model per rain gauge (KACE-1-0-G, nearest grid cell),
    clipped at 0. Haulawng uses the Lunglei model (Haulawng has no usable
    observed record for training; both stations share the same GCM cell).
  * temperature: one model per sub-basin for the sub-basin mean temperature
    built in step 2.

Input  : Data/GCM_Stationwise/PR_Stationwise_<period>_*.csv, TASAVG_Stationwise_<period>*.csv
         Models_StationWise/*.pkl
Output : Data/GCM_Stationwise/reproduced/*_BiasCorrected.csv
         Results/Stationwise_BC_comparison.csv  (vs the files used in the paper)

Run    : python 03_bias_correct_stationwise.py
"""

import warnings

import joblib
import numpy as np
import pandas as pd

from common import MODELS_SW_DIR, RESULTS_DIR, SW_DIR

warnings.filterwarnings("ignore", category=UserWarning)

PR_GCM = "KACE-1-0-G"
PR_STATIONS = ["Aizawl", "Sialsuk", "Neihbawi", "Kolasib", "Thingdawl", "Lunglei",
               "Hnahthial", "Haulawng", "Mamit", "kawrtethawveng", "Zawlnuam", "Serchhip"]
PR_MODEL_STATION = {s: s for s in PR_STATIONS}
PR_MODEL_STATION["Haulawng"] = "Lunglei"      # fallback, as in the original analysis

# (label, raw precipitation file, raw temperature file)
PERIODS = [
    ("Historical", "PR_Stationwise_Historical.csv", "TASAVG_Stationwise_Historical.csv"),
    ("ssp245", f"PR_Stationwise_ssp245_{PR_GCM}.csv", "TASAVG_Stationwise_ssp245_First.csv"),
    ("ssp585", f"PR_Stationwise_ssp585_{PR_GCM}.csv", "TASAVG_Stationwise_ssp585_First.csv"),
]


def correct(series: pd.Series, model, nonneg: bool) -> pd.Series:
    X = pd.DataFrame({"Modeled": pd.to_numeric(series, errors="coerce")})
    X["Modeled_Lag1"] = X["Modeled"].shift(1)
    X["DayOfYear"] = X.index.dayofyear
    X = X.dropna()
    out = X["Modeled"] + model.predict(X[["Modeled", "Modeled_Lag1", "DayOfYear"]])
    return out.clip(lower=0.0) if nonneg else out


def read(path):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    df.columns = [c if c[:3].lower() != "sub" else f"Sub{c[3:]}" for c in df.columns]
    return df


def main():
    out_dir = SW_DIR / "reproduced"
    out_dir.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for label, pr_file, t_file in PERIODS:
        # precipitation
        raw = read(SW_DIR / pr_file)
        pr = pd.DataFrame({s: correct(raw[s], joblib.load(
            MODELS_SW_DIR / f"PR_Stationwise_{PR_GCM}_{PR_MODEL_STATION[s]}_Tuned_XGBoost.pkl"), True)
            for s in PR_STATIONS})
        # temperature (step 2 output preferred for SSPs, else the file in Data/)
        t_src = out_dir / t_file if (out_dir / t_file).exists() else SW_DIR / t_file
        traw = read(t_src)
        tas = pd.DataFrame({f"Sub{i}": correct(traw[f"Sub{i}"], joblib.load(
            MODELS_SW_DIR / f"TASAVG_Stationwise_1_sub{i}_Tuned_XGBoost.pkl"), False)
            for i in range(1, 13)})

        for df, src, kind in ((pr, pr_file, "pr"), (tas, t_file, "tas")):
            name = src.replace(".csv", "_BiasCorrected.csv")
            df.index.name = "Date"
            df.reset_index().to_csv(out_dir / name, index=False)
            used = SW_DIR / name
            if used.exists():
                u = read(used).reindex(df.index)
                d = (df - u[df.columns]).abs()
                rows.append({"Period": label, "Variable": kind, "File": name,
                             "Mean_reproduced": float(df.values.mean()),
                             "Mean_used_in_paper": float(np.nanmean(u.values)),
                             "Mean_abs_diff": float(np.nanmean(d.values)),
                             "Max_abs_diff": float(np.nanmax(d.values))})
        print(f"{label:<11} bias-corrected pr (12 gauges) and temperature (12 sub-basins) -> reproduced/")

    if rows:
        cmp = pd.DataFrame(rows).round(4)
        cmp.to_csv(RESULTS_DIR / "Stationwise_BC_comparison.csv", index=False)
        print("\nComparison with the bias-corrected files used for the HEC-HMS runs in the paper:")
        print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()

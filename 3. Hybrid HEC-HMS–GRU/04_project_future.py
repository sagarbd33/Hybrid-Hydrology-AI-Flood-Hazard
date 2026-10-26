# -*- coding: utf-8 -*-
"""
Step 4 - Daily discharge projections with the hybrid HEC-HMS-GRU model
(Section 4.4): GCM-driven historical baseline and SSP245 / SSP585.

Input  : Data/Inputs/<Scenario>_EnsembleMean.csv
             Date, precip, tmin, tmax, srad  (ensemble mean of the bias-corrected
                                              basin-average GCM series, Module 1)
             simdischarge                     (HEC-HMS outlet flow, Module 2)
         Models/Hybrid_GRU_model.keras, Models/Hybrid_GRU_scalers.pkl
Output : Results/<Scenario>_EnsembleMean_OptiDischarge.csv
             (input columns + OptiDischarge, from day 15 onwards)
         Results/Projection_summary.csv
             mean annual maximum discharge per period, input range check

Run    : python 04_project_future.py
"""

import pandas as pd

from common import (FEATURES, INPUT_DIR, RESULTS_DIR, SCENARIOS, load_gru, load_scalers,
                    make_sequences, predict, read_csv)

PERIODS = {"Historical": [("1986-2014", "1986", "2014")],
           "Ssp245": [("2021-2047", "2021", "2047"), ("2048-2073", "2048", "2073"), ("2074-2099", "2074", "2099")],
           "Ssp585": [("2021-2047", "2021", "2047"), ("2048-2073", "2048", "2073"), ("2074-2099", "2074", "2099")]}


def project(scenario, model, scalers):
    df = read_csv(INPUT_DIR / f"{scenario}_EnsembleMean.csv")
    X, _, dates = make_sequences(df, FEATURES["hybrid"])
    q = predict(model, scalers, X).astype(float)
    out = df.loc[dates].copy()
    out["OptiDischarge"] = q
    out.index.name = "Date"
    out.reset_index().to_csv(RESULTS_DIR / f"{scenario}_EnsembleMean_OptiDischarge.csv", index=False)

    sx = scalers[0]
    feats = FEATURES["hybrid"]
    outside = ((df[feats] < sx.data_min_) | (df[feats] > sx.data_max_)).mean()
    rows = []
    for label, a, b in PERIODS[scenario]:
        amax = out.loc[a:b, "OptiDischarge"].groupby(out.loc[a:b].index.year).max()
        rows.append({"Scenario": scenario, "Period": label,
                     "Mean_annual_max_Q_m3s": float(amax.mean()),
                     "Mean_Q_m3s": float(out.loc[a:b, "OptiDischarge"].mean()),
                     **{f"Outside_training_range_{f}": outside[f] for f in feats}})
    print(f"{scenario:<10} {len(out)} days ({out.index[0].date()} to {out.index[-1].date()}); "
          f"inputs outside training range: " +
          ", ".join(f"{f} {100 * v:.0f}%" for f, v in outside.items() if v > 0))
    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, scalers = load_gru(), load_scalers()
    rows = []
    for s in SCENARIOS:
        rows += project(s, model, scalers)
    summ = pd.DataFrame(rows)
    summ.round(3).to_csv(RESULTS_DIR / "Projection_summary.csv", index=False)
    print("\n" + summ[["Scenario", "Period", "Mean_annual_max_Q_m3s", "Mean_Q_m3s"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()

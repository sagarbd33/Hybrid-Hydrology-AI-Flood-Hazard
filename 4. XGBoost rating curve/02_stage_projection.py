# -*- coding: utf-8 -*-
"""
Step 2 - Daily water-level (stage) projections at Bairabi for the historical
baseline and SSP245 / SSP585 (Section 4.4; input to the flood frequency
analysis, Module 5).

--method paper     (default) the method that produced the stage series of the
                   paper: an XGBoost model of the stage-discharge difference
                       stage(t) = Q(t) + f[Q(t), Q(t-1), day-of-year]
                   trained on the paired file of the same scenario
                   (Merged_Output_<ssp>.csv, train before 2021; hyper-parameters
                   by RandomizedSearchCV, 30 draws, 3-fold CV). The Historical
                   run uses the SSP245 model.
                   NOTE: when Q exceeds the training range the learned
                   difference stops changing, so stage then rises 1 m per
                   1 m3/s of extra discharge. The share of affected days is
                   reported.
--method observed  power-law rating curve fitted to observed discharge and
                   stage (step 1 --data observed; Models/RatingCurve_LS_observed.json).

Input  : projected discharge (Module 3 Results/ or Data/Inputs/)
Output : Results/Stage_<Scenario>_<method>.csv     Date, Discharge, Stage
         Results/Stage_annual_maxima_<method>.csv    annual maximum stage per year
         Results/Stage_summary_<method>.csv          mean annual maximum per period

Run    : python 02_stage_projection.py [--method paper|observed]
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from common import (BIAS_FEATURES, BIAS_PARAM_DIST, MODEL_DIR, PAIRED_FILE, RESULTS_DIR, SCENARIOS,
                    SEED, SPLIT_DATE, STAGE_MODEL_FOR, annual_max_by_period, bias_features,
                    read_csv, read_projection)

warnings.filterwarnings("ignore", category=UserWarning)


def train_stage_model(ssp):
    """XGBoost model of (stage - discharge) from the paired file of one SSP."""
    path = MODEL_DIR / f"StageModel_XGB_{ssp}.json"
    df = read_csv(PAIRED_FILE[ssp])
    d = bias_features(df["Modeled"])
    d["Bias"] = df["Observed"] - df["Modeled"]
    d = d.dropna()
    tr = d.index < SPLIT_DATE
    search = RandomizedSearchCV(XGBRegressor(random_state=SEED), BIAS_PARAM_DIST, n_iter=30, cv=3,
                                scoring="neg_mean_squared_error", random_state=SEED, n_jobs=1)
    search.fit(d.loc[tr, BIAS_FEATURES], d.loc[tr, "Bias"])
    model = search.best_estimator_
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(path)
    q_max = float(df.loc[df.index < SPLIT_DATE, "Modeled"].max())
    return model, q_max, search.best_params_


def main(method="paper"):
    RESULTS_DIR.mkdir(exist_ok=True)
    models, amax_rows, summ = {}, [], []
    if method == "observed":
        p = json.loads((MODEL_DIR / "RatingCurve_LS_observed.json").read_text())
    for scen in SCENARIOS:
        q = read_projection(scen)
        if method == "paper":
            ssp = STAGE_MODEL_FOR[scen]
            if ssp not in models:
                models[ssp] = train_stage_model(ssp)
                print(f"Stage model ({ssp}): {models[ssp][2]}")
            model, q_max, _ = models[ssp]
            X = bias_features(q).dropna()
            stage = X["Modeled"] + model.predict(X[BIAS_FEATURES])
            above = float((q > q_max).mean())
        else:
            stage = p["H0"] + p["c"] * q.clip(lower=0) ** p["m"]
            above = np.nan
        stage = stage.astype(float)
        out = pd.DataFrame({"Date": stage.index.strftime("%Y-%m-%d"),
                            "Discharge": q.loc[stage.index].values.round(4),
                            "Stage": stage.values.round(4)})
        out.to_csv(RESULTS_DIR / f"Stage_{scen}_{method}.csv", index=False)
        amax = stage.groupby(stage.index.year).max()
        for yr, v in amax.items():
            amax_rows.append({"Scenario": scen, "Year": yr, "Annual_max_stage_m": round(v, 4)})
        for label, v in annual_max_by_period(stage, scen):
            summ.append({"Scenario": scen, "Period": label, "Mean_annual_max_stage_m": v,
                         "Max_stage_m": float(amax.max()), "Year_of_max": int(amax.idxmax()),
                         "Share_of_days_Q_above_training_range": above})
        print(f"{scen:<10} stage {stage.index[0].date()}-{stage.index[-1].date()}: annual maxima "
              f"{amax.min():.2f}-{amax.max():.2f} m (max in {amax.idxmax()})"
              + ("" if np.isnan(above) else f"; Q above training range on {100 * above:.1f}% of days"))
    pd.DataFrame(amax_rows).to_csv(RESULTS_DIR / f"Stage_annual_maxima_{method}.csv", index=False)
    s = pd.DataFrame(summ)
    s.round(3).to_csv(RESULTS_DIR / f"Stage_summary_{method}.csv", index=False)
    print("\n" + s[["Scenario", "Period", "Mean_annual_max_stage_m"]].round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["paper", "observed"], default="paper")
    main(ap.parse_args().method)

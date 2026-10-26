# -*- coding: utf-8 -*-
"""
Step 1 - Discharge-stage rating curves at Bairabi (Table 6, Fig. 8; Section 4.3).

Two models predict stage H (m) from discharge Q (m3/s):
  * Least-squares power law  H = H0 + c * Q^m
      H0 by bounded 1-D search (-10 m to min(H)); c, m by linear regression of
      log(H - H0) on log(Q).
  * XGBoost regression of H on Q (500 trees max, depth 2, learning rate 0.05,
    subsample 0.9, early stopping after 20 rounds without improvement).

--data paper    (default) as in the paper: Q = HEC-HMS-GRU discharge of the SSP245
                run, paired with observed stage, Jun 2017 - Jul 2022
                (Data/restricted/Merged_Output_Ssp245.csv). Train: before 2021,
                test: 2021-2022 (year-based split). As in the original notebook,
                XGBoost early stopping monitors the test period.
--data observed observed discharge paired with observed stage (2018-2021,
                Data/restricted/Observed_Q_Stage_pairs.csv). Train: 2018-2020,
                test: 2021. The power law is fitted by non-linear least
                squares; XGBoost early stopping monitors a random 20 % of the
                training days, so the test year is never used for fitting.

Statistics: NSE, RMSE, MAE, R2, PBIAS and RMSE / PBIAS of the top 10 % of stages.

Output : Results/Table6_rating_curves_<data>.csv
         Results/Figures/Fig8_rating_curves_<data>.png   (scatter + residuals)
         Models/RatingCurve_XGB_<data>.json, Models/RatingCurve_LS_<data>.json

Run    : python 01_rating_curves.py [--data paper|observed]
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from xgboost import XGBRegressor

from common import (FIG_DIR, MODEL_DIR, OBS_PAIRS_FILE, PAIRED_FILE, RESULTS_DIR, SPLIT_DATE,
                    XGB_RC_EARLY_STOP, XGB_RC_PARAMS, read_csv, scores)

warnings.filterwarnings("ignore", category=UserWarning)


def fit_power_law(Q, H, h0_lower=-10.0):
    """H = H0 + c Q^m; returns (H0, c, m)."""
    Q, H = np.asarray(Q, float), np.asarray(H, float)

    def sse(h0):
        if h0 >= H.min():
            return 1e12
        y = np.log(H - h0)
        m, logc = np.polyfit(np.log(Q), y, 1)
        return np.sum((y - (logc + m * np.log(Q))) ** 2)

    h0 = minimize_scalar(sse, bounds=(h0_lower, H.min() - 1e-3), method="bounded").x
    m, logc = np.polyfit(np.log(Q), np.log(H - h0), 1)
    return float(h0), float(np.exp(logc)), float(m)


def fit_power_law_nonlinear(Q, H):
    """H = H0 + c Q^m by non-linear least squares on H itself (scipy curve_fit)."""
    from scipy.optimize import curve_fit
    p, _ = curve_fit(lambda q, h0, c, m: h0 + c * q ** m, np.asarray(Q, float), np.asarray(H, float),
                     p0=[float(np.min(H)) - 0.1, 0.5, 0.6], maxfev=20000)
    return tuple(float(v) for v in p)


def power_law(Q, h0, c, m):
    return h0 + c * np.asarray(Q, float) ** m


def load(data):
    if data == "paper":
        df = read_csv(PAIRED_FILE["Ssp245"]).dropna().rename(columns={"Modeled": "Q", "Observed": "H"})
    else:
        df = read_csv(OBS_PAIRS_FILE).dropna().rename(columns={"Discharge": "Q", "Stage": "H"})
    return df[df["Q"] > 0]


def main(data="paper", make_plots=True):
    df = load(data)
    tr, te = df.index < SPLIT_DATE, df.index >= SPLIT_DATE
    Qtr, Htr, Qte, Hte = df.loc[tr, ["Q"]], df.loc[tr, "H"], df.loc[te, ["Q"]], df.loc[te, "H"]

    # Least squares: log-space fit with H0 search as in the paper; for observed
    # pairs the non-linear fit in stage units (the log-space fit runs into the
    # -10 m bound for H0 and flattens the curve).
    if data == "paper":
        h0, c, m = fit_power_law(Qtr["Q"], Htr)
    else:
        h0, c, m = fit_power_law_nonlinear(Qtr["Q"], Htr)
    ls_tr, ls_te = power_law(Qtr["Q"], h0, c, m), power_law(Qte["Q"], h0, c, m)

    # XGBoost
    xgb = XGBRegressor(**XGB_RC_PARAMS, early_stopping_rounds=XGB_RC_EARLY_STOP)
    if data == "paper":
        eval_set = [(Qte, Hte)]                       # original notebook (test period monitored)
        fit_X, fit_y = Qtr, Htr
    else:
        rng = np.random.default_rng(42)               # random 20 % of training days
        val = np.zeros(len(Qtr), bool)
        val[rng.choice(len(Qtr), int(0.2 * len(Qtr)), replace=False)] = True
        fit_X, fit_y = Qtr[~val], Htr[~val]
        eval_set = [(Qtr[val], Htr[val])]
    xgb.fit(fit_X, fit_y, eval_set=eval_set, verbose=False)
    xg_tr, xg_te = xgb.predict(Qtr), xgb.predict(Qte)

    rows = []
    for model, ptr, pte in (("Least squares", ls_tr, ls_te), ("XGBoost", xg_tr, xg_te)):
        rows.append({"Model": model, "Period": f"Train (<{SPLIT_DATE[:4]})", **scores(Htr, ptr)})
        rows.append({"Model": model, "Period": f"Test (>={SPLIT_DATE[:4]})", **scores(Hte, pte)})
    t6 = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    t6.round(4).to_csv(RESULTS_DIR / f"Table6_rating_curves_{data}.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    xgb.save_model(MODEL_DIR / f"RatingCurve_XGB_{data}.json")
    (MODEL_DIR / f"RatingCurve_LS_{data}.json").write_text(json.dumps(
        {"form": "H = H0 + c * Q**m", "H0": h0, "c": c, "m": m,
         "note": "H0 at lower search bound" if abs(h0 + 10) < 1e-3 else ""}, indent=2))

    print(f"Data: {data} ({len(df)} pairs; train {tr.sum()}, test {te.sum()})")
    print(f"Least squares: H = {h0:.3f} + {c:.4f} * Q^{m:.4f}"
          + ("   [H0 at the -10 m search bound]" if abs(h0 + 10) < 1e-3 else ""))
    print(f"XGBoost: {xgb.best_iteration + 1} trees (early stopping)")
    print(t6.round(3).to_string(index=False))
    print(f"XGBoost maximum predictable stage: {float(xgb.predict(pd.DataFrame({'Q': [1e5]}))[0]):.2f} m "
          f"(tree models cannot extrapolate beyond the training range)")

    if make_plots:
        plot(df, tr, te, ls_tr, ls_te, xg_tr, xg_te, data)
    return t6


def plot(df, tr, te, ls_tr, ls_te, xg_tr, xg_te, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for j, (mask, ls, xg, label) in enumerate(((tr, ls_tr, xg_tr, "Training"), (te, ls_te, xg_te, "Testing"))):
        obs = df.loc[mask, "H"].values
        a = ax[0, j]
        a.scatter(obs, xg, s=8, alpha=0.5, label="XGBoost")
        a.scatter(obs, ls, s=8, alpha=0.5, label="Least squares")
        lim = [obs.min() - 0.5, obs.max() + 0.5]
        a.plot(lim, lim, "k--", lw=1)
        a.set(xlabel="Observed stage (m)", ylabel="Predicted stage (m)", title=f"({'ab'[j]}) {label}")
        a.legend()
        r = ax[1, j]
        r.scatter(obs, xg - obs, s=8, alpha=0.5, label="XGBoost")
        r.scatter(obs, ls - obs, s=8, alpha=0.5, label="Least squares")
        r.axhline(0, color="k", lw=1)
        r.set(xlabel="Observed stage (m)", ylabel="Residual (m)", title=f"({'cd'[j]}) {label} residuals")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"Fig8_rating_curves_{data}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["paper", "observed"], default="paper")
    a = ap.parse_args()
    if a.data == "observed" and not OBS_PAIRS_FILE.exists():
        print(f"[SKIP] {OBS_PAIRS_FILE} not found (restricted CWC data)")
    elif a.data == "paper" and not PAIRED_FILE["Ssp245"].exists():
        print(f"[SKIP] {PAIRED_FILE['Ssp245']} not found (restricted CWC data)")
    else:
        main(a.data)

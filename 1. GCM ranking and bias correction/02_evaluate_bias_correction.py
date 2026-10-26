# -*- coding: utf-8 -*-
"""
Step 2 - Out-of-sample skill of the XGBoost bias correction (Table 4).

For every top-5 GCM of every variable:
  1. Predictors: GCM value (t), GCM value (t-1), day of year.
     Target    : bias = Observed - GCM.
  2. Hyper-parameters tuned with RandomizedSearchCV (30 draws, 3-fold CV)
     on the TRAINING period only (1986-2006).
  3. The tuned model is evaluated on the held-out TESTING period (2007-2014):
     RMSE, MAE and R2 before (raw GCM) and after (corrected) correction.
  4. The RS-weighted ensemble of the five raw and five corrected series is
     evaluated on the same held-out period.

The hold-out models are written to Models_holdout/ and are used ONLY for this
skill assessment. The operational models in Models/ (used to correct the SSP
projections) are not modified.

Input  : Historical/<VAR>_Areal_<GCM>_historical_1986_2014.csv (Date, Observed, Raw_Modeled)
         Results/Top5_GCMs_and_Weights.csv          (from 01_rank_gcms.py)
Output : Results/Table4_BiasCorrection_Holdout.csv
         Results/Holdout_BestParams.csv
         Models_holdout/<VAR>_Areal_<GCM>_Tuned_XGBoost.pkl

Run    : python 02_evaluate_bias_correction.py            (all variables)
         python 02_evaluate_bias_correction.py --var Pr   (one variable)
"""

import argparse
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from common import (BASE_DIR, CV_FOLDS, EVAL_MODEL_DIR, FEATURES, N_ITER, PARAM_DIST,
                    RANDOM_STATE, RESULTS_DIR, TEST_END, TEST_START, TRAIN_END,
                    TRAIN_START, VARIABLES, historical_file, make_features, metrics,
                    model_file, top_models)

warnings.filterwarnings("ignore", category=UserWarning)


def load_historical(var: str, gcm: str) -> pd.DataFrame:
    path = historical_file(var, gcm)
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[["Observed", "Raw_Modeled"]].apply(pd.to_numeric, errors="coerce")
    # Re-index to a continuous daily calendar so the lag-1 predictor is always
    # the previous calendar day, even where observations are missing.
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full)


def evaluate_gcm(var: str, gcm: str):
    nonneg = VARIABLES[var]["nonneg"]
    hist = load_historical(var, gcm)
    X = make_features(hist["Raw_Modeled"])
    data = X.join(hist["Observed"]).dropna()
    data["Bias"] = data["Observed"] - data["Modeled"]

    train = data.loc[TRAIN_START:TRAIN_END]
    test = data.loc[TEST_START:TEST_END]

    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=PARAM_DIST, n_iter=N_ITER, cv=CV_FOLDS,
        scoring="neg_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    search.fit(train[FEATURES], train["Bias"])
    model = search.best_estimator_

    EVAL_MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, model_file(var, gcm, EVAL_MODEL_DIR))

    corrected = test["Modeled"] + model.predict(test[FEATURES])
    if nonneg:
        corrected = corrected.clip(lower=0.0)

    series = pd.DataFrame({"Observed": test["Observed"], "Raw": test["Modeled"],
                           "Corrected": corrected})
    return series, search.best_params_, len(train), len(test)


def main(selected=None):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, params_rows = [], []
    for var in (selected or VARIABLES):
        top = top_models(var)
        print(f"\n=== {var} ({VARIABLES[var]['unit']}) - train {TRAIN_START[:4]}-{TRAIN_END[:4]}, "
              f"test {TEST_START[:4]}-{TEST_END[:4]} ===")
        ens_raw, ens_cor, obs = 0.0, 0.0, None
        for _, r in top.iterrows():
            gcm, w = r["Model"], r["Weight"]
            s, best, n_tr, n_te = evaluate_gcm(var, gcm)
            b, a = metrics(s["Observed"], s["Raw"]), metrics(s["Observed"], s["Corrected"])
            rows.append({"Variable": var, "Model": gcm, "RS": r["RS"], "Weight": w,
                         "RMSE_before": b["RMSE"], "RMSE_after": a["RMSE"],
                         "MAE_before": b["MAE"], "MAE_after": a["MAE"],
                         "R2_before": b["R2"], "R2_after": a["R2"],
                         "N_train": n_tr, "N_test": n_te})
            params_rows.append({"Variable": var, "Model": gcm, **best})
            print(f"  {gcm:<18} RMSE {b['RMSE']:7.3f} -> {a['RMSE']:7.3f} | "
                  f"MAE {b['MAE']:6.3f} -> {a['MAE']:6.3f} | R2 {b['R2']:7.3f} -> {a['R2']:6.3f}")
            ens_raw = ens_raw + w * s["Raw"]
            ens_cor = ens_cor + w * s["Corrected"]
            obs = s["Observed"] if obs is None else obs.combine_first(s["Observed"])

        # Ensemble rows: (a) metrics of the weighted ensemble SERIES,
        #                (b) mean of the individual model metrics.
        idx = ens_raw.dropna().index.intersection(ens_cor.dropna().index)
        b, a = metrics(obs.loc[idx], ens_raw.loc[idx]), metrics(obs.loc[idx], ens_cor.loc[idx])
        rows.append({"Variable": var, "Model": "Weighted ensemble series",
                     "RMSE_before": b["RMSE"], "RMSE_after": a["RMSE"],
                     "MAE_before": b["MAE"], "MAE_after": a["MAE"],
                     "R2_before": b["R2"], "R2_after": a["R2"], "N_test": b["N"]})
        sub = pd.DataFrame([x for x in rows if x["Variable"] == var and "N_train" in x])
        mean_row = {"Variable": var, "Model": "Mean of 5 models"}
        for c in ["RMSE_before", "RMSE_after", "MAE_before", "MAE_after", "R2_before", "R2_after"]:
            mean_row[c] = sub[c].mean()
        rows.append(mean_row)
        print(f"  {'Weighted ensemble':<18} RMSE {b['RMSE']:7.3f} -> {a['RMSE']:7.3f} | "
              f"MAE {b['MAE']:6.3f} -> {a['MAE']:6.3f} | R2 {b['R2']:7.3f} -> {a['R2']:6.3f}")

    out = RESULTS_DIR / "Table4_BiasCorrection_Holdout.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    pd.DataFrame(params_rows).to_csv(RESULTS_DIR / "Holdout_BestParams.csv", index=False)
    print(f"\nTable 4 (held-out) -> {out.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=list(VARIABLES), nargs="*", help="variables to run")
    main(ap.parse_args().var)

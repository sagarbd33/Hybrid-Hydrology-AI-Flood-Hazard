# -*- coding: utf-8 -*-
"""
Step 3 - Apply the operational XGBoost bias-correction models to the CMIP6
SSP245 / SSP585 projections (replaces the four BiasCorrectionForAreal*.py
scripts with one script).

For each variable, each top-5 GCM and each scenario:
    corrected(t) = GCM(t) + bias_model[GCM(t), GCM(t-1), day-of-year]
Precipitation and radiation are clipped at zero.

Input  : Raw_Projections/<Var>/<PREFIX>_Areal_<ssp>_<GCM>.csv  raw areal-mean series (GEE)
         Models/<PREFIX>_Areal_<GCM>_Tuned_XGBoost.pkl
         Results/Top5_GCMs_and_Weights.csv
Output : Future_BiasCorrected/<PREFIX>_Areal_<ssp>_<GCM>_BiasCorrected.csv
         (columns: Date, <value column>, Raw)
         Results/Future_correction_summary.csv  (mean raw vs corrected; also
         compared with the corrected files used in the paper, DataGRU/)

Run    : python 03_apply_bias_correction.py
"""

import warnings

import joblib
import pandas as pd

from common import (BASE_DIR, DATAGRU_DIR, FUTURE_OUT_DIR, RESULTS_DIR, SCENARIOS, VARIABLES,
                    apply_correction, historical_file, model_file, raw_future_file,
                    read_gee_series, top_models)

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    FUTURE_OUT_DIR.mkdir(exist_ok=True)
    summary = []
    for var, v in VARIABLES.items():
        print(f"\n=== {var} ===")
        for gcm in top_models(var)["Model"]:
            model = joblib.load(model_file(var, gcm))
            obs_mean = pd.read_csv(historical_file(var, gcm))["Observed"].mean()
            for ssp in SCENARIOS:
                src = raw_future_file(var, ssp, gcm)
                if not src.exists():
                    print(f"  [MISSING] {src.relative_to(BASE_DIR)} - not available in "
                          f"NEX-GDDP-CMIP6; ensemble weights are renormalised in step 4.")
                    continue
                raw = read_gee_series(src)
                cor = apply_correction(model, raw, v["nonneg"])
                out = pd.DataFrame({"Date": cor.index.strftime("%Y-%m-%d"),
                                    raw.name: cor.values,
                                    "Raw": raw.loc[cor.index].values})
                dst = FUTURE_OUT_DIR / f"{v['prefix']}_Areal_{ssp}_{gcm}_BiasCorrected.csv"
                out.to_csv(dst, index=False)
                row = {"Variable": var, "Scenario": ssp, "Model": gcm,
                       "Obs_mean_1986_2014": obs_mean,
                       "Raw_mean": raw.mean(), "Corrected_mean": cor.mean(),
                       "Start": cor.index.min().date(), "End": cor.index.max().date()}
                # Comparison with the bias-corrected files used in the paper (DataGRU/)
                old = DATAGRU_DIR / dst.name
                if old.exists():
                    o = read_gee_series(old).reindex(cor.index)
                    row["DataGRU_mean"] = o.mean()
                    row["Max_abs_diff_vs_DataGRU"] = (o - cor).abs().max()
                summary.append(row)
                print(f"  {ssp} {gcm:<18} raw mean {raw.mean():8.3f} -> corrected "
                      f"{cor.mean():8.3f} (obs 1986-2014: {obs_mean:.3f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(summary).round(4).to_csv(RESULTS_DIR / "Future_correction_summary.csv", index=False)
    print(f"\nCorrected series -> {FUTURE_OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

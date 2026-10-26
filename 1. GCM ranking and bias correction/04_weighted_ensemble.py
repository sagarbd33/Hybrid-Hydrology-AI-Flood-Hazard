# -*- coding: utf-8 -*-
"""
Step 4 - RS-weighted multi-model ensemble of the bias-corrected projections.

    Ensemble(t) = sum_k w_k * Corrected_k(t),   w_k = RS_k / sum(RS)

If a GCM has no run for a scenario (e.g. HadGEM3-GC31-MM provides no SSP245
rsds in NEX-GDDP-CMIP6), its weight is dropped and the remaining weights are
renormalised to sum to 1. The weights actually used are written to file.

Input  : Future_BiasCorrected/*.csv  (from 03_apply_bias_correction.py)
         Results/Top5_GCMs_and_Weights.csv
Output : Ensemble/Ensemble_<ssp>.csv  columns Date, pr, rsds, tasmax, tasmin
         Results/Ensemble_weights_used.csv

Run    : python 04_weighted_ensemble.py
"""

import pandas as pd

from common import BASE_DIR, ENSEMBLE_DIR, FUTURE_OUT_DIR, RESULTS_DIR, SCENARIOS, VARIABLES, top_models

OUT_NAMES = {"Pr": "pr", "Srad": "rsds", "Tmax": "tasmax", "Tmin": "tasmin"}


def main():
    ENSEMBLE_DIR.mkdir(exist_ok=True)
    used = []
    for ssp in SCENARIOS:
        cols = {}
        for var, v in VARIABLES.items():
            top = top_models(var)
            series, weights, present = [], [], []
            for _, r in top.iterrows():
                f = FUTURE_OUT_DIR / f"{v['prefix']}_Areal_{ssp}_{r['Model']}_BiasCorrected.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
                series.append(d.drop(columns="Raw").iloc[:, 0])
                weights.append(r["Weight"])
                present.append(r["Model"])
            if not series:
                raise FileNotFoundError(f"No corrected files for {var} {ssp}; run step 3.")
            wsum = sum(weights)
            frame = pd.concat(series, axis=1).dropna()
            ens = sum(frame.iloc[:, i] * (w / wsum) for i, w in enumerate(weights))
            cols[OUT_NAMES[var]] = ens
            for gcm, w in zip(present, weights):
                used.append({"Scenario": ssp, "Variable": var, "Model": gcm,
                             "Weight_paper": w, "Weight_used": w / wsum})
            note = "" if len(series) == len(top) else f"  ({len(series)} of {len(top)} GCMs, weights renormalised)"
            print(f"{ssp} {var:<5} ensemble of {len(series)} GCMs{note}")

        out = pd.DataFrame(cols).dropna()
        out.index.name = "Date"
        dst = ENSEMBLE_DIR / f"Ensemble_{ssp}.csv"
        out.round(4).to_csv(dst)
        print(f"  -> {dst.relative_to(BASE_DIR)}  ({out.index.min().date()} to {out.index.max().date()})")

    pd.DataFrame(used).round(6).to_csv(RESULTS_DIR / "Ensemble_weights_used.csv", index=False)


if __name__ == "__main__":
    main()

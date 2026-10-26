# -*- coding: utf-8 -*-
"""
Quick test for Module 1 (runs in a few seconds).

1. Ranks the GCMs (Eq. 1) and checks that the top-5 GCMs per variable match the
   models used in the paper.
2. Bias-corrects a short sample (2015-2016, SSP585) of one GCM per variable
   with the saved XGBoost models and compares the result with the expected
   output in quick_test/expected/.

Run:  python quick_test/run_quick_test.py
Pass: prints "QUICK TEST PASSED".
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore", category=UserWarning)

from common import PERF_DIR, RS_WEIGHTS, VARIABLES, apply_correction, model_file, read_gee_series  # noqa: E402

EXPECTED_TOP5 = {
    "Pr": ["KACE-1-0-G", "EC-Earth3-Veg-LR", "INM-CM5-0", "HadGEM3-GC31-LL", "CNRM-CM6-1"],
    "Srad": ["CNRM-CM6-1", "MIROC-ES2L", "HadGEM3-GC31-MM", "EC-Earth3-Veg-LR", "MPI-ESM1-2-HR"],
    "Tmax": ["FGOALS-g3", "NESM3", "NorESM2-LM", "GFDL-CM4", "CMCC-CM2-SR5"],
    "Tmin": ["INM-CM4-8", "EC-Earth3-Veg-LR", "KACE-1-0-G", "UKESM1-0-LL", "MIROC-ES2L"],
}
SAMPLE_GCM = {"Pr": "KACE-1-0-G", "Srad": "MIROC-ES2L", "Tmax": "NESM3", "Tmin": "INM-CM4-8"}
TOL = 1e-4


def main():
    ok = True

    print("1) GCM ranking (Eq. 1)")
    w1, w2, w3 = RS_WEIGHTS
    for var, expected in EXPECTED_TOP5.items():
        d = pd.read_csv(PERF_DIR / VARIABLES[var]["perf"])
        d["RS"] = w1 * d.RMSE.min() / d.RMSE + w2 * d.MAE.min() / d.MAE + w3 * d.R2
        top = d.sort_values("RS", ascending=False).head(5)["Model"].tolist()
        good = top == expected
        ok &= good
        print(f"   {var:<5} {'OK  ' if good else 'FAIL'} {top}")

    print("2) Bias correction of sample input (2015-2016, SSP585)")
    for var, gcm in SAMPLE_GCM.items():
        prefix = VARIABLES[var]["prefix"]
        raw = read_gee_series(HERE / "input" / f"{prefix}_Areal_ssp585_{gcm}_2015_2016.csv")
        model = joblib.load(model_file(var, gcm))
        got = apply_correction(model, raw, VARIABLES[var]["nonneg"])
        exp = read_gee_series(HERE / "expected" / f"{prefix}_Areal_ssp585_{gcm}_2015_2016_BiasCorrected.csv")
        diff = float(np.max(np.abs(got.values - exp.reindex(got.index).values)))
        good = diff < TOL and len(got) == len(exp)
        ok &= good
        print(f"   {var:<5} {gcm:<12} {'OK  ' if good else 'FAIL'} max |difference| = {diff:.2e}")

    print("\nQUICK TEST PASSED" if ok else "\nQUICK TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

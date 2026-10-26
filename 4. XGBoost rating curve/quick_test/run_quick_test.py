# -*- coding: utf-8 -*-
"""
Quick test for Module 4 (a few seconds, no restricted data needed).

1. Power-law fitting: recovers H0, c, m from a synthetic rating curve
   H = 10 + 0.2 Q^0.7 (non-linear least squares).
2. XGBoost rating curve (Table 6 model): predictions on a discharge grid
   match quick_test/expected/.
3. Stage projection (Section 4.4 model): stage for a 2015-2016 SSP585
   discharge sample matches quick_test/expected/.
4. Statistics on a hand-computable example.

Run:  python quick_test/run_quick_test.py
Pass: prints "QUICK TEST PASSED".
"""

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor  # noqa: E402

from common import BIAS_FEATURES, MODEL_DIR, bias_features, read_csv, scores  # noqa: E402

rc = importlib.import_module("01_rating_curves")


def check(name, ok, detail=""):
    print(f"   {'OK  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def main():
    ok = True
    print("1) Power-law rating curve fit")
    Q = np.linspace(2, 250, 200)
    h0, c, m = rc.fit_power_law_nonlinear(Q, 10 + 0.2 * Q ** 0.7)
    ok &= check("H0, c, m recovered", abs(h0 - 10) < 1e-3 and abs(c - 0.2) < 1e-3 and abs(m - 0.7) < 1e-3,
                f"(H0 {h0:.3f}, c {c:.3f}, m {m:.3f})")

    print("2) XGBoost rating curve (Table 6)")
    exp = pd.read_csv(HERE / "expected" / "RatingCurve_XGB_paper_grid.csv")
    model = XGBRegressor()
    model.load_model(MODEL_DIR / "RatingCurve_XGB_paper.json")
    d = float(np.max(np.abs(model.predict(exp[["Q"]]) - exp["Stage"].values)))
    ok &= check("stage on discharge grid", d < 1e-4, f"(max |diff| {d:.1e} m)")

    print("3) Stage projection (2015-2016, SSP585)")
    q = read_csv(HERE / "input" / "Ssp585_OptiDischarge_2015_2016.csv")["OptiDischarge"]
    sm = XGBRegressor()
    sm.load_model(MODEL_DIR / "StageModel_XGB_Ssp585.json")
    X = bias_features(q).dropna()
    stage = X["Modeled"] + sm.predict(X[BIAS_FEATURES])
    exp = read_csv(HERE / "expected" / "Ssp585_Stage_2015_2016.csv")["Stage"]
    d = float(np.max(np.abs(stage.values - exp.values)))
    ok &= check("daily stage", len(stage) == len(exp) and d < 1e-3, f"({len(stage)} days, max |diff| {d:.1e} m)")

    print("4) Statistics")
    s = scores([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0])
    ok &= check("NSE/RMSE/MAE", abs(s["NSE"] - 0.8) < 1e-9 and abs(s["RMSE_m"] - 0.5) < 1e-9
                and abs(s["MAE_m"] - 0.25) < 1e-9)

    print("\nQUICK TEST PASSED" if ok else "\nQUICK TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

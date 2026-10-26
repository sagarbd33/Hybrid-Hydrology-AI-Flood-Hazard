# -*- coding: utf-8 -*-
"""
Quick test for Module 3 (about 10 s, no restricted data needed).

1. Loads the trained hybrid HEC-HMS-GRU model and checks its architecture
   against Table 3 (GRU 256-128-16, dropout 0.2, 15-day window, 5 inputs).
2. Runs the model on a 2015-2016 SSP585 sample (quick_test/input/) and compares
   the simulated discharge with the values used in the paper
   (quick_test/expected/, tolerance 0.01 m3/s).
3. Runs the standalone GRU on a Jan-Mar 2021 forcing sample and compares with
   quick_test/expected/.
4. Checks the performance statistics (NSE, RMSE, MAE, PBIAS, R2) on a small
   hand-computable example.

Run:  python quick_test/run_quick_test.py
Pass: prints "QUICK TEST PASSED".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (DROPOUT, FEATURES, GRU_UNITS, N_STEPS, STANDALONE_MODEL,  # noqa: E402
                    STANDALONE_SCALERS, load_gru, load_scalers, make_sequences, predict, read_csv,
                    scores)


def check(name, ok, detail=""):
    print(f"   {'OK  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def main():
    ok = True
    print("1) Model architecture")
    model, scalers = load_gru(), load_scalers()
    grus = [l for l in model.layers if l.__class__.__name__ == "GRU"]
    drops = [l for l in model.layers if l.__class__.__name__ == "Dropout"]
    ok &= check("GRU units", tuple(l.units for l in grus) == GRU_UNITS, str([l.units for l in grus]))
    ok &= check("dropout", all(abs(l.rate - DROPOUT) < 1e-9 for l in drops), f"{[l.rate for l in drops]}")
    ok &= check("input shape", tuple(model.input_shape[1:]) == (N_STEPS, len(FEATURES["hybrid"])),
                str(model.input_shape))

    print("2) Projection on 2015-2016 SSP585 sample")
    df = read_csv(HERE / "input" / "Ssp585_EnsembleMean_2015_2016.csv")
    X, _, dates = make_sequences(df, FEATURES["hybrid"])
    q = pd.Series(predict(model, scalers, X), index=dates)
    exp = read_csv(HERE / "expected" / "Ssp585_OptiDischarge_2015_2016.csv")["OptiDischarge"]
    d = float(np.max(np.abs(q.reindex(exp.index).values - exp.values)))
    ok &= check("discharge = paper", len(q) == len(exp) and d < 0.01,
                f"({len(q)} days, max |diff| {d:.1e} m3/s)")

    print("3) Standalone GRU on a Jan-Mar 2021 forcing sample")
    sm = load_gru(STANDALONE_MODEL)
    ok &= check("standalone architecture", tuple(sm.input_shape[1:]) == (N_STEPS, len(FEATURES["standalone"])),
                str(sm.input_shape))
    df = read_csv(HERE / "input" / "Standalone_forcing_2021Q1.csv")
    X, _, dates = make_sequences(df, FEATURES["standalone"])
    q = predict(sm, load_scalers(STANDALONE_SCALERS), X)
    exp = read_csv(HERE / "expected" / "Standalone_prediction_2021Q1.csv")["Predicted"]
    d = float(np.max(np.abs(q - exp.values)))
    ok &= check("prediction = expected", len(q) == len(exp) and d < 0.01, f"(max |diff| {d:.1e} m3/s)")

    print("4) Statistics")
    s = scores([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0])
    good = (abs(s["NSE"] - 0.8) < 1e-9 and abs(s["RMSE"] - 0.5) < 1e-9 and abs(s["MAE"] - 0.25) < 1e-9
            and abs(s["PBIAS_%"] - 10.0) < 1e-9)
    ok &= check("NSE/RMSE/MAE/PBIAS", good, f"(NSE {s['NSE']:.2f}, RMSE {s['RMSE']:.2f}, PBIAS {s['PBIAS_%']:.0f} %)")

    print("\nQUICK TEST PASSED" if ok else "\nQUICK TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

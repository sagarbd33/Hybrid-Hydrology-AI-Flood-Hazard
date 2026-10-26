# -*- coding: utf-8 -*-
"""
Step 5 (optional) - Sensitivity of the GRU results to random initialisation.

Trains the hybrid and the standalone GRU with several random seeds (same data,
split and architecture) and reports the spread of the validation-2021 scores.
A single training run of a deep network depends on its random initialisation;
the spread shows how robust the hybrid-vs-standalone comparison of Table 5 is.

Needs the restricted training/validation files (CWC discharge).
Output : Results/Seed_sensitivity.csv, Results/Seed_sensitivity_summary.csv
Run    : python 05_seed_sensitivity.py --seeds 1 2 3 4 5 [--protocol clean|paper]
         (about 1-3 min per model; default protocol: clean, see 02_train_gru.py)
"""

import argparse
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (DATA_FILES, FEATURES, RESULTS_DIR, SPLIT_SEED, TARGET, TEST_SIZE, load_gru,
                    load_scalers, make_sequences, predict, read_csv, scores)

train_mod = importlib.import_module("02_train_gru")


def evaluate(kind, model_path, scaler_path):
    model, scalers = load_gru(model_path), load_scalers(scaler_path)
    feats = FEATURES[kind]
    train_file, valid_file = DATA_FILES[kind]
    X, y, _ = make_sequences(read_csv(train_file), feats, TARGET, drop_last=True)
    _, te = train_test_split(range(len(y)), test_size=TEST_SIZE, random_state=SPLIT_SEED)
    p = predict(model, scalers, X)
    Xv, yv, _ = make_sequences(read_csv(valid_file), feats, TARGET, drop_last=True)
    pv = predict(model, scalers, Xv)
    return scores(y[te], p[te]), scores(yv, pv)


def main(seeds, protocol="clean"):
    if not DATA_FILES["hybrid"][0].exists():
        print(f"[SKIP] restricted training tables not found in {DATA_FILES['hybrid'][0].parent}")
        return
    rows = []
    for kind in ("hybrid", "standalone"):
        for s in seeds:
            mp, sp = train_mod.main(kind, seed=s, protocol=protocol, tag=f"_seed{s}", verbose=0)
            te, va = evaluate(kind, mp, sp)
            rows.append({"Model": kind, "Protocol": protocol, "Seed": s,
                         "Test_NSE": te["NSE"], "Test_RMSE": te["RMSE"],
                         "Val2021_NSE": va["NSE"], "Val2021_RMSE": va["RMSE"],
                         "Val2021_Top5pct_bias_%": va["Top5pct_bias_%"]})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.round(4).to_csv(RESULTS_DIR / "Seed_sensitivity.csv", index=False)
    summ = df.drop(columns=["Seed", "Protocol"]).groupby("Model").agg(["mean", "std", "min", "max"]).round(3)
    summ.to_csv(RESULTS_DIR / "Seed_sensitivity_summary.csv")
    print("\n", summ.T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--protocol", choices=["paper", "clean"], default="clean")
    a = ap.parse_args()
    main(a.seeds, a.protocol)

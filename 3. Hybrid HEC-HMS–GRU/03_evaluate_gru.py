# -*- coding: utf-8 -*-
"""
Step 3 - Performance of the hybrid HEC-HMS-GRU model, the standalone GRU and
standalone HEC-HMS against observed discharge at Bairabi (Table 5, Section 4.2).

Periods (as in the paper):
  * Training / testing: 2018-2020 sequences split 70/30 at random
    (sklearn train_test_split, random_state=42)
  * Independent validation: 2021

The models in Models/ are the trained models used in the paper; this script
reproduces the published statistics for both. A retrained model can be
evaluated with --kind / --model / --scalers (see 02_train_gru.py).

Needs the restricted tables in Data/restricted/ (CWC discharge; not
distributed). For --kind standalone the standalone tables are used.

Output : Results/Table5_<kind>_GRU.csv, Results/Table5_HEC-HMS.csv
         Results/Predictions_<kind>_{Train,Test,Validation2021}.csv

Run    : python 03_evaluate_gru.py                 (both models used in the paper)
         python 03_evaluate_gru.py --kind standalone --model <file> --scalers <file>
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (DATA_FILES, FEATURES, HYBRID_MODEL, HYBRID_SCALERS, PAPER_MODELS, RESULTS_DIR,
                    SPLIT_SEED,
                    TARGET, TEST_SIZE, load_gru, load_scalers, make_sequences, predict, read_csv,
                    scores)


def main(model_path=HYBRID_MODEL, scaler_path=HYBRID_SCALERS, kind="hybrid"):
    TRAIN_FILE, VALID_FILE = DATA_FILES[kind]
    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        print(f"[SKIP] restricted observed-discharge files not found in {TRAIN_FILE.parent}")
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    feats = FEATURES[kind]
    model, scalers = load_gru(model_path), load_scalers(scaler_path)

    train = read_csv(TRAIN_FILE)
    X, y, dates = make_sequences(train, feats, TARGET, drop_last=True)
    idx_tr, idx_te = train_test_split(range(len(y)), test_size=TEST_SIZE, random_state=SPLIT_SEED)
    p = predict(model, scalers, X)

    valid = read_csv(VALID_FILE)
    Xv, yv, dv = make_sequences(valid, feats, TARGET, drop_last=True)
    pv = predict(model, scalers, Xv)

    rows = [{"Period": "Training (70%, 2018-2020)", **scores(y[idx_tr], p[idx_tr])},
            {"Period": "Testing (30%, 2018-2020)", **scores(y[idx_te], p[idx_te])},
            {"Period": "Validation 2021", **scores(yv, pv)}]
    t5 = pd.DataFrame(rows)
    t5.round(4).to_csv(RESULTS_DIR / f"Table5_{kind}_GRU.csv", index=False)
    print(f"{'Hybrid HEC-HMS-GRU' if kind == 'hybrid' else 'Standalone GRU'} ({Path(model_path).name}):")
    print(t5.round(3).to_string(index=False))

    pd.DataFrame({"Date": dates[idx_tr], "Observed": y[idx_tr], "Predicted": p[idx_tr]}).to_csv(
        RESULTS_DIR / f"Predictions_{kind}_Train.csv", index=False)
    pd.DataFrame({"Date": dates[idx_te], "Observed": y[idx_te], "Predicted": p[idx_te]}).to_csv(
        RESULTS_DIR / f"Predictions_{kind}_Test.csv", index=False)
    pd.DataFrame({"Date": dv, "Observed": yv, "Predicted": pv}).to_csv(
        RESULTS_DIR / f"Predictions_{kind}_Validation2021.csv", index=False)
    if kind != "hybrid":
        return

    # Standalone HEC-HMS on the same days
    hms = [{"Period": "Calibration 2018-2020", **scores(train[TARGET], train["simdischarge"])},
           {"Period": "Validation 2021", **scores(valid[TARGET], valid["simdischarge"])},
           {"Period": "Validation 2021 (same days as GRU)",
            **scores(yv, valid["simdischarge"].loc[dv].values)}]
    hms = pd.DataFrame(hms)
    hms.round(4).to_csv(RESULTS_DIR / "Table5_HEC-HMS.csv", index=False)
    print("\nStandalone HEC-HMS:")
    print(hms.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--scalers")
    ap.add_argument("--kind", choices=list(FEATURES))
    a = ap.parse_args()
    if a.model:
        main(Path(a.model), Path(a.scalers), a.kind or "hybrid")
    else:
        for kind, (m, sc) in PAPER_MODELS.items():
            if a.kind in (None, kind):
                main(m, sc, kind)
                print()

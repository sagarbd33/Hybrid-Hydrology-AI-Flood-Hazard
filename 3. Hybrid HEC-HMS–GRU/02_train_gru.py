# -*- coding: utf-8 -*-
"""
Step 2 - Train the GRU models (Section 3.1, Table 3).

  hybrid     : inputs precip, tmin, tmax, srad + HEC-HMS simulated discharge
  standalone : inputs precip, tmin, tmax, srad only (baseline in Table 5)

Same code for both models; only the input table differs (common.DATA_FILES).
This is a cleaned version of the original Colab notebook
(docs/original_gru_training_colab.py), with identical architecture:
  GRU 256 (ReLU, L2 0.01, return sequences) - Dropout 0.2
  GRU 128 (tanh, return sequences)          - Dropout 0.2
  GRU 16  (tanh)                            - Dropout 0.2
  Dense 1 (swish = SiLU)
  Nadam (lr 0.001), MSE loss, batch 24, up to 300 epochs, early stopping
  (patience 20, active from epoch 50, best weights restored).
Data: 15-day windows from 2018-2020, MinMax scaling (0-1), 70/30 random split
(train_test_split, random_state=42).

Two protocols:
  --protocol paper  (default) exactly as in the original notebook:
        scalers fitted on ALL sequences; the 30 % test set is also the
        validation set that drives early stopping.
  --protocol clean  scalers fitted on the training sequences only; early
        stopping monitors 20 % of the training sequences, so the test set
        is never seen during training.

The trained model used in the paper is in Models/. Retraining gives slightly
different weights (random initialisation, hardware); new models go to
Models_retrained/ and never overwrite Models/.

Needs the restricted tables in Data/restricted/ (CWC discharge; not distributed).

Run:  python 02_train_gru.py                                   (hybrid, paper protocol)
      python 02_train_gru.py --kind standalone
      python 02_train_gru.py --protocol clean
      python 02_train_gru.py --epochs 5                        (quick check)
"""

import argparse
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from common import (BATCH_SIZE, DATA_FILES, DROPOUT, FEATURES, GRU_UNITS, L2_FIRST, LEARNING_RATE,
                    MAX_EPOCHS, N_STEPS, PATIENCE, RETRAIN_DIR, SEED, SPLIT_SEED, START_FROM_EPOCH,
                    TARGET, TEST_SIZE, make_sequences, read_csv, scores)

VAL_FRACTION = 0.2   # clean protocol: part of the training sequences used for early stopping


def build_model(n_features):
    import keras
    from keras import layers, regularizers
    model = keras.Sequential([
        keras.Input(shape=(N_STEPS, n_features)),
        layers.GRU(GRU_UNITS[0], activation="relu", return_sequences=True,
                   kernel_regularizer=regularizers.l2(L2_FIRST)),
        layers.Dropout(DROPOUT),
        layers.GRU(GRU_UNITS[1], return_sequences=True),
        layers.Dropout(DROPOUT),
        layers.GRU(GRU_UNITS[2]),
        layers.Dropout(DROPOUT),
        layers.Dense(1, activation="silu"),
    ])
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def main(kind="hybrid", epochs=MAX_EPOCHS, seed=SEED, protocol="paper", tag="", verbose=2):
    train_file = DATA_FILES[kind][0]
    if not train_file.exists():
        print(f"[SKIP] {train_file} not found (restricted CWC discharge)")
        return None
    import keras
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    feats = FEATURES[kind]
    X, y, _ = make_sequences(read_csv(train_file), feats, TARGET, drop_last=True)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=SPLIT_SEED)

    fit_idx = np.arange(len(y)) if protocol == "paper" else idx_tr
    sx = MinMaxScaler().fit(X[fit_idx].reshape(-1, len(feats)))
    sy = MinMaxScaler().fit(y[fit_idx].reshape(-1, 1))
    Xs = sx.transform(X.reshape(-1, len(feats))).reshape(X.shape)
    ys = sy.transform(y.reshape(-1, 1)).ravel()

    model = build_model(len(feats))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE,
                                         restore_best_weights=True, start_from_epoch=START_FROM_EPOCH)
    if protocol == "paper":
        val = dict(validation_data=(Xs[idx_te], ys[idx_te]))
    else:
        val = dict(validation_split=VAL_FRACTION)
    hist = model.fit(Xs[idx_tr], ys[idx_tr], epochs=epochs, batch_size=BATCH_SIZE,
                     callbacks=[stop], verbose=verbose, **val)

    pred = sy.inverse_transform(model.predict(Xs, verbose=0)).ravel()
    tr, te = scores(y[idx_tr], pred[idx_tr]), scores(y[idx_te], pred[idx_te])
    print(f"\n{kind} ({protocol} protocol, seed {seed}): {len(hist.history['loss'])} epochs | "
          f"train NSE {tr['NSE']:.3f} RMSE {tr['RMSE']:.2f} | test NSE {te['NSE']:.3f} RMSE {te['RMSE']:.2f}")

    RETRAIN_DIR.mkdir(exist_ok=True)
    name = f"{kind.capitalize()}_GRU_{protocol}{tag}"
    model.save(RETRAIN_DIR / f"{name}_model.keras")
    with open(RETRAIN_DIR / f"{name}_scalers.pkl", "wb") as f:
        pickle.dump((sx, sy), f)
    print(f"Saved -> {RETRAIN_DIR.name}/{name}_model.keras (+ _scalers.pkl)")
    print(f"Evaluate: python 03_evaluate_gru.py --kind {kind} "
          f"--model {RETRAIN_DIR.name}/{name}_model.keras --scalers {RETRAIN_DIR.name}/{name}_scalers.pkl")
    return RETRAIN_DIR / f"{name}_model.keras", RETRAIN_DIR / f"{name}_scalers.pkl"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(FEATURES), default="hybrid")
    ap.add_argument("--protocol", choices=["paper", "clean"], default="paper")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    main(a.kind, a.epochs, a.seed, a.protocol, tag="" if a.seed == SEED else f"_seed{a.seed}")

# -*- coding: utf-8 -*-
"""
Shared settings for Module 3 (hybrid HEC-HMS-GRU streamflow model).

All paths are relative to this folder, so the scripts run unchanged from
    C:\\Users\\AE-lab\\Downloads\\Paper 6\\3. Hybrid HEC-HMS-GRU
or from a cloned repository.

Manuscript: "A Hybrid Hydrology-AI Framework with Terrain-Based Inundation
Modeling for Probabilistic Flood Hazard-Exposure Assessment under CMIP6
Scenarios" (Debbarma, Bandyopadhyay, Bhadra), Sections 3.1 and 4.2.
"""

import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TensorFlow

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
INPUT_DIR = DATA_DIR / "Inputs"            # GRU inputs for the climate runs (used in the paper)
RESTRICTED_DIR = DATA_DIR / "restricted"   # contains CWC observed discharge - NOT for distribution
MODEL_DIR = BASE_DIR / "Models"            # trained hybrid model used in the paper
RETRAIN_DIR = BASE_DIR / "Models_retrained"
RESULTS_DIR = BASE_DIR / "Results"

# Sibling modules (for 01_build_gru_inputs.py)
MODULE1_DIR = BASE_DIR.parent / "1. GCM ranking and bias correction"
MODULE2_DIR = BASE_DIR.parent / "2. HEC-HMS"

HYBRID_MODEL = MODEL_DIR / "Hybrid_GRU_model.keras"
HYBRID_SCALERS = MODEL_DIR / "Hybrid_GRU_scalers.pkl"
STANDALONE_MODEL = MODEL_DIR / "Standalone_GRU_model.keras"
STANDALONE_SCALERS = MODEL_DIR / "Standalone_GRU_scalers.pkl"
PAPER_MODELS = {"hybrid": (HYBRID_MODEL, HYBRID_SCALERS),
                "standalone": (STANDALONE_MODEL, STANDALONE_SCALERS)}

# ---------------------------------------------------------------------------
# Data layout
#   forcing  : basin-average daily precipitation (mm), tmin, tmax (degC),
#              shortwave radiation (kWh/m2/day)
#   simdischarge : HEC-HMS simulated outlet discharge (m3/s)
#   discharge    : observed discharge at Bairabi (m3/s, CWC; target)
# ---------------------------------------------------------------------------
FORCING = ["precip", "tmin", "tmax", "srad"]
FEATURES = {"hybrid": FORCING + ["simdischarge"], "standalone": FORCING}
TARGET = "discharge"

# Training (2018-2020) and validation (2021) tables, as used in the paper.
# The standalone GRU used its own tables (LSTM.csv / LSTM_2021.csv); their
# tmin/tmax differ from the hybrid tables (mean +0.5 / +0.7 degC).
DATA_FILES = {
    "hybrid": (RESTRICTED_DIR / "Training_2018_2020.csv", RESTRICTED_DIR / "Validation_2021.csv"),
    "standalone": (RESTRICTED_DIR / "Standalone_Training_2018_2020.csv",
                   RESTRICTED_DIR / "Standalone_Validation_2021.csv"),
}
TRAIN_FILE, VALID_FILE = DATA_FILES["hybrid"]
SCENARIOS = ["Historical", "Ssp245", "Ssp585"]

# ---------------------------------------------------------------------------
# GRU set-up (Section 3.1, Table 3; identical to the saved model)
# ---------------------------------------------------------------------------
N_STEPS = 15            # look-back window (days)
TEST_SIZE = 0.30        # 70/30 split of the 2018-2020 sequences
SPLIT_SEED = 42         # train_test_split(random_state=42, shuffle=True) as in the paper
SEED = 42
GRU_UNITS = (256, 128, 16)
DROPOUT = 0.20
L2_FIRST = 0.01
LEARNING_RATE = 0.001
BATCH_SIZE = 24
MAX_EPOCHS = 300        # upper limit; early stopping ends training sooner
PATIENCE = 20
START_FROM_EPOCH = 50   # early stopping is active only after epoch 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()


def make_sequences(df: pd.DataFrame, features, target=None, n_steps=N_STEPS, drop_last=False):
    """Sliding windows of n_steps days. Sample k uses days k..k+n_steps-1 and
    predicts the discharge of the LAST day of the window (day k+n_steps-1).

    drop_last=True omits the final window, exactly as in the original training
    and evaluation code (needed to reproduce the 70/30 split of the paper).
    Returns X (n, n_steps, n_features), y (n,) or None, and the target dates.
    """
    values = df[features].to_numpy(dtype=float)
    n = len(df) - n_steps + (0 if drop_last else 1)
    X = np.stack([values[i:i + n_steps] for i in range(n)])
    dates = df.index[n_steps - 1:n_steps - 1 + n]
    y = df[target].to_numpy(dtype=float)[n_steps - 1:n_steps - 1 + n] if target else None
    return X, y, dates


def load_scalers(path=HYBRID_SCALERS):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")     # (scaler_X, scaler_y)


def load_gru(path=HYBRID_MODEL):
    import keras
    return keras.models.load_model(path, compile=False)


def predict(model, scalers, X):
    sx, sy = scalers
    Xs = sx.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return sy.inverse_transform(model.predict(Xs, verbose=0, batch_size=2048)).ravel()


def scores(obs, sim) -> dict:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[ok], sim[ok]
    e = s - o
    top = o >= np.quantile(o, 0.95)
    return {"N": int(ok.sum()),
            "NSE": 1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAE": float(np.mean(np.abs(e))),
            "PBIAS_%": float(100 * e.sum() / o.sum()),
            "R2": float(np.corrcoef(o, s)[0, 1] ** 2),
            "Top5pct_bias_%": float(100 * e[top].sum() / o[top].sum())}

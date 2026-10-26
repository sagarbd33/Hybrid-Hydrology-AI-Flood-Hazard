# -*- coding: utf-8 -*-
"""
Shared settings and helpers for Module 1 (GCM ranking and XGBoost bias correction).

All paths are resolved relative to the folder that contains this file, so the
scripts run unchanged from

    C:\\Users\\AE-lab\\Downloads\\Paper 6\\1. GCM ranking and bias correction

or from any other location (e.g. a cloned GitHub repository).

Manuscript: "A Hybrid Hydrology-AI Framework with Terrain-Based Inundation
Modeling for Probabilistic Flood Hazard-Exposure Assessment under CMIP6
Scenarios" (Debbarma, Bandyopadhyay, Bhadra).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Folder layout (relative to this file)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

HIST_DIR = BASE_DIR / "Historical"         # Date, Observed, Raw_Modeled (1986-2014) per top-5 GCM
PERF_DIR = BASE_DIR / "Performance"        # skill of every GCM (input to the ranking)
RAW_DIR = BASE_DIR / "Raw_Projections"     # raw SSP245/SSP585 areal series from GEE
MODEL_DIR = BASE_DIR / "Models"            # bias-correction models used in the paper (.pkl)
RESULTS_DIR = BASE_DIR / "Results"         # ranking tables, Table 4 metrics, logs
EVAL_MODEL_DIR = BASE_DIR / "Models_holdout"   # models refitted on 1986-2006 for skill assessment
FUTURE_OUT_DIR = BASE_DIR / "Future_BiasCorrected"  # bias-corrected SSP series (per GCM)
DATAGRU_DIR = BASE_DIR / "DataGRU"         # bias-corrected SSP series used in the paper (GRU input)
ENSEMBLE_DIR = BASE_DIR / "Ensemble"       # RS-weighted multi-model ensemble series

# ---------------------------------------------------------------------------
# Variables
#   key      : short name used in folder / file names on disk
#   prefix   : prefix used in GCM file names
#   raw_dir  : sub-folder of Raw_Projections/ with the SSP areal-mean CSVs
#   perf     : file in Performance/ with RMSE, MAE, R2 of every GCM
#   nonneg   : clip corrected values at zero (physical constraint)
#   unit     : unit label for logs
# ---------------------------------------------------------------------------
VARIABLES = {
    "Pr": dict(prefix="PR", raw_dir="Pr", perf="Pr_Performance.csv", nonneg=True, unit="mm/day"),
    "Srad": dict(prefix="RSDS", raw_dir="Srad", perf="Srad_Performance.csv", nonneg=True, unit="kWh/m2/day"),
    "Tmax": dict(prefix="TASMAX", raw_dir="Tmax", perf="Tmax_Performance.csv", nonneg=False, unit="degC"),
    "Tmin": dict(prefix="TASMIN", raw_dir="Tmin", perf="Tmin_Performance.csv", nonneg=False, unit="degC"),
}

SCENARIOS = ["ssp245", "ssp585"]

# Reference period split used for out-of-sample skill (Section 2.3)
TRAIN_START, TRAIN_END = "1986-01-01", "2006-12-31"
TEST_START, TEST_END = "2007-01-01", "2014-12-31"

N_TOP = 5            # number of GCMs retained per variable
RS_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)   # w1 (RMSE), w2 (MAE), w3 (R2) in Eq. (1)
RANDOM_STATE = 42

FEATURES = ["Modeled", "Modeled_Lag1", "DayOfYear"]

# Hyper-parameter space for RandomizedSearchCV (same as original analysis)
PARAM_DIST = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
}
N_ITER = 30
CV_FOLDS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_features(modeled: pd.Series) -> pd.DataFrame:
    """Build the XGBoost predictors from a daily GCM series (DatetimeIndex).

    Modeled      : GCM value on day t
    Modeled_Lag1 : GCM value on day t-1 (GCM only, so no observation leakage)
    DayOfYear    : seasonal indicator
    The first day is dropped because its lag is undefined.
    """
    s = pd.to_numeric(modeled, errors="coerce").sort_index()
    df = pd.DataFrame({"Modeled": s})
    df["Modeled_Lag1"] = df["Modeled"].shift(1)
    df["DayOfYear"] = df.index.dayofyear
    return df.dropna()


def apply_correction(model, modeled: pd.Series, nonneg: bool) -> pd.Series:
    """Corrected = GCM + predicted bias (bias = Observed - Modeled)."""
    X = make_features(modeled)
    corrected = X["Modeled"] + model.predict(X[FEATURES])
    if nonneg:
        corrected = corrected.clip(lower=0.0)
    return corrected


def metrics(obs, sim) -> dict:
    """RMSE, MAE and R2 (coefficient of determination, as in Table 4)."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[ok], sim[ok]
    err = sim - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "N": int(ok.sum())}


def clean_model_name(name: str) -> str:
    """'PR_Areal_ACCESS-CM2' -> 'ACCESS-CM2' (prefix in the performance files)."""
    name = str(name).strip()
    if "_Areal_" in name:
        name = name.split("_Areal_", 1)[1]
    return name


def model_file(var: str, gcm: str, folder: Path = MODEL_DIR) -> Path:
    return folder / f"{VARIABLES[var]['prefix']}_Areal_{gcm}_Tuned_XGBoost.pkl"


def historical_file(var: str, gcm: str) -> Path:
    """Date, Observed, Raw_Modeled (basin-average) for the 1986-2014 reference period."""
    return HIST_DIR / f"{VARIABLES[var]['prefix']}_Areal_{gcm}_historical_1986_2014.csv"


def raw_future_file(var: str, scenario: str, gcm: str) -> Path:
    v = VARIABLES[var]
    return RAW_DIR / v["raw_dir"] / f"{v['prefix']}_Areal_{scenario}_{gcm}.csv"


def read_gee_series(path: Path) -> pd.Series:
    """Read an areal-mean series: Date + one value column.

    Works for the cleaned files in Raw_Projections/ and for untouched Google
    Earth Engine exports (extra 'system:index' and '.geo' columns are ignored).
    """
    df = pd.read_csv(path)
    value_cols = [c for c in df.columns if c not in ("system:index", "Date", ".geo")]
    if len(value_cols) != 1:
        raise ValueError(f"{path.name}: expected one value column, found {value_cols}")
    s = pd.Series(pd.to_numeric(df[value_cols[0]], errors="coerce").values,
                  index=pd.to_datetime(df["Date"]), name=value_cols[0])
    return s.sort_index()


def top_models(var: str) -> pd.DataFrame:
    """Top-N GCMs and ensemble weights written by 01_rank_gcms.py."""
    path = RESULTS_DIR / "Top5_GCMs_and_Weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 01_rank_gcms.py first.")
    df = pd.read_csv(path)
    return df[df["Variable"] == var].reset_index(drop=True)

# -*- coding: utf-8 -*-
"""
Shared settings for Module 4 (discharge-stage relationship at Bairabi).

All paths are relative to this folder, so the scripts run unchanged from
    C:\\Users\\AE-lab\\Downloads\\Paper 6\\4. XGBoost rating curve
or from a cloned repository.

Manuscript: "A Hybrid Hydrology-AI Framework with Terrain-Based Inundation
Modeling for Probabilistic Flood Hazard-Exposure Assessment under CMIP6
Scenarios" (Debbarma, Bandyopadhyay, Bhadra), Sections 3.2, 4.3 and 4.4.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
INPUT_DIR = DATA_DIR / "Inputs"              # projected discharge from Module 3
RESTRICTED_DIR = DATA_DIR / "restricted"     # CWC stage / discharge - NOT for distribution
MODEL_DIR = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"

MODULE3_RESULTS = BASE_DIR.parent / "3. Hybrid HEC-HMS–GRU" / "Results"

SCENARIOS = ["Historical", "Ssp245", "Ssp585"]
SPLIT_DATE = "2021-01-01"     # train: Jun 2017 - Dec 2020, test: Jan 2021 - Jul 2022 (as in the paper)
SEED = 42

# Paired data used in the paper: HEC-HMS-GRU discharge of the SSP run
# ("Modeled", m3/s) and observed stage at Bairabi ("Observed", m), 2017-2022.
PAIRED_FILE = {"Ssp245": RESTRICTED_DIR / "Merged_Output_Ssp245.csv",
               "Ssp585": RESTRICTED_DIR / "Merged_Output_Ssp585.csv"}
OBS_PAIRS_FILE = RESTRICTED_DIR / "Observed_Q_Stage_pairs.csv"   # observed Q and stage, 2018-2021

# XGBoost rating curve exactly as in the original notebook (Table 6)
XGB_RC_PARAMS = dict(n_estimators=500, max_depth=2, learning_rate=0.05, subsample=0.9,
                     colsample_bytree=1.0, random_state=SEED)
XGB_RC_EARLY_STOP = 20

# XGBoost stage-projection model (Section 4.4): stage = Q + bias(Q, Q_lag1, day-of-year)
BIAS_FEATURES = ["Modeled", "Modeled_Lag1", "DayOfYear"]
BIAS_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
}
# GCM-driven runs -> which paired file trained the stage model (as in the paper)
STAGE_MODEL_FOR = {"Historical": "Ssp245", "Ssp245": "Ssp245", "Ssp585": "Ssp585"}

PERIODS = {"Historical": [("1986-2014", 1986, 2014)],
           "Ssp245": [("2021-2047", 2021, 2047), ("2048-2073", 2048, 2073), ("2074-2099", 2074, 2099)],
           "Ssp585": [("2021-2047", 2021, 2047), ("2048-2073", 2048, 2073), ("2074-2099", 2074, 2099)]}


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()


def read_projection(scenario) -> pd.Series:
    """Projected daily discharge (Module 3). Uses the Module 3 results folder when present."""
    m3 = MODULE3_RESULTS / f"{scenario}_EnsembleMean_OptiDischarge.csv"
    src = m3 if m3.exists() else INPUT_DIR / f"{scenario}_OptiDischarge.csv"
    return read_csv(src)["OptiDischarge"].astype(float)


def scores(obs, sim) -> dict:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    e = sim - obs
    top = obs >= np.quantile(obs, 0.9)
    return {"N": len(obs),
            "NSE": 1 - np.sum(e ** 2) / np.sum((obs - obs.mean()) ** 2),
            "RMSE_m": float(np.sqrt(np.mean(e ** 2))),
            "MAE_m": float(np.mean(np.abs(e))),
            "R2": float(np.corrcoef(obs, sim)[0, 1] ** 2),
            "PBIAS_%": float(100 * e.sum() / obs.sum()),
            "Top10_RMSE_m": float(np.sqrt(np.mean(e[top] ** 2))),
            "Top10_PBIAS_%": float(100 * e[top].sum() / obs[top].sum())}


def bias_features(q: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"Modeled": q.astype(float)})
    d["Modeled_Lag1"] = d["Modeled"].shift(1)
    d["DayOfYear"] = d.index.dayofyear
    return d


def annual_max_by_period(stage: pd.Series, scenario: str) -> list:
    amax = stage.groupby(stage.index.year).max()
    return [(label, float(amax.loc[a:b].mean())) for label, a, b in PERIODS[scenario]]

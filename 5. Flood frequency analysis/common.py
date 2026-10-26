# -*- coding: utf-8 -*-
"""
Shared settings and statistical functions for Module 5 (trend tests and
flood frequency analysis of annual maximum water level at Bairabi).

All paths are relative to this folder, so the scripts run unchanged from
    C:\\Users\\AE-lab\\Downloads\\Paper 6\\5. Flood frequency analysis
or from a cloned repository.

Manuscript: "A Hybrid Hydrology-AI Framework with Terrain-Based Inundation
Modeling for Probabilistic Flood Hazard-Exposure Assessment under CMIP6
Scenarios" (Debbarma, Bandyopadhyay, Bhadra), Sections 3.3, 4.4 and 4.5.

The L-moment estimators follow Hosking (1990) and Hosking & Wallis (1997) and
reproduce the parameterisations reported by HEC-SSP 2.3.
"""

from math import gamma as G
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Data" / "Inputs"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = RESULTS_DIR / "Figures"
MODULE3_RESULTS = BASE_DIR.parent / "3. Hybrid HEC-HMS–GRU" / "Results"
MODULE4_RESULTS = BASE_DIR.parent / "4. XGBoost rating curve" / "Results"

SCENARIOS = ["Historical", "Ssp245", "Ssp585"]
# Period of the annual-maximum series used for the frequency analysis
# (HEC-SSP: 85 values = 2015-2099) and for the trend tests (Table 7).
FFA_PERIOD = {"Historical": (1986, 2014), "Ssp245": (2015, 2099), "Ssp585": (2015, 2099)}
TREND_PERIOD = {"Historical": (1986, 2014), "Ssp245": (2021, 2099), "Ssp585": (2021, 2099)}
SUB_PERIODS = [("2021-2047", 2021, 2047), ("2048-2073", 2048, 2073), ("2074-2099", 2074, 2099)]

RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
HAZARD_CLASSES = [("H1 very frequent", None, 10), ("H2 occasional", 10, 100),
                  ("H3 rare", 100, 1000), ("H4 very rare", 1000, None)]
EULER = 0.5772156649


# ---------------------------------------------------------------------------
# Input series
# ---------------------------------------------------------------------------
def read_daily(kind: str, scenario: str, stage_version: str = "paper") -> pd.Series:
    """kind = 'Stage' (Module 4) or 'Discharge' (Module 3). Sibling module
    folders are used when present, otherwise Data/Inputs/."""
    if kind == "Stage":
        f = MODULE4_RESULTS / f"Stage_{scenario}_{stage_version}.csv"
        f = f if f.exists() else INPUT_DIR / f"Stage_{scenario}_{stage_version}.csv"
        return pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Stage"]
    f = MODULE3_RESULTS / f"{scenario}_EnsembleMean_OptiDischarge.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["OptiDischarge"]
    f = INPUT_DIR / f"Stage_{scenario}_{stage_version}.csv"      # carries the Discharge column
    return pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Discharge"]


def annual_max(s: pd.Series, first: int, last: int) -> pd.Series:
    a = s.groupby(s.index.year).max()
    return a.loc[first:last]


# ---------------------------------------------------------------------------
# Trend tests
# ---------------------------------------------------------------------------
def mann_kendall(x) -> dict:
    """Mann-Kendall test with tie correction and Sen's slope."""
    x = np.asarray(x, float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return {"n": n, "S": float(s), "Z": float(z), "p": float(p), "Sen_slope": float(np.median(slopes)),
            "Trend": "increasing" if (p < 0.05 and z > 0) else "decreasing" if (p < 0.05 and z < 0) else "no trend"}


# ---------------------------------------------------------------------------
# Sample L-moments
# ---------------------------------------------------------------------------
def lmoments(x):
    """Unbiased sample L-moments: l1, l2, t3 (L-skew), t4 (L-kurtosis)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1, l2, l3, l4 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0, 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l3 / l2, l4 / l2


# ---------------------------------------------------------------------------
# Distributions fitted by L-moments (HEC-SSP parameterisation)
# Each returns dict(params=..., cdf=f(x), ppf=f(F))
# ---------------------------------------------------------------------------
def gumbel_from_lmoments(l1, l2):
    scale = l2 / np.log(2)
    return l1 - EULER * scale, scale


def gev_from_lmoments(l1, l2, t3):
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c ** 2
    scale = l2 * k / ((1 - 2 ** (-k)) * G(1 + k))
    return l1 - scale * (1 - G(1 + k)) / k, scale, k


def fit_gumbel(x):
    l1, l2, _, _ = lmoments(x)
    scale = l2 / np.log(2)
    loc = l1 - EULER * scale
    return {"params": {"Loctn": loc, "Scale": scale},
            "cdf": lambda v: np.exp(-np.exp(-(np.asarray(v, float) - loc) / scale)),
            "ppf": lambda F: loc - scale * np.log(-np.log(np.asarray(F, float)))}


def fit_gev(x):
    """Hosking (1990). Shape k < 0: heavy upper tail (HEC-SSP sign convention)."""
    loc, scale, k = gev_from_lmoments(*lmoments(x)[:3])

    def cdf(v):
        y = 1 - k * (np.asarray(v, float) - loc) / scale
        y = np.where(y > 0, y, np.nan)
        out = np.exp(-y ** (1 / k))
        return np.where(np.isnan(out), np.where(k < 0, 0.0, 1.0), out)

    return {"params": {"Loctn": loc, "Scale": scale, "Shape": k}, "cdf": cdf,
            "ppf": lambda F: loc + scale * (1 - (-np.log(np.asarray(F, float))) ** k) / k}


def fit_log10_normal(x):
    l1, l2, _, _ = lmoments(np.log10(x))
    mu, sd = l1, l2 * np.sqrt(np.pi)
    return {"params": {"MeanLog": mu, "StDvLog": sd},
            "cdf": lambda v: stats.norm.cdf((np.log10(v) - mu) / sd),
            "ppf": lambda F: 10 ** (mu + sd * stats.norm.ppf(F))}


def fit_log_logistic(x):
    l1, l2, _, _ = lmoments(np.log(x))
    scale, shape = np.exp(l1), 1 / l2
    return {"params": {"Scale": scale, "Shape": shape},
            "cdf": lambda v: 1 / (1 + (np.asarray(v, float) / scale) ** (-shape)),
            "ppf": lambda F: scale * (np.asarray(F, float) / (1 - np.asarray(F, float))) ** (1 / shape)}


def fit_gamma(x):
    """Hosking & Wallis (1997) rational approximation from L-CV."""
    l1, l2, _, _ = lmoments(x)
    t = l2 / l1
    if t < 0.5:
        z = np.pi * t ** 2
        a = (1 - 0.3080 * z) / (z - 0.05812 * z ** 2 + 0.01765 * z ** 3)
    else:
        z = 1 - t
        a = (0.7213 * z - 0.5947 * z ** 2) / (1 - 2.1817 * z + 1.2113 * z ** 2)
    scale = l1 / a
    return {"params": {"Shape": a, "Scale": scale},
            "cdf": lambda v: stats.gamma.cdf(v, a, scale=scale),
            "ppf": lambda F: stats.gamma.ppf(F, a, scale=scale)}


DISTRIBUTIONS = {"Gumbel": fit_gumbel, "GEV": fit_gev, "Log10-Normal": fit_log10_normal,
                 "Log-Logistic": fit_log_logistic, "Gamma": fit_gamma}


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------
def ks_stat(x, cdf):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    F = np.clip(cdf(x), 0, 1)
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - F, F - (i - 1) / n)))


def ad_stat(x, cdf):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    F = np.clip(cdf(x), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n)


def chi2_stat(x, ppf, n_bins=None):
    """Chi-square with equal-probability bins (number of bins = int(2 n^0.4))."""
    x = np.asarray(x, float)
    n = len(x)
    k = n_bins or int(2 * n ** 0.4)
    edges = ppf(np.linspace(0, 1, k + 1)[1:-1])
    obs = np.bincount(np.searchsorted(edges, x), minlength=k)
    exp = n / k
    return float(np.sum((obs - exp) ** 2 / exp))


def return_level(fit, T):
    return float(fit["ppf"](1 - 1 / T))

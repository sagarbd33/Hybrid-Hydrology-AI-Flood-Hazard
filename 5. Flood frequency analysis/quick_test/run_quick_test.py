# -*- coding: utf-8 -*-
"""
Quick test for Module 5 (a few seconds).   python quick_test/run_quick_test.py

1. L-moment fits recover the parameters of a large synthetic Gumbel and GEV sample.
2. From the L-moments printed in the HEC-SSP report of the paper, the Python
   formulas give the HEC-SSP Gumbel/GEV parameters and design stages.
3. Mann-Kendall on the projected annual-maximum discharge gives the Table 7
   values of the paper (SSP585: Z = 2.80, p = 0.005, Sen = 0.124 m3/s/yr).
4. Steps 1-3 on the HEC-SSP annual maxima reproduce the paper's Gumbel
   parameters (SSP245 16.964/2.465, SSP585 17.498/5.263) and the expected
   Table 7/8 files.
Ends with QUICK TEST PASSED.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as c  # noqa: E402


def check(name, ok):
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        sys.exit("QUICK TEST FAILED")


rng = np.random.default_rng(1)
x = 20 - 2.5 * np.log(-np.log(rng.uniform(size=20000)))
p = c.fit_gumbel(x)["params"]
check("Gumbel fit (loc 20, scale 2.5)", abs(p["Loctn"] - 20) < 0.1 and abs(p["Scale"] - 2.5) < 0.1)
k0 = -0.2
y = 15 + 2 * (1 - (-np.log(rng.uniform(size=20000))) ** k0) / k0
p = c.fit_gev(y)["params"]
check("GEV fit (shape -0.2)", abs(p["Shape"] - k0) < 0.03 and abs(p["Scale"] - 2) < 0.1)

cmp = importlib.import_module("04_compare_hecssp")
h = cmp.parse_report(cmp.RPT)
loc, sc = c.gumbel_from_lmoments(h["L_mean"], h["L_CV"] * h["L_mean"])
check("Gumbel parameters = HEC-SSP report", abs(loc - h["Gumbel_loc"]) < 0.01 and abs(sc - h["Gumbel_scale"]) < 0.01)
check("Gumbel 100-yr stage = HEC-SSP report",
      abs(loc - sc * np.log(-np.log(0.99)) - h["Design"][1.0]) < 0.02)

importlib.import_module("01_annual_maxima").main("hecssp")
importlib.import_module("02_trend_tests").main("hecssp")
importlib.import_module("03_frequency_analysis").main("hecssp", plots=False)
t7 = pd.read_csv(c.RESULTS_DIR / "Table7_trends_hecssp.csv")
q = t7[t7.Variable == "Discharge_m3s"].set_index("Scenario")
check("Table 7 discharge Z (-0.69, 0.88, 2.80)",
      np.allclose(q.loc[["Historical", "Ssp245", "Ssp585"], "Z"], [-0.69, 0.88, 2.80], atol=0.01))
check("Table 7 SSP585 Sen slope 0.124", abs(q.loc["Ssp585", "Sen_slope"] - 0.1244) < 0.001)
t8 = pd.read_csv(c.RESULTS_DIR / "Table8_Gumbel_design_stages_hecssp.csv").set_index("Scenario")
check("Paper Gumbel SSP245 16.964/2.465, SSP585 17.498/5.263",
      np.allclose(t8.loc[["Ssp245", "Ssp585"], ["Gumbel_loc", "Gumbel_scale"]].values,
                  [[16.964, 2.465], [17.498, 5.263]], atol=0.002))
for f in ("Table7_trends_hecssp.csv", "Table8_Gumbel_design_stages_hecssp.csv"):
    a = pd.read_csv(c.RESULTS_DIR / f).select_dtypes("number")
    b = pd.read_csv(HERE / "expected" / f).select_dtypes("number")
    check(f"{f} matches expected", np.allclose(a.values, b.values, atol=1e-3, equal_nan=True))
print("QUICK TEST PASSED")

# -*- coding: utf-8 -*-
"""
Step 4 - Check the Python frequency analysis against the HEC-SSP 2.3 report
of the paper (docs/HEC-SSP_Bairabi_Water_Level/DistributionFitting/Ssp245/Ssp245.rpt).

Two checks:
 A. Formulas: the L-moments printed in the HEC-SSP report are converted to
    Gumbel and GEV parameters and Gumbel design stages with the Python code
    and compared with the HEC-SSP values (identical input -> must agree).
 B. Data: the Python results for the SSP245 annual-maximum series of step 1
    are compared with the HEC-SSP report (agreement depends on how closely
    the stage series reproduces the one entered in HEC-SSP).

Output : Results/HECSSP_comparison_<stage>.csv
Run    : python 04_compare_hecssp.py [--stage hecssp|paper|observed]
"""

import argparse
import re

import pandas as pd

from common import BASE_DIR, RESULTS_DIR, gev_from_lmoments, gumbel_from_lmoments

RPT = BASE_DIR / "docs" / "HEC-SSP_Bairabi_Water_Level" / "DistributionFitting" / "Ssp245" / "Ssp245.rpt"


def parse_report(path):
    t = path.read_text(errors="replace")
    num = lambda pat: float(re.search(pat, t).group(1))
    out = {"N": num(r"Sample Size\s*\|\s*([\d.]+)"),
           "Min": num(r"Min\s*\|\s*([\d.]+)"), "Max": num(r"Max\s*\|\s*([\d.]+)"),
           "L_mean": num(r"L-Mean\s*\|\s*([\d.]+)"), "L_CV": num(r"L-CV\s*\|\s*([\d.]+)"),
           "L_skew": num(r"L-Skew\s*\|\s*([-\d.]+)")}
    for name, key in (("Gumbel (LM)", "Gumbel"), ("Generalized Extreme Value (LM)", "GEV"),
                      ("Log10-Normal (LM)", "Log10-Normal"), ("Log-Logistic (LM)", "Log-Logistic"),
                      ("Gamma (LM)", "Gamma")):
        row = re.search(re.escape(name) + r"\s*\|\s*(.*?)\|\s*([\d.]+)\s*\|", t)
        out[f"{key}_params"] = row.group(1).strip()
        out[f"{key}_KS"] = float(row.group(2))
    sel = re.search(r"Loctn=([\d.]+)\s*,Scale=([\d.]+)\s*\|\n\|-", t)
    out["Gumbel_loc"], out["Gumbel_scale"] = float(sel.group(1)), float(sel.group(2))
    curve = re.findall(r"\|\s+([\d.]+)\s+([\d.]+)\s*\|\s+([\d.]+)\s+\|", t)
    out["Design"] = {float(p): float(m) for m, _, p in curve}      # % exceedance -> median curve stage
    return out


def main(stage="paper"):
    h = parse_report(RPT)
    rows = []
    # A. formulas, from the HEC-SSP L-moments
    l1, l2, t3 = h["L_mean"], h["L_CV"] * h["L_mean"], h["L_skew"]
    loc, scale = gumbel_from_lmoments(l1, l2)
    gl, gs, gk = gev_from_lmoments(l1, l2, t3)
    import numpy as np
    rows += [("A formulas", "Gumbel location", h["Gumbel_loc"], loc),
             ("A formulas", "Gumbel scale", h["Gumbel_scale"], scale),
             ("A formulas", "GEV shape", float(re.search(r"Shape=([-\d.]+)", h["GEV_params"]).group(1)), gk)]
    for pct, T in ((10.0, 10), (1.0, 100), (0.1, 1000)):
        q = h["Gumbel_loc"] - h["Gumbel_scale"] * np.log(-np.log(1 - 1 / T))
        rows.append(("A formulas", f"Gumbel stage T={T} yr", h["Design"][pct], q))
    # B. data
    gof = pd.read_csv(RESULTS_DIR / f"FFA_parameters_GoF_{stage}.csv")
    g = gof[(gof.Scenario == "Ssp245")]
    gum = g[g.Distribution == "Gumbel"].iloc[0]
    rows += [("B data", "N", h["N"], gum.N), ("B data", "Sample min", h["Min"], gum.Sample_min),
             ("B data", "Sample max", h["Max"], gum.Sample_max), ("B data", "L-mean", h["L_mean"], gum.L_mean),
             ("B data", "L-skew", h["L_skew"], gum.L_skew),
             ("B data", "Gumbel location", h["Gumbel_loc"], gum.p_Loctn),
             ("B data", "Gumbel scale", h["Gumbel_scale"], gum.p_Scale)]
    for d in ("Gumbel", "GEV", "Log10-Normal", "Log-Logistic", "Gamma"):
        rows.append(("B data", f"K-S {d}", h[f"{d}_KS"], g[g.Distribution == d].KS.iloc[0]))
    df = pd.DataFrame(rows, columns=["Check", "Quantity", "HEC-SSP", "Python"])
    df["Difference"] = df["Python"] - df["HEC-SSP"]
    df.round(4).to_csv(RESULTS_DIR / f"HECSSP_comparison_{stage}.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hecssp", "paper", "observed"], default="hecssp")
    main(ap.parse_args().stage)

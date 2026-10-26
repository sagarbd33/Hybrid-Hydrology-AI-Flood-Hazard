# -*- coding: utf-8 -*-
"""
Step 3 - Flood frequency analysis of annual maximum water level at Bairabi
(Sections 3.3 and 4.5; Table 8, Figs. 11-12). Python replacement for the
HEC-SSP 2.3 "Distribution Fitting Analysis" used in the paper.

* Series: annual maximum stage, 2015-2099 (85 values, as in HEC-SSP) for
  SSP245 and SSP585, and 1986-2014 for the historical baseline.
* Distributions fitted by L-moments: Gumbel, GEV, Log10-Normal, Log-Logistic,
  Gamma (HEC-SSP parameterisations).
* Goodness of fit: Kolmogorov-Smirnov, Anderson-Darling, Chi-square
  (equal-probability bins).
* Return levels for T = 2 ... 1000 years for every distribution; Gumbel
  design stages at the hazard-class boundaries T = 10, 100, 1000 years
  (H1 < 10 yr, H2 10-100 yr, H3 100-1000 yr, H4 > 1000 yr).

Input  : Results/AMS_<stage>.csv (step 1)
Output : Results/FFA_parameters_GoF_<stage>.csv   (parameters + K-S, A-D, Chi-square)
         Results/FFA_return_levels_<stage>.csv    (all distributions)
         Results/Table8_Gumbel_design_stages_<stage>.csv
         Results/Figures/Fig11_GoF_<stage>.png, Fig12_Gumbel_<stage>.png

Run    : python 03_frequency_analysis.py [--stage hecssp|paper|observed]
"""

import argparse

import numpy as np
import pandas as pd

from common import (DISTRIBUTIONS, FFA_PERIOD, FIG_DIR, HAZARD_CLASSES, RESULTS_DIR, RETURN_PERIODS,
                    SCENARIOS, ad_stat, chi2_stat, ks_stat, lmoments, return_level)


def main(stage="paper", plots=True):
    ams = pd.read_csv(RESULTS_DIR / f"AMS_{stage}.csv")
    gof_rows, rl_rows, t8_rows, fits = [], [], [], {}
    for scen in SCENARIOS:
        a, b = FFA_PERIOD[scen]
        x = ams[(ams.Scenario == scen) & ams.Year.between(a, b)]["Stage_m"].values
        l1, l2, t3, t4 = lmoments(x)
        for name, fit_fn in DISTRIBUTIONS.items():
            fit = fit_fn(x)
            fits[(scen, name)] = fit
            gof_rows.append({"Scenario": scen, "Period": f"{a}-{b}", "N": len(x), "Distribution": name,
                             **{f"p_{k}": v for k, v in fit["params"].items()},
                             "KS": ks_stat(x, fit["cdf"]), "AD": ad_stat(x, fit["cdf"]),
                             "ChiSquare": chi2_stat(x, fit["ppf"]),
                             "L_mean": l1, "L_CV": l2 / l1, "L_skew": t3, "L_kurtosis": t4,
                             "Sample_min": x.min(), "Sample_max": x.max()})
            rl_rows.append({"Scenario": scen, "Distribution": name,
                            **{f"T{T}": return_level(fit, T) for T in RETURN_PERIODS}})
        g = fits[(scen, "Gumbel")]
        lv = {T: return_level(g, T) for T in (2, 10, 100, 1000)}
        t8_rows.append({"Scenario": scen, "Gumbel_loc": g["params"]["Loctn"], "Gumbel_scale": g["params"]["Scale"],
                        "T2_m": lv[2], "H1_upper_T10_m": lv[10], "H2_upper_T100_m": lv[100],
                        "H3_upper_T1000_m": lv[1000], "H2_range_m": lv[100] - lv[10],
                        "H3_range_m": lv[1000] - lv[100]})

    gof = pd.DataFrame(gof_rows)
    gof.round(4).to_csv(RESULTS_DIR / f"FFA_parameters_GoF_{stage}.csv", index=False)
    pd.DataFrame(rl_rows).round(3).to_csv(RESULTS_DIR / f"FFA_return_levels_{stage}.csv", index=False)
    t8 = pd.DataFrame(t8_rows)
    t8.round(3).to_csv(RESULTS_DIR / f"Table8_Gumbel_design_stages_{stage}.csv", index=False)

    print(gof[["Scenario", "Distribution", "KS", "AD", "ChiSquare"]].round(3).to_string(index=False))
    print("\nGumbel design stages (m):")
    print(t8.round(2).to_string(index=False))
    gev = gof[gof.Distribution == "GEV"][["Scenario", "p_Shape"]]
    rl = pd.DataFrame(rl_rows)
    for _, r in gev.iterrows():
        t1000 = rl[(rl.Scenario == r.Scenario) & (rl.Distribution == "GEV")]["T1000"].iloc[0]
        print(f"GEV {r.Scenario}: shape {r.p_Shape:.3f}, T=1000 yr stage {t1000:.2f} m")

    if plots:
        plot(gof, ams, fits, stage)


def plot(gof, ams, fits, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fut = gof[gof.Scenario != "Historical"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (col, title) in zip(ax, (("KS", "(a) Kolmogorov-Smirnov"), ("ChiSquare", "(b) Chi-Square"),
                                    ("AD", "(c) Anderson-Darling"))):
        p = fut.pivot(index="Distribution", columns="Scenario", values=col)
        p.plot.bar(ax=a, rot=30)
        a.set(title=title, xlabel="", ylabel="Test statistic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"Fig11_GoF_{stage}.png", dpi=300)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(7, 5))
    T = np.logspace(np.log10(1.01), 3.2, 200)
    for scen, color in (("Ssp245", "tab:blue"), ("Ssp585", "tab:red")):
        g = fits[(scen, "Gumbel")]
        a.plot(T, g["ppf"](1 - 1 / T), color=color, label=f"{scen.upper()} Gumbel")
        a.scatter([10, 100, 1000], [g["ppf"](1 - 1 / t) for t in (10, 100, 1000)], color=color, zorder=3)
        x = np.sort(ams[(ams.Scenario == scen) & ams.Year.between(2015, 2099)]["Stage_m"].values)
        n = len(x)
        Te = 1 / (1 - (np.arange(1, n + 1) - 0.3) / (n + 0.4))      # median plotting position
        a.scatter(Te, x, s=8, color=color, alpha=0.4)
    a.set_xscale("log")
    a.set(xlabel="Return period (years)", ylabel="Water level (m)",
          title="Gumbel flood frequency curves, Bairabi")
    a.legend()
    a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"Fig12_Gumbel_{stage}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hecssp", "paper", "observed"], default="hecssp")
    main(ap.parse_args().stage)

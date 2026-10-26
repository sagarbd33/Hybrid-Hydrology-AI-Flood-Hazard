# -*- coding: utf-8 -*-
"""
Step 1 - Rank the CMIP6 GCMs with the composite Rank Score of Eq. (1).

    RS_j = w1 * (RMSE_min / RMSE_j) + w2 * (MAE_min / MAE_j) + w3 * R2_j
    with w1 = w2 = w3 = 1/3   (Jose and Dwarakish, 2022)

Input  : Performance/<Var>_Performance.csv (columns Model, RMSE, MAE, R2) for
         Pr, Srad, Tmax and Tmin - daily skill of each bias-corrected GCM against
         the observed basin-average series.
Output : Results/<Var>_Ranking_Results.csv   - all GCMs, sorted by RS
         Results/Top5_GCMs_and_Weights.csv   - top-5 GCMs per variable and
                                               ensemble weights RS / sum(RS)

Run    : python 01_rank_gcms.py
"""

import pandas as pd

from common import BASE_DIR, N_TOP, PERF_DIR, RESULTS_DIR, RS_WEIGHTS, VARIABLES, clean_model_name


def rank_variable(var: str) -> pd.DataFrame:
    path = PERF_DIR / VARIABLES[var]["perf"]
    df = pd.read_csv(path)
    missing = {"Model", "RMSE", "MAE", "R2"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df["Model"] = df["Model"].map(clean_model_name)
    w1, w2, w3 = RS_WEIGHTS
    df["RMSE_term"] = df["RMSE"].min() / df["RMSE"]
    df["MAE_term"] = df["MAE"].min() / df["MAE"]
    df["RS"] = w1 * df["RMSE_term"] + w2 * df["MAE_term"] + w3 * df["R2"]
    df = df.sort_values("RS", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Plausibility screen: after bias correction all GCMs have similar RMSE and
    # MAE, so a GCM whose R2 is far above every other GCM usually indicates a
    # transcription error in the input table (it would dominate the ranking).
    r2_sorted = df["R2"].sort_values(ascending=False).values
    if len(r2_sorted) > 1 and r2_sorted[0] - r2_sorted[1] > 0.1:
        top = df.loc[df["R2"].idxmax()]
        print(f"  [CHECK] {var}: {top['Model']} has R2 = {top['R2']:.3f}, but the next best "
              f"GCM has {r2_sorted[1]:.3f} with similar RMSE/MAE - verify this value.")
    return df


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    top_rows = []
    for var in VARIABLES:
        print(f"\n=== {var} ===")
        ranked = rank_variable(var)
        out = RESULTS_DIR / f"{var}_Ranking_Results.csv"
        ranked.round(4).to_csv(out, index=False)
        print(f"  {len(ranked)} GCMs ranked -> {out.relative_to(BASE_DIR)}")

        top = ranked.head(N_TOP).copy()
        top["Weight"] = top["RS"] / top["RS"].sum()
        top.insert(0, "Variable", var)
        top_rows.append(top[["Variable", "Rank", "Model", "RMSE", "MAE", "R2", "RS", "Weight"]])
        print(top[["Rank", "Model", "RS", "Weight"]].round(4).to_string(index=False))

    top_all = pd.concat(top_rows, ignore_index=True)
    out = RESULTS_DIR / "Top5_GCMs_and_Weights.csv"
    top_all.round(6).to_csv(out, index=False)
    print(f"\nTop-{N_TOP} table -> {out.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Step 1 - Annual maximum series of daily discharge (Module 3) and water level
(Module 4) at Bairabi.

--stage paper     stage projected with the method used in the paper 
--stage observed  stage from the observed power-law rating curve (Module 4)
--stage hecssp    SSP annual-maximum stages exactly as entered in HEC-SSP for
                  the paper (Data/Inputs/AMS_stage_HECSSP_paper.csv, 2015-2099);
                  historical stage and all discharge as in --stage paper.
                  Use this to reproduce the published Tables 7-8.

Output : Results/AMS_<stage>.csv   Scenario, Year, Discharge_m3s, Stage_m

Run    : python 01_annual_maxima.py [--stage hecssp|paper|observed]
"""

import argparse

import pandas as pd

from common import INPUT_DIR, RESULTS_DIR, SCENARIOS, read_daily


def main(stage="paper"):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    daily = "paper" if stage == "hecssp" else stage
    if stage == "hecssp":
        hec = pd.read_csv(INPUT_DIR / "AMS_stage_HECSSP_paper.csv").set_index("Year")
    for scen in SCENARIOS:
        q = read_daily("Discharge", scen, daily)
        h = read_daily("Stage", scen, daily)
        aq = q.groupby(q.index.year).max()
        ah = h.groupby(h.index.year).max()
        if stage == "hecssp" and scen in hec:
            ah = hec[scen]
        df = pd.DataFrame({"Discharge_m3s": aq, "Stage_m": ah}).dropna()
        df.insert(0, "Scenario", scen)
        df.index.name = "Year"
        rows.append(df.reset_index())
        print(f"{scen:<10} {df.index.min()}-{df.index.max()}: {len(df)} years; stage "
              f"{df.Stage_m.min():.2f}-{df.Stage_m.max():.2f} m, discharge "
              f"{df.Discharge_m3s.min():.1f}-{df.Discharge_m3s.max():.1f} m3/s")
    out = pd.concat(rows)
    out.round(4).to_csv(RESULTS_DIR / f"AMS_{stage}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hecssp", "paper", "observed"], default="hecssp")
    main(ap.parse_args().stage)

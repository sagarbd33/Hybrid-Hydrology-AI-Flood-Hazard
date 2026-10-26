# -*- coding: utf-8 -*-
"""
Step 6 - Export simulated daily discharge at the basin outlet (Bairabi) from
the HEC-HMS result files to CSV. These series are the 'HEC-HMS simulated
flow' input of the hybrid HEC-HMS-GRU model (Module 3).

Input  : HMS_Projects/<Scenario>/Cal.dss   record //Outlet/FLOW//1DAY/RUN:Cal/
Output : Results/HMS_Outlet_Q_<Scenario>.csv   columns Date, Q_HMS_m3s

Run    : python 06_export_results.py            (all scenarios)
         python 06_export_results.py Ssp245     (selected scenarios)
"""

import sys

import numpy as np
import pandas as pd

from common import OUTLET, RESULTS_DIR, RUN_NAME, SCENARIOS, project_dir


def read_daily(dss_file, pathname, start, end) -> pd.Series:
    """Read a daily regular series; value at 00:00 of day T is labelled T."""
    from pydsstools.heclib.dss.HecDss import Open

    window = (pd.Timestamp(start).strftime("%d%b%Y 00:00").upper(),
              (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%d%b%Y 00:00").upper())
    with Open(str(dss_file)) as fid:
        ts = fid.read_ts(pathname, window=window, value_precision="double")
        values = np.array(ts.values, dtype=float)
        t0 = ts.start_time
        first = pd.Timestamp(t0.year(), t0.month(), t0.day())   # pydsstools HecTime
    s = pd.Series(values, index=pd.date_range(first, periods=len(values), freq="D"))
    s[s <= -900] = np.nan          # DSS missing value
    return s.loc[start:end]


def export(scenario: str, outlet=OUTLET, run=RUN_NAME):
    cfg = SCENARIOS[scenario]
    dss = project_dir(scenario) / f"{run}.dss"
    if not dss.exists():
        print(f"[SKIP] {scenario}: {dss.name} not found (run 05_run_hms.py first)")
        return None
    q = read_daily(dss, f"//{outlet}/FLOW//1DAY/RUN:{run}/", cfg["start"], cfg["end"])
    out = pd.DataFrame({"Date": q.index.strftime("%Y-%m-%d"), "Q_HMS_m3s": q.values.round(4)})
    RESULTS_DIR.mkdir(exist_ok=True)
    dst = RESULTS_DIR / f"HMS_Outlet_Q_{scenario}.csv"
    out.to_csv(dst, index=False)
    print(f"{scenario:<20} {len(out)} days, mean {np.nanmean(q):7.2f} m3/s, "
          f"max {np.nanmax(q):8.2f} m3/s -> {dst.name}")
    return dst


if __name__ == "__main__":
    for s in sys.argv[1:] or SCENARIOS:
        export(s)

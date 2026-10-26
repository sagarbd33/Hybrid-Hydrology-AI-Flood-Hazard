# -*- coding: utf-8 -*-
"""
Step 4 - Build one runnable HEC-HMS 4.10 project per scenario.

For every scenario in common.SCENARIOS this script
  1. copies the calibrated template project (HMS_Template/) to
     HMS_Projects/<Scenario>/ and renames it <Scenario>.hms/.gage/.run;
  2. writes the daily forcing (12 rain gages, 12 sub-basin temperature gages
     and, if available, observed discharge at Bairabi) to
     HMS_Projects/<Scenario>/<Scenario>.dss  (HEC-DSS 7, via pydsstools);
  3. updates the gage start/end times and the Calibration.control window.

The calibrated basin model (Tlawng_SMA_FN.basin) and meteorologic model
(Met_1.met: weighted gages, Hargreaves ET) are copied unchanged.

Run:  python 04_setup_projects.py                   (all scenarios)
      python 04_setup_projects.py Ssp245 Ssp585     (selected scenarios)
"""

import re
import shutil
import sys

import pandas as pd

from common import (BASIN_FILE, FLOW_GAGE, OBS_FLOW_CSV, PRECIP_GAGES, PROJECTS_DIR,
                    SCENARIOS, TEMP_GAGES, TEMPLATE_DIR, TEMPLATE_NAME, hms_date,
                    project_dir, read_forcing)

EOL = "\r\n"   # HEC-HMS files use Windows line endings


# ---------------------------------------------------------------------------
# DSS writing
# ---------------------------------------------------------------------------
def write_dss(dss_path, series: dict, units: str, dtype: str, cpart: str):
    """Write daily regular time series. series: {gage name: pd.Series}."""
    from pydsstools.heclib.dss.HecDss import Open

    with Open(str(dss_path), mode="rw") as fid:
        for name, s in series.items():
            s = s.asfreq("D")
            start = s.index[0]
            fid.put_ts(f"//{name}/{cpart}//1DAY/GAGE/",
                       values=s.astype(float).to_numpy(),
                       start_time=start.strftime("%d%b%Y %H:%M").upper(),
                       data_units=units, data_type=dtype)


# ---------------------------------------------------------------------------
# Text-file editing helpers
# ---------------------------------------------------------------------------
def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def write_text(path, text):
    path.write_bytes(text.replace("\n", EOL).encode("utf-8"))


def set_gage_block(text, gage, dss_name, start, end):
    """Update DSS file and start/end time inside the 'Gage: <name>' block."""
    pat = re.compile(rf"(^Gage: {re.escape(gage)}\n.*?^End:)", re.S | re.M)
    m = pat.search(text)
    if not m:
        raise KeyError(f"Gage '{gage}' not found in template .gage file")
    block = m.group(1)
    block = re.sub(r"(DSS File Name: ).*", rf"\g<1>{dss_name}", block)
    block = re.sub(r"(Start Time: ).*", rf"\g<1>{hms_date(start)}, 00:00", block)
    block = re.sub(r"(End Time: ).*", rf"\g<1>{hms_date(end)}, 00:00", block)
    return text[:m.start()] + block + text[m.end():]


def set_control(path, start, end):
    """Values are stamped at 00:00 of each date (= 24:00 of the previous day)."""
    t = read_text(path)
    s0, e0 = start - pd.Timedelta(days=1), end - pd.Timedelta(days=1)
    t = re.sub(r"(Start Date: ).*", rf"\g<1>{hms_date(s0)}", t)
    t = re.sub(r"(Start Time: ).*", r"\g<1>24:00", t)
    t = re.sub(r"(End Date: ).*", rf"\g<1>{hms_date(e0)}", t)
    t = re.sub(r"(End Time: ).*", r"\g<1>24:00", t)
    write_text(path, t)


# ---------------------------------------------------------------------------
def setup(scenario: str, cfg: dict = None, projects_dir=None, verbose=True):
    """Create HMS project <projects_dir>/<scenario>. cfg defaults to SCENARIOS[scenario]."""
    cfg = cfg or SCENARIOS[scenario]
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    pdir = (projects_dir / scenario) if projects_dir else project_dir(scenario)
    if pdir.exists():
        shutil.rmtree(pdir)
    shutil.copytree(TEMPLATE_DIR, pdir)

    # Rename project files and internal names
    for ext in ("hms", "gage", "run"):
        (pdir / f"{TEMPLATE_NAME}.{ext}").rename(pdir / f"{scenario}.{ext}")
    hms = read_text(pdir / f"{scenario}.hms")
    hms = re.sub(r"^Project: .*", f"Project: {scenario}", hms, flags=re.M)
    hms = re.sub(r"(DSS File Name: ).*", rf"\g<1>{scenario}.dss", hms)
    write_text(pdir / f"{scenario}.hms", hms)
    run = read_text(pdir / f"{scenario}.run")
    run = re.sub(r"\s+Last Execution (Date|Time): .*", "", run)
    write_text(pdir / f"{scenario}.run", run)

    # Forcing
    pr = read_forcing(cfg["precip"], start, end)
    tas = read_forcing(cfg["temp"], start, end)
    for df, what in ((pr, "precipitation"), (tas, "temperature")):
        if df.index[0] != start or df.index[-1] != end or len(df) != (end - start).days + 1:
            raise ValueError(f"{scenario}: {what} does not cover {start.date()}-{end.date()} completely")

    n_nan = int(pr.isna().sum().sum())
    pr = pr.fillna(0.0)          # as in the original analysis (missing rain -> 0)
    dss = pdir / f"{scenario}.dss"
    write_dss(dss, {g: pr[c] for g, c in PRECIP_GAGES.items()}, "MM", "PER-CUM", "PRECIP-INC")
    write_dss(dss, {g: tas[c] for g, c in TEMP_GAGES.items()}, "C", "PER-AVER", "TEMPERATURE")

    gage = read_text(pdir / f"{scenario}.gage")
    gage = re.sub(r"^Gage Manager: .*", f"Gage Manager: {scenario}", gage, flags=re.M)
    for g in list(PRECIP_GAGES) + list(TEMP_GAGES):
        gage = set_gage_block(gage, g, dss.name, start, end)

    # Observed discharge (only for observed-forcing scenarios and only if available)
    obs_note = "no observed discharge"
    basin = read_text(pdir / BASIN_FILE)
    use_obs = scenario.startswith("Observed") and OBS_FLOW_CSV.exists()
    if use_obs:
        q = pd.read_csv(OBS_FLOW_CSV, parse_dates=["Date"]).set_index("Date")[FLOW_GAGE]
        q = q.asfreq("D").loc[start:end]
        if q.notna().sum() == 0:
            use_obs = False
    if use_obs:
        q = q.loc[q.first_valid_index():q.last_valid_index()]
        q_dss = q.astype(float).fillna(-901.0)      # -901 = DSS missing value
        write_dss(dss, {FLOW_GAGE: q_dss}, "CMS", "INST-VAL", "FLOW")
        gage = set_gage_block(gage, FLOW_GAGE, dss.name, q.index[0], q.index[-1])
        obs_note = f"observed discharge {q.index[0].date()} to {q.index[-1].date()} ({q.notna().sum()} days)"
    else:
        # no observations: unlink the observed hydrograph from the outlet
        basin = re.sub(r"\n\s+Observed Hydrograph Gage: .*", "", basin)
    write_text(pdir / BASIN_FILE, basin)
    write_text(pdir / f"{scenario}.gage", gage)

    set_control(pdir / "Calibration.control", start, end)

    if verbose:
        print(f"{scenario:<20} {start.date()} to {end.date()} | {len(pr)} days | "
              f"{n_nan} missing rain values set to 0 | {obs_note}")
    return pdir


def main(selected):
    PROJECTS_DIR.mkdir(exist_ok=True)
    for scenario in selected or SCENARIOS:
        setup(scenario)
    print(f"\nProjects written to {PROJECTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])

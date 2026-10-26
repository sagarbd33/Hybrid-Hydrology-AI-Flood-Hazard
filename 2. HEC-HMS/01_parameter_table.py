# -*- coding: utf-8 -*-
"""
Step 1 - Extract the calibrated HEC-HMS parameters (Table 2 of the paper)
directly from the basin model file, so the table always matches the model.

Input  : HMS_Template/Tlawng_SMA_FN.basin
Output : Results/HMS_parameters_by_element.csv  every parameter, every sub-basin / reach
         Results/Table2_parameter_ranges.csv    min-max per parameter (Table 2)

Run    : python 01_parameter_table.py
"""

import re

import pandas as pd

from common import BASIN_FILE, RESULTS_DIR, TEMPLATE_DIR

# (process, label in Table 2, key in .basin file, element type)
TABLE2 = [
    ("Canopy interception", "Initial canopy storage (%)", "Initial Canopy Storage Percent", "Subbasin"),
    ("Canopy interception", "Maximum canopy storage (mm)", "Canopy Storage Capacity", "Subbasin"),
    ("Canopy interception", "Crop coefficient", "Crop Coefficient", "Subbasin"),
    ("Surface depression", "Initial surface storage (%)", "Initial Surface Storage Percent", "Subbasin"),
    ("Surface depression", "Maximum surface storage (mm)", "Surface Storage Capacity", "Subbasin"),
    ("SMA", "Impervious area (%)", "Percent Impervious Area", "Subbasin"),
    ("SMA", "Initial soil storage (%)", "Initial Soil Storage Percent", "Subbasin"),
    ("SMA", "Initial GW 1 storage (%)", "Initial Gw1 Storage Percent", "Subbasin"),
    ("SMA", "Initial GW 2 storage (%)", "Initial Gw2 Storage Percent", "Subbasin"),
    ("SMA", "Maximum infiltration (mm/h)", "Soil Maximum Infiltration", "Subbasin"),
    ("SMA", "Soil storage (mm)", "Soil Storage Capacity", "Subbasin"),
    ("SMA", "Tension storage (mm)", "Soil Tension Capacity", "Subbasin"),
    ("SMA", "Soil percolation (mm/h)", "Soil Maximum Percolation", "Subbasin"),
    ("SMA", "GW 1 storage (mm)", "Groundwater 1 Storage Capacity", "Subbasin"),
    ("SMA", "GW 1 percolation (mm/h)", "Groundwater 1 Maximum Percolation", "Subbasin"),
    ("SMA", "GW 1 storage coefficient (h)", "Groundwater 1 Routing Coefficient", "Subbasin"),
    ("SMA", "GW 2 storage (mm)", "Groundwater 2 Storage Capacity", "Subbasin"),
    ("SMA", "GW 2 percolation (mm/h)", "Groundwater 2 Maximum Percolation", "Subbasin"),
    ("SMA", "GW 2 storage coefficient (h)", "Groundwater 2 Routing Coefficient", "Subbasin"),
    ("Clark unit hydrograph", "Time of concentration (h)", "Time of Concentration", "Subbasin"),
    ("Clark unit hydrograph", "Storage coefficient (h)", "Storage Coefficient", "Subbasin"),
    ("Linear reservoir", "GW 1 initial baseflow (m3/s)", "GW-1 Initial Baseflow", "Subbasin"),
    ("Linear reservoir", "Baseflow GW 1 storage coefficient (h)", "GW-1 Routing Coefficient", "Subbasin"),
    ("Linear reservoir", "GW 1 reservoirs", "GW-1 Number Reservoirs", "Subbasin"),
    ("Linear reservoir", "GW 1 baseflow fraction", "GW-1 Baseflow Fraction", "Subbasin"),
    ("Linear reservoir", "GW 2 initial baseflow (m3/s)", "GW-2 Initial Baseflow", "Subbasin"),
    ("Linear reservoir", "Baseflow GW 2 storage coefficient (h)", "GW-2 Routing Coefficient", "Subbasin"),
    ("Linear reservoir", "GW 2 reservoirs", "GW-2 Number Reservoirs", "Subbasin"),
    ("Linear reservoir", "GW 2 baseflow fraction", "GW-2 Baseflow Fraction", "Subbasin"),
    ("Muskingum routing", "K (h)", "Muskingum K", "Reach"),
    ("Muskingum routing", "X", "Muskingum x", "Reach"),
    ("Muskingum routing", "Number of subreaches", "Muskingum Steps", "Reach"),
]


def parse_basin(path):
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    rows = []
    for kind, name, body in re.findall(r"^(Subbasin|Reach): (.+?)\n(.*?)^End:", text, re.S | re.M):
        row = {"Type": kind, "Name": name.strip()}
        for key, val in re.findall(r"^\s+([A-Za-z0-9 \-]+?): (.+)$", body, re.M):
            row.setdefault(key.strip(), val.strip())   # first occurrence (GW layer 1 vs 2 handled below)
        # Linear reservoir: two 'Groundwater Layer' blocks with repeated keys
        layers = re.split(r"\n\s+Groundwater Layer: ", body)
        for blk in layers[1:]:
            layer = blk.split("\n", 1)[0].strip()
            for key, val in re.findall(r"^\s+(GW-\d [A-Za-z ]+?): (.+)$", blk, re.M):
                if key.startswith(f"GW-{layer}"):
                    row[key] = val.strip()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = parse_basin(TEMPLATE_DIR / BASIN_FILE)
    keys = [k for _, _, k, _ in TABLE2]
    out = df[["Type", "Name"] + [k for k in keys if k in df.columns] + (["Area"] if "Area" in df else [])]
    out.to_csv(RESULTS_DIR / "HMS_parameters_by_element.csv", index=False)

    rows = []
    for proc, label, key, etype in TABLE2:
        if key not in df.columns:
            continue
        v = pd.to_numeric(df.loc[df["Type"] == etype, key], errors="coerce").dropna()
        if v.empty:
            continue
        rows.append({"Hydrological process": proc, "Parameter": label,
                     "Min": v.min(), "Max": v.max(), "N_elements": len(v)})
    t2 = pd.DataFrame(rows)
    t2.to_csv(RESULTS_DIR / "Table2_parameter_ranges.csv", index=False)
    pd.set_option("display.width", 160)
    print(t2.to_string(index=False))
    print(f"\n{(df.Type == 'Subbasin').sum()} sub-basins, {(df.Type == 'Reach').sum()} reaches "
          f"-> {RESULTS_DIR.name}/Table2_parameter_ranges.csv")


if __name__ == "__main__":
    main()

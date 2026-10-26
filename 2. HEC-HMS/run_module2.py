# -*- coding: utf-8 -*-
"""
Run Module 2 end-to-end:
    00 data check -> 01 parameter table -> 02 sub-basin temperature
    -> 03 station-wise bias correction -> 04 build projects
    -> 05 run HEC-HMS (Windows) -> 06 export outlet flow -> 07 evaluate -> 08 GIS

Run:  python run_module2.py              all steps
      python run_module2.py --no-hms     skip step 5 (no HEC-HMS available)
"""

import argparse
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["00_check_input_data.py", "01_parameter_table.py",
         "02_build_subbasin_temperature.py", "03_bias_correct_stationwise.py", "04_setup_projects.py",
         "05_run_hms.py", "06_export_results.py", "07_evaluate.py", "08_export_gis.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hms", action="store_true", help="skip running HEC-HMS (step 5)")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    sys.argv = sys.argv[:1]          # steps read their own (empty) arguments
    for step in STEPS:
        if args.no_hms and step.startswith("03"):
            continue
        print(f"\n{'#' * 70}\n# {step}\n{'#' * 70}")
        t0 = time.time()
        runpy.run_path(str(HERE / step), run_name="__main__")
        print(f"[{step} finished in {time.time() - t0:.1f} s]")

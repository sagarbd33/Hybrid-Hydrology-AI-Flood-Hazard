# -*- coding: utf-8 -*-
"""
Run Module 3 end-to-end:
    01 build GRU inputs -> 03 evaluate trained model (Table 5)
    -> 04 projections (historical, SSP245, SSP585)
Training (02) and the seed study (05) are run separately because they
re-train the networks.

Run:  python run_module3.py
"""

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_build_gru_inputs.py", "03_evaluate_gru.py", "04_project_future.py"]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.argv = sys.argv[:1]
    for step in STEPS:
        print(f"\n{'#' * 70}\n# {step}\n{'#' * 70}")
        t0 = time.time()
        runpy.run_path(str(HERE / step), run_name="__main__")
        print(f"[{step} finished in {time.time() - t0:.1f} s]")

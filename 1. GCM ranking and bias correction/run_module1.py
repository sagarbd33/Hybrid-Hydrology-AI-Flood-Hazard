# -*- coding: utf-8 -*-
"""
Run Module 1 end-to-end:
    01_rank_gcms.py -> 02_evaluate_bias_correction.py
    -> 03_apply_bias_correction.py -> 04_weighted_ensemble.py

Run from any folder:  python run_module1.py
Skip the (slower) hold-out evaluation:  python run_module1.py --skip-eval
"""

import argparse
import runpy
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_rank_gcms.py", "02_evaluate_bias_correction.py",
         "03_apply_bias_correction.py", "04_weighted_ensemble.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true", help="skip step 2 (hold-out skill)")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(HERE))
    for step in STEPS:
        if args.skip_eval and step.startswith("02"):
            continue
        print(f"\n{'#' * 70}\n# {step}\n{'#' * 70}")
        t0 = time.time()
        runpy.run_path(str(HERE / step), run_name="__main__")
        print(f"[{step} finished in {time.time() - t0:.1f} s]")

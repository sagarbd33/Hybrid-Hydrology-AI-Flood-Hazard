# -*- coding: utf-8 -*-
"""
Run Module 4 end-to-end:
    01 rating curves (paper data and observed pairs) -> 02 stage projections
       (paper method and observed power-law rating)

Run:  python run_module4.py
"""

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = [("01_rating_curves.py", ["--data", "paper"]),
        ("01_rating_curves.py", ["--data", "observed"]),
        ("02_stage_projection.py", ["--method", "paper"]),
        ("02_stage_projection.py", ["--method", "observed"])]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for step, args in RUNS:
        print(f"\n{'#' * 70}\n# {step} {' '.join(args)}\n{'#' * 70}")
        sys.argv = [step] + args
        t0 = time.time()
        runpy.run_path(str(HERE / step), run_name="__main__")
        print(f"[finished in {time.time() - t0:.1f} s]")

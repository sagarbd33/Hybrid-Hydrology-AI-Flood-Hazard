# -*- coding: utf-8 -*-
"""Run the quick test of every module.   python run_all_quick_tests.py

Each test uses only the data in this repository (no restricted data, no
HEC-HMS installation) and ends with "QUICK TEST PASSED".
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "quick_test" / "run_quick_test.py").exists())

results = {}
for m in MODULES:
    print(f"\n######## {m.name} ########", flush=True)
    r = subprocess.run([sys.executable, str(m / "quick_test" / "run_quick_test.py")], cwd=m)
    results[m.name] = r.returncode == 0

print("\nSummary")
for name, ok in results.items():
    print(f"  {'PASSED' if ok else 'FAILED'}  {name}")
sys.exit(0 if all(results.values()) else 1)

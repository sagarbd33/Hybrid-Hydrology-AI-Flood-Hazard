# -*- coding: utf-8 -*-
"""Run all Module 5 steps for both stage versions.   python run_module5.py"""
import importlib

for stage in ("hecssp", "paper", "observed"):
    print(f"\n######## stage version: {stage} ########")
    for step in ("01_annual_maxima", "02_trend_tests", "03_frequency_analysis"):
        print(f"\n==== {step} ====")
        importlib.import_module(step).main(stage)
print("\n==== 04_compare_hecssp ====")
importlib.import_module("04_compare_hecssp").main("hecssp")

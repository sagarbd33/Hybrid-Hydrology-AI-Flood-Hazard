# Module 5 – Trend tests and flood frequency analysis at Bairabi

Code and data for Module 5 of:

> Debbarma, S., Bandyopadhyay, A., Bhadra, A. *A Hybrid Hydrology–AI Framework with
> Terrain-Based Inundation Modeling for Probabilistic Flood Hazard–Exposure Assessment
> under CMIP6 Scenarios.* Submitted to Computers & Geosciences.

Contact: Sagar Debbarma (sagarbd33@gmail.com), NERIST, Arunachal Pradesh, India.

This module covers Sections 3.3, 4.4 and 4.5 of the paper:

* Mann–Kendall trend test and Sen's slope for annual maximum discharge and water level
  (Table 7).
* Flood frequency analysis of annual maximum water level, with five distributions
  fitted by L-moments, goodness-of-fit tests (Fig. 11) and Gumbel design stages for the
  hazard classes (Table 8, Fig. 12).

The paper's frequency analysis was run in **HEC-SSP 2.3**. This module reproduces it in
open Python code, using the same L-moment estimators and HEC-SSP's parameterisations. The
original HEC-SSP project is kept in `docs/` for reference.

## Requirements

Python 3.10 or newer: `pip install -r requirements.txt` (numpy, pandas, scipy,
matplotlib).

## Quick test (a few seconds)

```
python quick_test/run_quick_test.py
```

The test checks four things:

1. The L-moment fits recover the parameters of synthetic Gumbel and GEV samples.
2. The Gumbel parameters and 100-year stage computed from the L-moments in the HEC-SSP
   report match the report.
3. The Table 7 discharge trend values are reproduced (SSP585 Z = 2.80, p = 0.005).
4. The paper's Gumbel parameters are reproduced from the HEC-SSP annual maxima, and the
   Table 7 and Table 8 outputs match `quick_test/expected/`.

It ends with `QUICK TEST PASSED`.

## Full run

```
cd "<path>/5. Flood frequency analysis"
python run_module5.py
```

| Script | Purpose | Output |
|---|---|---|
| `01_annual_maxima.py --stage hecssp\|paper\|observed` | Annual maxima of daily discharge (Module 3) and stage (Module 4) | `Results/AMS_<stage>.csv` |
| `02_trend_tests.py` | Mann–Kendall test (tie-corrected, α = 0.05), Sen's slope and 30-year means | `Results/Table7_trends_<stage>.csv` |
| `03_frequency_analysis.py` | Fits Gumbel, GEV, Log10-Normal, Log-Logistic and Gamma by L-moments; K-S, A-D and Chi-square tests; return levels for 2–1000 years; Gumbel hazard-class stages | `Results/FFA_parameters_GoF_*.csv`, `FFA_return_levels_*.csv`, `Table8_*.csv`, `Figures/Fig11_*`, `Fig12_*` |
| `04_compare_hecssp.py` | Compares the Python results with the HEC-SSP report | `Results/HECSSP_comparison_hecssp.csv` |

**Periods.**

* Frequency analysis: 2015–2099 for the SSPs (85 values, as entered in HEC-SSP) and
  1986–2014 for the historical baseline.
* Trend tests and 30-year means (Table 7): 2021–2099, with sub-periods 2021–2047,
  2048–2073 and 2074–2099.

**Hazard classes (Gumbel return period).**

| Class | Return period |
|---|---|
| H1 | < 10 yr |
| H2 | 10–100 yr |
| H3 | 100–1000 yr |
| H4 | > 1000 yr |

**Stage versions.**

* `hecssp` (default): the SSP annual-maximum stages exactly as entered in HEC-SSP
  for the paper (`Data/Inputs/AMS_stage_HECSSP_paper.csv`). This reproduces the
  published values: Gumbel SSP245 16.964/2.465 and SSP585 17.498/5.263 (location/scale);
  GEV shape −0.41 and −0.56.
* `paper`: stage projected with the method used in the paper (Module 4,
  `--method paper`).
* `observed`: stage from the power-law rating curve fitted to observed discharge–stage
  pairs.

## Data

`Data/Inputs/` holds the daily projected discharge and stage (Date, Discharge, Stage) for
the Historical, SSP245 and SSP585 runs, in both stage versions, plus `AMS_stage_HECSSP_paper.csv` (Year, Ssp245, Ssp585), the
annual-maximum stages used in HEC-SSP. These are model outputs,
not restricted observations. When the sibling Module 3 and Module 4 folders are present,
their `Results/` are read instead.

## License

MIT (see repository root).

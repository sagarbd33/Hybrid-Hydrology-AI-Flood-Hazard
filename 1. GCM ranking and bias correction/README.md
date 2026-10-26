# Module 1 – GCM ranking and XGBoost bias correction

Code and data for Module 1 of:

> Debbarma, S., Bandyopadhyay, A., Bhadra, A. *A Hybrid Hydrology–AI Framework with
> Terrain-Based Inundation Modeling for Probabilistic Flood Hazard–Exposure Assessment
> under CMIP6 Scenarios.* Submitted to Computers & Geosciences.

Contact: Sagar Debbarma (sagarbd33@gmail.com), Department of Agricultural Engineering,
NERIST, Arunachal Pradesh, India.

This module covers Sections 2.3 and 4.1 of the paper:

1. Rank NEX-GDDP-CMIP6 GCMs per variable with the composite Rank Score (Eq. 1).
2. Measure the out-of-sample skill of the XGBoost bias correction (Table 4).
3. Bias-correct the SSP245 and SSP585 projections of the top-5 GCMs.
4. Build the RS-weighted multi-model ensemble.

Variables are daily basin-average (areal) series for the Tlawng River Basin, Mizoram,
India: precipitation `pr` (mm/day), shortwave radiation `rsds` (kWh/m²/day), and maximum
and minimum air temperature `tasmax`, `tasmin` (°C).

## Requirements

* Python 3.10 or newer
* `pip install -r requirements.txt`
* The `.pkl` models were saved with **xgboost 3.4.1**. Install that version to load them.
* Optional: a Google Earth Engine account, only to re-extract the raw GCM series
  (`GEE/extract_areal_nexgddp_cmip6.js`).

## Quick test (about 10 s)

```
python quick_test/run_quick_test.py
```

The test:

* re-ranks the GCMs and checks that the top-5 per variable matches the paper;
* bias-corrects a 2015–2016 SSP585 sample (`quick_test/input/`, one GCM per variable);
* compares the result with `quick_test/expected/` (tolerance 1e-4).

It ends with `QUICK TEST PASSED`.

## Full run (about 2 min)

```
cd "<path>/1. GCM ranking and bias correction"
python run_module1.py              # steps 1-4
python run_module1.py --skip-eval  # skip step 2 (hold-out skill)
```

Each step can also be run on its own. All paths are relative to this folder.

| Script | Purpose | Main output |
|---|---|---|
| `01_rank_gcms.py` | Eq. (1) Rank Score, top-5 GCMs, weights `RS/ΣRS` | `Results/Top5_GCMs_and_Weights.csv`, `Results/<Var>_Ranking_Results.csv` |
| `02_evaluate_bias_correction.py` | Tune on 1986–2006, evaluate on held-out 2007–2014 | `Results/Table4_BiasCorrection_Holdout.csv`, `Models_holdout/` |
| `03_apply_bias_correction.py` | Correct SSP245/SSP585 series with the models in `Models/` | `Future_BiasCorrected/`, `Results/Future_correction_summary.csv` |
| `04_weighted_ensemble.py` | RS-weighted ensemble per scenario | `Ensemble/Ensemble_ssp245.csv`, `Ensemble/Ensemble_ssp585.csv` |
| `common.py` | Paths, settings, shared functions | – |

## Folder contents

```
GEE/extract_areal_nexgddp_cmip6.js    Earth Engine script: basin-average NEX-GDDP-CMIP6 series
Basin Shapefile/Tlawng_Catchment.*    catchment boundary used for areal averaging
Observed Data/
    Tlawng_Areal_Rainfall_IWRD.csv    basin-average (Thiessen-weighted) station rainfall, 1986-2023
    Tlawng_Temp_Min_Max_CPC.csv       CPC daily tmax / tmin, basin average
    Solar_Radiation_Tlawng_ERA5.csv   ERA5 surface solar radiation, basin average
Performance/<Var>_Performance.csv     daily RMSE, MAE, R2 of each bias-corrected GCM
                                      vs observations (34 GCMs pr/rsds, 31 tasmax/tasmin)
Historical/<VAR>_Areal_<GCM>_historical_1986_2014.csv
                                      Date, Observed, Raw_Modeled for the top-5 GCMs
Raw_Projections/<Var>/<VAR>_Areal_<ssp>_<GCM>.csv
                                      raw SSP245/SSP585 basin-average series, 2015-2099
Models/<VAR>_Areal_<GCM>_Tuned_XGBoost.pkl
                                      bias-correction models used for the projections
DataGRU/<VAR>_Areal_<ssp>_<GCM>_BiasCorrected.csv
                                      bias-corrected SSP series used as GRU input in the paper
quick_test/                           sample input, expected output, test script
Results/                              outputs of a full run (Table 4 hold-out metrics, etc.)
```

## Method

* **Areal series:** daily NEX-GDDP-CMIP6 fields averaged over the catchment in Google
  Earth Engine (`reduceRegion`, mean, 25 km scale). Units: pr ×86400 → mm/day;
  rsds ×0.024 → kWh/m²/day; tas −273.15 → °C.
* **Rank Score (Eq. 1):** `RS = 1/3·(RMSE_min/RMSE) + 1/3·(MAE_min/MAE) + 1/3·R²`. The top-5
  GCMs per variable are kept, with weights `w = RS/ΣRS`. CESM2, CESM2-WACCM and IITM-ESM
  have no tasmax/tasmin, so 31 GCMs are ranked for temperature.
* **Bias model:** XGBoost regression of the daily bias (Observed − GCM) on the GCM value
  on day t, the GCM value on day t−1, and the day of year. Hyper-parameters come from
  `RandomizedSearchCV` (30 draws, 3-fold CV). Corrected = GCM + predicted bias, with pr
  and rsds clipped at 0.
* **Skill assessment:** hyper-parameters tuned and models fitted on 1986–2006, metrics
  computed on the held-out 2007–2014 period.
* **Ensemble:** RS-weighted mean. HadGEM3-GC31-MM has no SSP245 rsds, so that ensemble
  uses 4 GCMs with renormalised weights.

## Data availability

The NEX-GDDP-CMIP6, CPC and ERA5 data are public. Observed rainfall is the
basin-average of IWRD (Government of Mizoram) station records. Observed discharge (CWC)
is not used in this module.

## License

MIT (see the repository root).

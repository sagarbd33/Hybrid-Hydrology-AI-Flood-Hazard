# Module 2 – HEC-HMS continuous simulation (physics-based component)

Code, model and data for Module 2 of:

> Debbarma, S., Bandyopadhyay, A., Bhadra, A. *A Hybrid Hydrology–AI Framework with
> Terrain-Based Inundation Modeling for Probabilistic Flood Hazard–Exposure Assessment
> under CMIP6 Scenarios.* Submitted to Computers & Geosciences.

Contact: Sagar Debbarma (sagarbd33@gmail.com), NERIST, Arunachal Pradesh, India.

This module covers Section 3.1 of the paper. It contains a calibrated, semi-distributed
HEC-HMS 4.10 model of the Tlawng River Basin above Bairabi: 12 sub-basins with Soil
Moisture Accounting losses, Clark unit hydrograph, linear-reservoir baseflow, Muskingum
routing and Hargreaves evapotranspiration, run at a daily time step.

* **Observed forcing:** 12 IWRD rain gauges and CPC temperature, for calibration
  (2018–2020), validation (2021) and a 1986–2014 run.
* **GCM forcing:** NEX-GDDP-CMIP6 values at the gauges (nearest grid cell) from the
  top-ranked GCM per variable in Module 1 (pr: KACE-1-0-G; tasmax: FGOALS-g3; tasmin:
  INM-CM4-8), bias-corrected with XGBoost. Runs cover the historical period (1984–2014)
  and SSP245/SSP585 (2015–2099).

The simulated outlet discharge is an input to the hybrid HEC-HMS–GRU model (Module 3).

## Requirements

* Python 3.10 or newer: `pip install -r requirements.txt`
* **HEC-HMS 4.10** (free, US Army Corps of Engineers,
  https://www.hec.usace.army.mil/software/hec-hms/). The scripts expect the portable
  Windows build unzipped into `HEC-HMS-4.10/` in this folder. Otherwise set `HMS_DIR` in
  `common.py`. HEC-HMS is needed only for step 5 and is not included in the GitHub
  repository (`.gitignore`).
* The station-wise `.pkl` models were saved with **xgboost 3.4.1**.
* When `pydsstools` is imported it may print a warning that `rasterio` is missing. You
  can ignore it.

## Quick test (about 5 s, HEC-HMS not needed)

```
python quick_test/run_quick_test.py
```

The test checks four things:

1. Table 2 extracted from the `.basin` file matches the expected values.
2. A short HEC-HMS project (Jun–Dec 2017) is built with the correct control and gage
   windows, and the rainfall written to HEC-DSS equals the input CSV.
3. The sub-basin temperature rebuilt from tasmax/tasmin, and the XGBoost bias correction
   of a rainfall sample, match the expected values.
4. The outlet-flow export and the NSE, RMSE, MAE, PBIAS and R² calculations reproduce the
   expected values.

It ends with `QUICK TEST PASSED`.

## Full run

```
cd "<path>/2. HEC-HMS"
python run_module2.py            # steps 0-8
python run_module2.py --no-hms   # all steps except running HEC-HMS (step 5)
```

| Script | Purpose | Output |
|---|---|---|
| `00_check_input_data.py` | Flags station-years with near-zero or missing rainfall | `Results/Station_rainfall_*.csv` |
| `01_parameter_table.py` | Calibrated parameters from the `.basin` file (Table 2) | `Results/Table2_parameter_ranges.csv`, `Results/HMS_parameters_by_element.csv` |
| `02_build_subbasin_temperature.py` | Sub-basin temperature = Thiessen-weighted (tasmax+tasmin)/2 of the gauges | `Data/GCM_Stationwise/reproduced/TASAVG_*_First.csv` |
| `03_bias_correct_stationwise.py` | XGBoost bias correction of station-wise rainfall and sub-basin temperature | `Data/GCM_Stationwise/reproduced/*_BiasCorrected.csv`, `Results/Stationwise_BC_comparison.csv` |
| `04_setup_projects.py` | Builds one HEC-HMS project per scenario and writes the forcing to HEC-DSS | `HMS_Projects/<Scenario>/` |
| `05_run_hms.py` | Runs HEC-HMS (engine `hms.Hms -script`) | `HMS_Projects/<Scenario>/Cal.dss`, `Cal.log` |
| `06_export_results.py` | Exports outlet discharge to CSV | `Results/HMS_Outlet_Q_<Scenario>.csv` |
| `07_evaluate.py` | Calibration 2018–2020 / validation 2021 statistics; annual-maximum summary | `Results/HMS_performance_CalVal.csv`, `Results/HMS_annual_maxima_summary.csv` |
| `08_export_gis.py` | Sub-basin and reach geometry (optional, geopandas) | `GIS/*.gpkg` |

`GEE/extract_stationwise_nexgddp_cmip6.js` is the Google Earth Engine script that
extracts station-wise NEX-GDDP-CMIP6 series. Enter the gauge coordinates before running
it.

## Scenarios (`common.py`)

| Scenario | Rainfall (12 gauges) | Temperature (12 sub-basins) | Period |
|---|---|---|---|
| `Observed_CalVal` | IWRD | CPC | 2017-06-01 – 2022-07-31 (warm-up to Dec 2017; calibration 2018–2020; validation 2021) |
| `Observed_Historical` | IWRD | CPC | 1986–2014 |
| `Historical` | KACE-1-0-G, bias-corrected | FGOALS-g3 / INM-CM4-8, bias-corrected | 1984–2014 (1984–85 warm-up) |
| `Ssp245`, `Ssp585` | KACE-1-0-G, bias-corrected | FGOALS-g3 / INM-CM4-8, bias-corrected | 2015–2099 |

`FORCING_VERSION` in `common.py` selects the bias-corrected GCM forcing:

* `"paper"`: the files used for the published results (`Data/GCM_Stationwise/*_BiasCorrected.csv`).
* `"reproduced"`: the files regenerated in step 3 from the saved models in `Models_StationWise/`.

## Folder contents

```
HMS_Template/            calibrated HEC-HMS 4.10 project (basin, met, control, gage, run files,
                         sub-basin geometry .sqlite)
HMS_Projects/            ready-to-run projects for all scenarios (rebuilt by step 4)
HEC-HMS-4.10/            portable HEC-HMS 4.10 (local use only; not in the GitHub repository)
Models_StationWise/      XGBoost bias-correction models (12 rainfall stations / 12 sub-basins)
Data/Observed/           ObservedPr.csv (IWRD gauges, 1986-2023), ObservedTemp.csv (CPC, 1986-2022)
Data/GCM_Stationwise/    raw and bias-corrected station-wise GCM series
Data/Temperature_station_weights.csv   Thiessen weights for the sub-basin temperature
GEE/                     Earth Engine extraction script
quick_test/              test script, sample input and expected output
```

## Data availability

Observed discharge at Bairabi (Central Water Commission) is restricted and **not
included**. To compute the calibration and validation statistics, place
`Data/Observed/Bairabi_Discharge_CWC.csv` (columns `Date, Bairabi`, m³/s) and re-run
steps 4–7. The 30 m SRTM DEM used for basin delineation is public (USGS EarthExplorer).

## License

MIT (see repository root). HEC-HMS is distributed by USACE under its own terms of use.

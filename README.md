# Hybrid Hydrology–AI Framework for Probabilistic Flood Hazard under CMIP6 Scenarios

Code, trained models and data for:

> Debbarma, S., Bandyopadhyay, A., Bhadra, A. *A Hybrid Hydrology–AI Framework with
> Terrain-Based Inundation Modeling for Probabilistic Flood Hazard–Exposure Assessment
> under CMIP6 Scenarios.* Submitted to *Computers & Geosciences*.

**Study area:** Tlawng River basin above the Bairabi gauge, Mizoram, India.

**Contact:** Sagar Debbarma (sagarbd33@gmail.com), Department of Agricultural Engineering,
North Eastern Regional Institute of Science and Technology (NERIST), Arunachal Pradesh, India.

## Workflow

The framework is split into five modules, each in its own folder. Each module's outputs
are the next module's inputs.

```
1. GCM ranking and bias correction    NEX-GDDP-CMIP6 -> top-ranked GCMs, XGBoost bias correction
        |  bias-corrected rainfall, temperature, solar radiation
        v
2. HEC-HMS                            calibrated SMA/Clark/Muskingum model, historical + SSP245/SSP585 runs
        |  simulated daily discharge
        v
3. Hybrid HEC-HMS–GRU                 GRU post-processor of HEC-HMS discharge (hybrid vs standalone)
        |  projected daily discharge at Bairabi, 1986-2014 and 2015-2099
        v
4. XGBoost rating curve               discharge -> water level (least squares vs XGBoost)
        |  projected daily water level
        v
5. Flood frequency analysis           Mann-Kendall / Sen's slope, L-moment frequency analysis,
                                      Gumbel design stages for hazard classes H1-H4
```

| Module | Paper sections | Tables / figures | Main language / tools |
|---|---|---|---|
| [1. GCM ranking and bias correction](1.%20GCM%20ranking%20and%20bias%20correction) | 2.3, 4.1 | Eq. 1, Table 4 | Python (XGBoost), Google Earth Engine (JavaScript) |
| [2. HEC-HMS](2.%20HEC-HMS) | 3.1 | Table 2 | Python, HEC-HMS 4.10, HEC-DSS (pydsstools) |
| [3. Hybrid HEC-HMS–GRU](3.%20Hybrid%20HEC-HMS%E2%80%93GRU) | 3.1, 4.2, 4.4 | Tables 3, 5 | Python (TensorFlow/Keras) |
| [4. XGBoost rating curve](4.%20XGBoost%20rating%20curve) | 3.2, 4.3, 4.4 | Table 6, Fig. 8 | Python (XGBoost, SciPy) |
| [5. Flood frequency analysis](5.%20Flood%20frequency%20analysis) | 3.3, 4.4, 4.5 | Tables 7, 8, Figs. 11, 12 | Python (SciPy); replaces HEC-SSP 2.3 |

Each module folder has its own `README.md` with the method, run instructions, folder
contents and data notes.

## Installation

Python 3.10–3.13.

```
git clone https://github.com/sagarbd33/Hybrid-Hydrology-AI-Flood-Hazard.git
cd Hybrid-Hydrology-AI-Flood-Hazard
pip install -r requirements.txt
```

Each module also has its own `requirements.txt` if you only need one module.

HEC-HMS 4.10 (free from the U.S. Army Corps of Engineers) is needed only to re-run the
hydrological simulations of Module 2. It is not included in the repository; see
`2. HEC-HMS/README.md`.

## Quick test (about 1 minute)

All five quick tests at once:

```
python run_all_quick_tests.py
```

Or one module:

```
cd "3. Hybrid HEC-HMS–GRU"
python quick_test/run_quick_test.py
```

Each test uses only data in this repository, needs no restricted data and no HEC-HMS
installation, and ends with `QUICK TEST PASSED`. They check:

* **Module 1:** the Rank Score (Eq. 1) gives the top-5 GCMs used in the paper; the saved
  XGBoost models reproduce a 2015–2016 SSP585 bias-corrected sample.
* **Module 2:** Table 2 from the calibrated `.basin` file; building a short HEC-HMS project
  and writing HEC-DSS forcing; station-wise forcing and bias correction; result export and
  statistics.
* **Module 3:** the trained hybrid GRU matches Table 3 and reproduces the 2015–2016 SSP585
  discharge used in the paper; the standalone GRU reproduces a 2021 sample.
* **Module 4:** rating-curve fitting, and the saved models reproduce the stored stages.
* **Module 5:** L-moment estimators against the HEC-SSP report; paper Table 7 and
  Gumbel parameters.

## Full run

Run the modules in order. Each folder has a `run_moduleN.py` that runs all of that
module's steps:

```
cd "1. GCM ranking and bias correction" && python run_module1.py
cd "../2. HEC-HMS"                       && python run_module2.py
cd "../3. Hybrid HEC-HMS–GRU"            && python run_module3.py
cd "../4. XGBoost rating curve"          && python run_module4.py
cd "../5. Flood frequency analysis"      && python run_module5.py
```

* All paths are relative to each module folder.
* When a sibling module is present, its `Results/` are read directly. Otherwise the copy
  in `Data/Inputs/` is used, so every module also runs on its own.
* Steps that need restricted observations (model training and evaluation against observed
  discharge or stage) print a message and are skipped when those data are absent.

## Data

| Data | Source | In repository |
|---|---|---|
| NEX-GDDP-CMIP6 daily pr, tasmax, tasmin, rsds (basin-average and station-wise) | NASA, via Google Earth Engine (scripts in `GEE/`) | Yes |
| Station rainfall, 12 gauges | Irrigation & Water Resources Department (IWRD), Government of Mizoram | Yes |
| Daily min/max temperature | NOAA CPC Global Unified Temperature | Yes |
| Solar radiation | ERA5 (ECMWF) | Yes |
| Basin and sub-basin shapefiles, HEC-HMS basin model | Authors (derived from SRTM DEM) | Yes |
| Trained models (XGBoost, GRU, scalers), model outputs | Authors | Yes |
| Observed daily discharge and water level at Bairabi | Central Water Commission (CWC), India | **No (restricted)** |

The CWC observations are restricted and cannot be redistributed. Every folder that would
hold them (`Data/restricted/`, `Data/Observed/Bairabi_Discharge_CWC.csv`) is excluded by
`.gitignore`. Researchers can request the data from CWC; the authors can share them
with CWC approval. All results that depend on them (trained models, performance tables,
projections) are included, so every downstream step can be reproduced without them.

## Not included

The terrain-based inundation mapping and the exposure analysis (Sections 3.4, 3.5, 4.6 and
4.7) were carried out in ArcGIS Pro with standard Spatial Analyst tools, as described in the
paper. They are not part of this repository.

## Citation

See `CITATION.cff`, or cite the article above.

## License

MIT; see `LICENSE`. Third-party data keep their own terms of use.

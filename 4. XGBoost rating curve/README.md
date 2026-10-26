# Module 4 – Discharge–stage relationship at Bairabi

Code, models and data for Module 4 of:

> Debbarma, S., Bandyopadhyay, A., Bhadra, A. *A Hybrid Hydrology–AI Framework with
> Terrain-Based Inundation Modeling for Probabilistic Flood Hazard–Exposure Assessment
> under CMIP6 Scenarios.* Submitted to Computers & Geosciences.

Contact: Sagar Debbarma (sagarbd33@gmail.com), NERIST, Arunachal Pradesh, India.

This module covers Sections 3.2, 4.3 and 4.4 of the paper. It has two parts:

1. **Rating curves (Table 6, Fig. 8):** a least-squares power law `H = H0 + c·Q^m` and an
   XGBoost regression predict the water level at Bairabi from discharge.
2. **Stage projections (Section 4.4):** converts the projected daily discharge of the
   hybrid HEC-HMS–GRU model (Module 3) into daily water levels for the historical baseline
   and SSP245/SSP585. These feed the flood frequency analysis (Module 5).

## Requirements

Python 3.10 or newer: `pip install -r requirements.txt` (numpy, pandas, scipy,
scikit-learn, xgboost, matplotlib).

## Quick test (a few seconds, no restricted data needed)

```
python quick_test/run_quick_test.py
```

The test checks four things:

1. Power-law fitting recovers the parameters of a synthetic curve.
2. The saved XGBoost rating curve gives the expected stages on a discharge grid.
3. The saved stage-projection model gives the expected 2015–2016 SSP585 stages.
4. The statistics are correct.

It ends with `QUICK TEST PASSED`.

## Full run

```
cd "<path>/4. XGBoost rating curve"
python run_module4.py
```

| Script | Purpose | Output |
|---|---|---|
| `01_rating_curves.py --data paper` | Table 6: least squares vs XGBoost on the data used in the paper | `Results/Table6_rating_curves_paper.csv`, `Results/Figures/Fig8_rating_curves_paper.png`, `Models/RatingCurve_*_paper.json` |
| `01_rating_curves.py --data observed` | The same two models fitted to observed discharge–stage pairs (2018–2021) | `*_observed.*` |
| `02_stage_projection.py --method paper` | Daily stage projections with the method used in the paper | `Results/Stage_<Scenario>_paper.csv`, `Results/Stage_summary_paper.csv` |
| `02_stage_projection.py --method observed` | Daily stage projections with the observed power-law rating curve | `Results/Stage_<Scenario>_observed.csv` |

## Methods

**Data used in the paper (`--data paper`).** Daily discharge from the hybrid
HEC-HMS–GRU model of the SSP245 run, paired with observed stage at Bairabi on the 1,305
days with a stage reading (June 2017 – July 2022).

* Split by year: training June 2017 – December 2020 (907 days); testing January 2021 –
  July 2022 (398 days).
* Least squares: `H0` found by a bounded search (−10 m to the minimum stage); `c` and `m`
  from linear regression of `log(H − H0)` on `log(Q)`.
* XGBoost: up to 500 trees, depth 2, learning rate 0.05, subsample 0.9, early stopping
  after 20 rounds. As in the original notebook, the testing period was monitored for early
  stopping (stopped at 112 trees).
* Statistics: NSE, RMSE, MAE, R², PBIAS, and RMSE/PBIAS of the top 10 % of stages.

**Observed pairs (`--data observed`).** Observed discharge and stage, 2018–2021 (1,033
days). Training 2018–2020, testing 2021.

* The power law is fitted by non-linear least squares.
* XGBoost early stopping monitors a random 20 % of the training days, so the test year is
  never used for fitting.

**Stage projections, paper method.** An XGBoost model of the stage–discharge difference,
`stage(t) = Q(t) + f[Q(t), Q(t−1), day-of-year]`.

* Trained on the paired file of each scenario (training data before 2021).
* Hyper-parameters chosen by `RandomizedSearchCV` (30 draws, 3-fold CV).
* The historical run uses the SSP245 model.
* This reproduces the mean annual maximum stages of Section 4.4 to within about 0.1 m.
  The run-to-run difference comes from the XGBoost version.

## Data availability

Observed stage and discharge at Bairabi come from the Central Water Commission (CWC). They
are restricted and **not included** in the public repository (`Data/restricted/`,
`.gitignore`). With CWC approval they can be obtained from the authors. The projected
discharge in `Data/Inputs/` (from Module 3) and all fitted models are included.

```
Data/Inputs/       projected daily discharge (Historical, Ssp245, Ssp585) from Module 3
Data/restricted/   Observed_Stage.csv, Merged_Output_Ssp245/585.csv, Observed_Q_Stage_pairs.csv
Models/            fitted rating curves (.json) and stage-projection models (.json)
docs/              original Colab notebooks (for reference)
```

## License

MIT (see repository root).

# Module 3 – Hybrid HEC-HMS–GRU streamflow model

Code, trained model and data for Module 3 of:

> Debbarma, S., Bandyopadhyay, A., Bhadra, A. *A Hybrid Hydrology–AI Framework with
> Terrain-Based Inundation Modeling for Probabilistic Flood Hazard–Exposure Assessment
> under CMIP6 Scenarios.* Submitted to Computers & Geosciences.

Contact: Sagar Debbarma (sagarbd33@gmail.com), NERIST, Arunachal Pradesh, India.

This module covers Sections 3.1, 4.2 and 4.4 of the paper. A gated recurrent unit (GRU)
network corrects the daily HEC-HMS discharge at Bairabi.

* **Inputs:** the last 15 days of basin-average precipitation, minimum and maximum
  temperature, shortwave radiation, and HEC-HMS simulated discharge.
* **Output:** discharge on the last day of the window.
* **Climate runs:** the trained model is driven by the bias-corrected CMIP6 ensemble
  (Module 1) and the HEC-HMS runs (Module 2). It produces daily discharge for the
  historical baseline (1984–2014) and for SSP245/SSP585 (2015–2099). These series feed
  the rating curve (Module 4).

## Requirements

Python 3.10 or newer: `pip install -r requirements.txt` (TensorFlow ≥ 2.18,
scikit-learn ≥ 1.6, pandas, numpy). The model was saved with Keras 3.8.0 and the scalers
with scikit-learn 1.6.1. The code was tested with TensorFlow 2.21, Keras 3.15 and
scikit-learn 1.9. A GPU is not needed.

## Quick test (about 10 s, no restricted data needed)

```
python quick_test/run_quick_test.py
```

The test checks three things:

1. The trained model's architecture matches Table 3.
2. The hybrid model reproduces the paper's discharge for a 2015–2016 SSP585 sample
   (tolerance 0.01 m³/s).
3. The standalone GRU gives the expected output for a January–March 2021 sample.
4. The performance statistics are correct.

It ends with `QUICK TEST PASSED`.

## Full run

```
cd "<path>/3. Hybrid HEC-HMS–GRU"
python run_module3.py                       # steps 1, 3, 4
python 02_train_gru.py [--kind standalone]  # re-train (optional)
python 05_seed_sensitivity.py               # seed sensitivity (optional, ~15 min)
```

| Script | Purpose | Output |
|---|---|---|
| `01_build_gru_inputs.py` | Builds the GRU input tables from Module 1 (GCM forcing) and Module 2 (HEC-HMS discharge) | `Data/Inputs_rebuilt/`, `Results/GRU_inputs_comparison.csv` |
| `02_train_gru.py` | Trains the hybrid or standalone GRU (Table 3 set-up); `--protocol paper` follows the original notebook exactly, `--protocol clean` keeps the test set out of training | `Models_retrained/` |
| `03_evaluate_gru.py` | Train/test (2018–2020) and validation (2021) statistics for the hybrid model, the standalone GRU and standalone HEC-HMS (Table 5) | `Results/Table5_*.csv`, `Results/Predictions_*.csv` |
| `04_project_future.py` | Daily discharge for the historical baseline, SSP245 and SSP585; mean annual maxima per period | `Results/<Scenario>_EnsembleMean_OptiDischarge.csv`, `Results/Projection_summary.csv` |
| `05_seed_sensitivity.py` | Re-trains with several random seeds and reports the spread of the scores | `Results/Seed_sensitivity*.csv` |

## Model (Table 3)

| Item | Setting |
|---|---|
| Inputs | precip, tmin, tmax, srad, HEC-HMS discharge (hybrid); first four only (standalone) |
| Window | 15 days; target = discharge on the last day of the window |
| Layers | GRU 256 (ReLU, L2 0.01) → Dropout 0.2 → GRU 128 → Dropout 0.2 → GRU 16 → Dropout 0.2 → Dense 1 (SiLU) |
| Training | Nadam, learning rate 0.001, MSE loss, batch 24, ≤ 300 epochs, early stopping on validation loss (patience 20, active from epoch 50, best weights restored) |
| Scaling | MinMax (0–1) for inputs and target |
| Split | 2018–2020 sequences, 70/30 random split (`train_test_split`, `random_state=42`) |
| Validation | Independent year 2021 |

`02_train_gru.py` has two protocols:

* `paper` reproduces the original Colab notebook (`docs/original_gru_training_colab.py`).
  The scalers are fitted on all 2018–2020 sequences, and the 30 % test set also drives
  early stopping.
* `clean` fits the scalers on the training sequences only and stops on 20 % of the
  training sequences, so the test set stays unseen.

The same code trains both models. Only the input table differs (`DATA_FILES` in
`common.py`).

`Models/` holds the trained hybrid and standalone models and scalers used in the paper.
With them, `03_evaluate_gru.py` reproduces every row of Table 5, and
`04_project_future.py` reproduces the projections of Section 4.4. The standalone model
was trained with the same notebook but with at most 150 epochs (best epoch 145).
Re-training (step 2) gives slightly different weights because of random initialisation;
step 5 quantifies this.

## Folder contents

```
Models/                 trained hybrid HEC-HMS-GRU and standalone GRU models (.keras) and
                        their MinMax scalers (.pkl)
Data/Inputs/            GRU inputs for the climate runs (Historical, Ssp245, Ssp585)
Data/restricted/        training (2018-2020) and validation (2021) tables with CWC discharge,
                        hybrid (Training_*, Validation_*) and standalone (Standalone_*)
                        - NOT distributed (see below)
docs/                   original Colab training notebook (for reference)
quick_test/             test script, sample input and expected output
Results/                outputs of the scripts
```

## Data availability

The training and validation tables contain observed discharge at Bairabi from the Central
Water Commission (CWC). These data are restricted and are **not included** in the public
repository (`.gitignore`). With CWC approval they can be obtained from the authors. Place
them as `Data/restricted/Training_2018_2020.csv` and `Data/restricted/Validation_2021.csv`
(columns `Date, precip, tmin, tmax, srad, simdischarge, discharge`) and the standalone
tables `Standalone_Training_2018_2020.csv` / `Standalone_Validation_2021.csv` (same columns
without `simdischarge`) to run steps 2, 3 and 5. The quick test and the projections (step 4) do not need them.

## License

MIT (see repository root).

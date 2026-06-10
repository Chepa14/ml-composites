# Evaluation summary

This folder contains three types of evaluation:

1. **In-sample** — prediction for all points using the already trained model.
2. **Random holdout** — 80/20 split.
3. **GroupKFold by `r`** — the most thesis-friendly test, because the model is evaluated on a value of `r` it did not see during training.

## Recommended numbers for thesis
Use **GroupKFold by `r`** as the main validation result.

## ALL metrics

- In-sample: MAE=0.906980, RMSE=1.294489, R2=0.999089
- Random holdout: MAE=0.471941, RMSE=0.690935, R2=0.999775
- GroupKFold: MAE=1.019974, RMSE=1.779520, R2=0.998279

## Files

- `metrics_in_sample.json` — metrics on the full training grid
- `metrics_random_holdout.json` — metrics for 80/20 split
- `metrics_group_kfold.json` — main cross-validation report
- `predictions_*.csv` — true values, predictions, absolute and relative errors
- `parity_*.png` — ANSYS vs model comparison
- `hist_relative_error_*.png` — relative error histograms
- `mae_vs_r.png` — average error by held-out group
- `slice_*.png` — example curves for thesis figures
# ML Composites

This repository contains experiments for predicting effective thermal properties of a hexagonal fiber-reinforced composite with a small multilayer perceptron (MLP) trained on simulation data.

The project takes geometric and material parameters as input and predicts two effective coefficients:

- `Ky`
- `Kz`

The main use case is replacing repeated finite-element or APDL-based calculations with a lightweight surrogate model that can be trained once and then used for fast inference.

## Problem setup

The datasets in this repository describe composite configurations using:

- `r` — fiber radius or geometric parameter
- `Kf` — fiber conductivity
- `Km` — matrix conductivity
- `PSI` — derived volumetric fraction-like parameter

Target values:

- `Ky`
- `Kz`

In the current training configuration, the default feature set is:

- `r`
- `Kf`
- `Km`

`PSI` can optionally be included as an additional feature through configuration.

## Approach

The training pipeline is intentionally simple:

1. Load a tabular dataset from CSV or TXT.
2. Standardize input features and targets with `StandardScaler`.
3. Train a feed-forward PyTorch MLP with two hidden layers.
4. Save:
   - model weights
   - input scaler
   - target scaler
5. Reuse the saved artifacts for evaluation and single-point prediction.

The core network used in `v1` is:

- input layer: 3 or 4 features
- hidden layer 1: 64 units + ReLU
- hidden layer 2: 64 units + ReLU
- output layer: 2 values (`Ky`, `Kz`)

Default hyperparameters:

- epochs: `2500`
- learning rate: `1e-3`
- seed: `42` in `v1`, `41` in `v2`

## Repository structure

- [requirements.txt](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/requirements.txt?type=file&root=%252F)  
  Python dependencies.

- [v1](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1?type=directory&root=%252F)  
  Main working implementation.

- [v2](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v2?type=directory&root=%252F)  
  Partial refactor with separated `core`, `utils`, and `datasets` modules.

## `v1` contents

`v1` is the most complete and operational part of the repository.

Key files:

- [train_full_model.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/train_full_model.py?type=file&root=%252F)  
  Trains the MLP on the full dataset and saves artifacts to `trained_model/`.

- [predict_with_model.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/predict_with_model.py?type=file&root=%252F)  
  Loads the saved model and predicts `Ky`, `Kz` for one manually specified input point.

- [evaluate_mmc_model.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/evaluate_mmc_model.py?type=file&root=%252F)  
  Evaluation pipeline for saved models on a larger dataset, including metric calculation and plot generation.

- [evaluate_mmc_model_v2.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/evaluate_mmc_model_v2.py?type=file&root=%252F)  
  Extended evaluation script with:
  - in-sample evaluation
  - random holdout split
  - `GroupKFold` by `r`
  - `GroupKFold` by combined `Kf`/`Km` groups
  - parity plots
  - relative error histograms
  - error-vs-group plots
  - CSV and JSON reports

- [dataset_generation.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/dataset_generation.py?type=file&root=%252F)  
  Generates APDL input text for simulation batches used to produce training data.

- [config.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/config.py?type=file&root=%252F)  
  Central training configuration for `v1`.

- [model.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/model.py?type=file&root=%252F)  
  MLP definition.

Datasets in `v1`:

- [variants_144_v2.csv](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/variants_144_v2.csv?type=file&root=%252F)  
  Compact dataset with 144 parameter combinations.

- [variants_1000_GEX.csv](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/variants_1000_GEX.csv?type=file&root=%252F)  
  Larger evaluation dataset used by the evaluation scripts.

- [Gexagonal_with_meta.txt](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/Gexagonal_with_meta.txt?type=file&root=%252F)  
  Additional source data / simulation-related artifact.

## `v2` contents

`v2` appears to be an in-progress cleanup of the original codebase.

Current structure:

- [core/config.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v2/core/config.py?type=file&root=%252F)
- [core/model.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v2/core/model.py?type=file&root=%252F)
- [utils/utils.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v2/utils/utils.py?type=file&root=%252F)
- [utils/dataset_generation.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v2/utils/dataset_generation.py?type=file&root=%252F)
- [train_full_model.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v2/train_full_model.py?type=file&root=%252F)

At the moment, `v2` should be treated as a refactor branch rather than the primary documented workflow. `v1` is the safer entry point for training, evaluation, and prediction.

## Installation

Use Python 3.10+ if possible.

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies currently listed:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `torch`

## Training

The training scripts use relative paths, so run them from the version directory they belong to.

Example for `v1`:

```bash
cd v1
python train_full_model.py
```

This creates:

- `trained_model/model_mlp.pth`
- `trained_model/x_scaler.joblib`
- `trained_model/y_scaler.joblib`

Main configuration options are defined in [config.py](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/config.py?type=file&root=%252F):

- dataset path
- output directory
- feature selection (`USE_PSI_AS_FEATURE`)
- random seed
- MLP hyperparameters

## Prediction

For single-case prediction in `v1`:

```bash
cd v1
python predict_with_model.py
```

Edit the constants near the top of the file before running:

- `R`
- `KF`
- `KM`
- optionally `PSI`

The script prints the input values and predicted `Ky` / `Kz`.

## Evaluation

The repository includes a more rigorous validation workflow than simply checking the model on the same training points.

Recommended evaluation modes:

1. In-sample evaluation  
   Confirms the trained model reproduces the training grid.

2. Random holdout (`80/20`)  
   Gives a baseline estimate of generalization.

3. `GroupKFold` by `r`  
   This is the most thesis-relevant evaluation because all samples with one radius are excluded from training and used only for testing. That checks interpolation across an unseen geometric parameter.

The extended evaluation script:

```bash
cd v1
python evaluate_mmc_model_v2.py
```

It generates artifacts in `evaluation_artifacts/`, including:

- metrics in JSON format
- predictions in CSV format
- parity plots for `Ky` and `Kz`
- relative error histograms
- mean absolute error versus group
- slice plots for selected parameter combinations

An example summary is available in [README.md](air-file://9rifcnf3qalark3d5ufl/Users/chepa/Projects/ml-composites/v1/evaluation_artifacts/README.md?type=file&root=%252F).

## Metrics

The evaluation scripts compute the following metrics for `Ky`, `Kz`, and the combined output:

- `MAE`
- `RMSE`
- `R2`
- `MAPE (%)`
- `MaxAE`

## Data generation

The dataset generation scripts build APDL batch files for hexagonal-cell thermal simulations. These scripts are used to generate simulation cases across combinations of:

- radius `r`
- fiber conductivity `Kf`
- matrix conductivity `Km`

The generated simulations compute and export:

- `Ky`
- `Kz`
- `PSI`

This makes the repository a hybrid workflow:

1. generate or collect simulation data
2. train a surrogate ML model
3. evaluate generalization
4. use the trained model for fast predictions

## Notes and current limitations

- The project is built around local scripts rather than a packaged Python module.
- Most scripts depend on being run from their own directory because dataset paths are relative.
- `v1` is the most complete workflow.
- `v2` is a partial refactor and may require path/import cleanup before being used as the default version.
- There is no automated test suite in the repository yet.

## Suggested next improvements

- make dataset paths configurable from the command line
- package shared logic to avoid code duplication between scripts
- add a reproducible environment file with pinned dependency versions
- add tests for data loading, scaling, model serialization, and inference
- unify `v1` and `v2` into a single maintained workflow

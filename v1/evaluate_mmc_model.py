import os
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


# =========================
# Config
# =========================
DATA_PATH = "variants_1000_GEX.csv"
TRAINED_MODEL_DIR = "trained_model"
OUT_DIR = "evaluation_artifacts"
USE_PSI_AS_FEATURE = False
TARGET_COLS = ["Ky", "Kz"]
FEATURE_COLS = ["r", "Kf", "Km"] + (["PSI"] if USE_PSI_AS_FEATURE else [])
GROUP_COL = "r"  # thesis-friendly: leave one radius out
RANDOM_STATE = 42
EPOCHS = 2500
LR = 1e-3
H1 = 64
H2 = 64
DEVICE = "cpu"


# =========================
# Helpers
# =========================
@dataclass
class MetricSet:
    MAE: float
    RMSE: float
    R2: float
    MAPE_percent: float
    MaxAE: float


class MLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)



def calc_metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> MetricSet:
    return MetricSet(
        MAE=float(mean_absolute_error(y_true, y_pred)),
        RMSE=float(math.sqrt(mean_squared_error(y_true, y_pred))),
        R2=float(r2_score(y_true, y_pred)),
        MAPE_percent=safe_mape(y_true, y_pred),
        MaxAE=float(np.max(np.abs(y_true - y_pred))),
    )



def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str]) -> dict:
    result = {}

    for idx, col in enumerate(target_cols):
        result[col] = asdict(calc_metric_set(y_true[:, idx], y_pred[:, idx]))

    result["ALL"] = asdict(calc_metric_set(y_true.reshape(-1), y_pred.reshape(-1)))
    return result



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = set(FEATURE_COLS + TARGET_COLS + [GROUP_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {df.columns.tolist()}")

    for c in set(FEATURE_COLS + TARGET_COLS + [GROUP_COL]):
        df[c] = df[c].astype(float)

    return df



def fit_model(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[MLP, StandardScaler, StandardScaler]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_s = x_scaler.transform(X_train)
    y_train_s = y_scaler.transform(y_train)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32, device=DEVICE)

    model = MLP(in_dim=X_train.shape[1], h1=H1, h2=H2, out_dim=y_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    return model, x_scaler, y_scaler



def predict_with_model(model: MLP, x_scaler: StandardScaler, y_scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_s = x_scaler.transform(X)
    X_t = torch.tensor(X_s, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        y_pred_s = model(X_t).cpu().numpy()
    return y_scaler.inverse_transform(y_pred_s)



def load_saved_model(model_dir: str) -> tuple[MLP, object, object, list[str], list[str]]:
    chk = torch.load(Path(model_dir) / "model_mlp.pth", map_location=DEVICE)
    x_scaler = joblib.load(Path(model_dir) / "x_scaler.joblib")
    y_scaler = joblib.load(Path(model_dir) / "y_scaler.joblib")

    model = MLP(
        in_dim=chk["in_dim"],
        h1=chk["h1"],
        h2=chk["h2"],
        out_dim=chk["out_dim"],
    ).to(DEVICE)
    model.load_state_dict(chk["state_dict"])
    model.eval()

    return model, x_scaler, y_scaler, chk["feature_cols"], chk["target_cols"]



def add_prediction_columns(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for i, target in enumerate(TARGET_COLS):
        out[f"{target}_pred"] = y_pred[:, i]
        out[f"{target}_abs_err"] = np.abs(out[target] - out[f"{target}_pred"])
        out[f"{target}_rel_err_percent"] = (
            np.abs(out[target] - out[f"{target}_pred"]) / np.maximum(np.abs(out[target]), 1e-9) * 100.0
        )
    return out



def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def plot_parity(df_pred: pd.DataFrame, out_path: Path) -> None:
    for target in TARGET_COLS:
        plt.figure(figsize=(6.5, 6.0))
        x = df_pred[target].values
        y = df_pred[f"{target}_pred"].values
        plt.scatter(x, y, alpha=0.8)
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel(f"True {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"Parity plot for {target}")
        plt.tight_layout()
        plt.savefig(out_path / f"parity_{target}.png", dpi=220)
        plt.close()



def plot_relative_error_hist(df_pred: pd.DataFrame, out_path: Path) -> None:
    for target in TARGET_COLS:
        plt.figure(figsize=(7.0, 4.8))
        vals = df_pred[f"{target}_rel_err_percent"].values
        plt.hist(vals, bins=24)
        plt.xlabel(f"Relative error of {target}, %")
        plt.ylabel("Count")
        plt.title(f"Relative error distribution for {target}")
        plt.tight_layout()
        plt.savefig(out_path / f"hist_relative_error_{target}.png", dpi=220)
        plt.close()



def plot_error_vs_group(df_pred: pd.DataFrame, out_path: Path) -> None:
    grouped = df_pred.groupby(GROUP_COL)[[f"{t}_abs_err" for t in TARGET_COLS]].mean().reset_index()

    plt.figure(figsize=(7.5, 4.8))
    for target in TARGET_COLS:
        plt.plot(grouped[GROUP_COL], grouped[f"{target}_abs_err"], marker="o", label=target)
    plt.xlabel(GROUP_COL)
    plt.ylabel("Mean absolute error")
    plt.title(f"Mean absolute error vs {GROUP_COL}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"mae_vs_{GROUP_COL}.png", dpi=220)
    plt.close()



def plot_slice_examples(df_pred: pd.DataFrame, out_path: Path) -> None:
    # thesis-friendly slices: fixed matrix conductivity, compare trend vs fiber radius
    selected_km = sorted(df_pred["Km"].unique())[1] if len(df_pred["Km"].unique()) > 1 else sorted(df_pred["Km"].unique())[0]

    for target in TARGET_COLS:
        plt.figure(figsize=(7.5, 4.8))
        for kf in sorted(df_pred["Kf"].unique()):
            part = df_pred[(df_pred["Km"] == selected_km) & (df_pred["Kf"] == kf)].sort_values("r")
            if part.empty:
                continue
            plt.plot(part["r"], part[target], marker="o", label=f"True, Kf={kf}")
            plt.plot(part["r"], part[f"{target}_pred"], marker="x", linestyle="--", label=f"Pred, Kf={kf}")
        plt.xlabel("r")
        plt.ylabel(target)
        plt.title(f"{target} vs r at Km={selected_km}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path / f"slice_{target}_Km_{selected_km}.png", dpi=220)
        plt.close()



def evaluate_saved_model_on_all_data(df: pd.DataFrame, out_dir: Path) -> dict:
    model, x_scaler, y_scaler, feature_cols, target_cols = load_saved_model(TRAINED_MODEL_DIR)

    if feature_cols != FEATURE_COLS:
        raise ValueError(f"Feature mismatch. Saved model expects {feature_cols}, script uses {FEATURE_COLS}")
    if target_cols != TARGET_COLS:
        raise ValueError(f"Target mismatch. Saved model expects {target_cols}, script uses {TARGET_COLS}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_true = df[TARGET_COLS].values.astype(np.float32)
    y_pred = predict_with_model(model, x_scaler, y_scaler, X)

    df_pred = add_prediction_columns(df, y_pred)
    df_pred.to_csv(out_dir / "predictions_in_sample.csv", index=False)

    metrics = calc_metrics(y_true, y_pred, TARGET_COLS)
    save_json(out_dir / "metrics_in_sample.json", metrics)

    plot_parity(df_pred, out_dir)
    plot_relative_error_hist(df_pred, out_dir)
    plot_error_vs_group(df_pred, out_dir)
    plot_slice_examples(df_pred, out_dir)

    return metrics



def evaluate_random_holdout(df: pd.DataFrame, out_dir: Path) -> dict:
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=RANDOM_STATE
    )

    model, x_scaler, y_scaler = fit_model(X_train, y_train, seed=RANDOM_STATE)
    y_pred = predict_with_model(model, x_scaler, y_scaler, X_test)

    df_pred = add_prediction_columns(df.loc[idx_test].reset_index(drop=True), y_pred)
    df_pred.to_csv(out_dir / "predictions_random_holdout.csv", index=False)

    metrics = calc_metrics(y_test, y_pred, TARGET_COLS)
    save_json(out_dir / "metrics_random_holdout.json", metrics)

    return metrics



def evaluate_group_kfold(df: pd.DataFrame, out_dir: Path) -> dict:
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)
    groups = df[GROUP_COL].values.astype(float)

    gkf = GroupKFold(n_splits=len(np.unique(groups)))

    all_pred_rows = []
    fold_summary = []

    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        model, x_scaler, y_scaler = fit_model(X[train_idx], y[train_idx], seed=RANDOM_STATE + fold_id)
        y_pred = predict_with_model(model, x_scaler, y_scaler, X[test_idx])

        fold_df = add_prediction_columns(df.iloc[test_idx].reset_index(drop=True), y_pred)
        fold_df["fold"] = fold_id
        fold_df["held_out_group"] = groups[test_idx]
        all_pred_rows.append(fold_df)

        fold_metrics = calc_metrics(y[test_idx], y_pred, TARGET_COLS)
        fold_summary.append(
            {
                "fold": fold_id,
                "held_out_group_values": sorted(np.unique(groups[test_idx]).tolist()),
                "metrics": fold_metrics,
            }
        )

    df_pred = pd.concat(all_pred_rows, ignore_index=True)
    df_pred.to_csv(out_dir / "predictions_group_kfold.csv", index=False)

    metrics_overall = calc_metrics(df_pred[TARGET_COLS].values, df_pred[[f"{t}_pred" for t in TARGET_COLS]].values, TARGET_COLS)
    report = {
        "scheme": f"GroupKFold by {GROUP_COL}",
        "n_splits": int(len(np.unique(groups))),
        "overall_metrics": metrics_overall,
        "folds": fold_summary,
    }
    save_json(out_dir / "metrics_group_kfold.json", report)

    plot_parity(df_pred, out_dir)
    plot_relative_error_hist(df_pred, out_dir)
    plot_error_vs_group(df_pred, out_dir)
    plot_slice_examples(df_pred, out_dir)

    return report



def make_summary_md(out_dir: Path, in_sample: dict, random_holdout: dict, group_report: dict) -> None:
    lines = [
        "# Evaluation summary",
        "",
        "This folder contains three types of evaluation:",
        "",
        "1. **In-sample** — prediction for all points using the already trained model.",
        "2. **Random holdout** — 80/20 split.",
        f"3. **GroupKFold by `{GROUP_COL}`** — the most thesis-friendly test, because the model is evaluated on a value of `{GROUP_COL}` it did not see during training.",
        "",
        "## Recommended numbers for thesis",
        "Use **GroupKFold by `r`** as the main validation result.",
        "",
        "## ALL metrics",
        "",
        f"- In-sample: MAE={in_sample['ALL']['MAE']:.6f}, RMSE={in_sample['ALL']['RMSE']:.6f}, R2={in_sample['ALL']['R2']:.6f}",
        f"- Random holdout: MAE={random_holdout['ALL']['MAE']:.6f}, RMSE={random_holdout['ALL']['RMSE']:.6f}, R2={random_holdout['ALL']['R2']:.6f}",
        f"- GroupKFold: MAE={group_report['overall_metrics']['ALL']['MAE']:.6f}, RMSE={group_report['overall_metrics']['ALL']['RMSE']:.6f}, R2={group_report['overall_metrics']['ALL']['R2']:.6f}",
        "",
        "## Files",
        "",
        "- `metrics_in_sample.json` — metrics on the full training grid",
        "- `metrics_random_holdout.json` — metrics for 80/20 split",
        "- `metrics_group_kfold.json` — main cross-validation report",
        "- `predictions_*.csv` — true values, predictions, absolute and relative errors",
        "- `parity_*.png` — ANSYS vs model comparison",
        "- `hist_relative_error_*.png` — relative error histograms",
        f"- `mae_vs_{GROUP_COL}.png` — average error by held-out group",
        "- `slice_*.png` — example curves for thesis figures",
    ]

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    out_dir = ensure_dir(OUT_DIR)
    df = load_dataset(DATA_PATH)

    in_sample_metrics = evaluate_saved_model_on_all_data(df, out_dir)
    random_holdout_metrics = evaluate_random_holdout(df, out_dir)
    group_report = evaluate_group_kfold(df, out_dir)

    make_summary_md(out_dir, in_sample_metrics, random_holdout_metrics, group_report)

    print("Done. Artifacts saved to:", out_dir.resolve())
    print("Recommended for thesis: metrics_group_kfold.json + parity/histogram figures.")


if __name__ == "__main__":
    main()

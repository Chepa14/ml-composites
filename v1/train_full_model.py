import os
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================
TXT_PATH = "variants_144_v2.csv"
OUT_DIR = "trained_model"
os.makedirs(OUT_DIR, exist_ok=True)

USE_PSI_AS_FEATURE = False
SEED = 42

# MLP hyperparams
EPOCHS = 2500
LR = 1e-3
H1 = 64
H2 = 64

# =========================
# Load data
# =========================
df = pd.read_csv(TXT_PATH, engine="python")
# df = pd.read_csv(TXT_PATH, sep=r"\s+", engine="python")
df.columns = [c.strip() for c in df.columns]

required = {"r", "Kf", "Km", "Ky", "Kz", "PSI"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

for c in ["r", "Kf", "Km", "Ky", "Kz", "PSI"]:
    df[c] = df[c].astype(float)

feature_cols = ["r", "Kf", "Km"] + (["PSI"] if USE_PSI_AS_FEATURE else [])
target_cols = ["Ky", "Kz"]

X = df[feature_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

print(f"Loaded rows: {len(df)}")
print(f"Features: {feature_cols}")
print(f"Targets: {target_cols}")

# =========================
# Scaling
# =========================
x_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)

X_s = x_scaler.transform(X)
y_s = y_scaler.transform(y)

X_t = torch.tensor(X_s, dtype=torch.float32)
y_t = torch.tensor(y_s, dtype=torch.float32)

# =========================
# Model
# =========================
torch.manual_seed(SEED)
np.random.seed(SEED)

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

model = MLP(in_dim=X.shape[1], h1=H1, h2=H2, out_dim=y.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# =========================
# Train on ALL data
# =========================
model.train()
for ep in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    loss.backward()
    opt.step()

    if ep % 250 == 0:
        print(f"Epoch {ep:4d} | Loss: {loss.item():.6f}")

# =========================
# Save artifacts
# =========================
torch.save(
    {
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "h1": H1,
        "h2": H2,
        "in_dim": int(X.shape[1]),
        "out_dim": int(y.shape[1]),
    },
    os.path.join(OUT_DIR, "model_mlp.pth")
)

joblib.dump(x_scaler, os.path.join(OUT_DIR, "x_scaler.joblib"))
joblib.dump(y_scaler, os.path.join(OUT_DIR, "y_scaler.joblib"))

print(f"\nSaved to folder: {OUT_DIR}")
print("- model_mlp.pth")
print("- x_scaler.joblib")
print("- y_scaler.joblib")

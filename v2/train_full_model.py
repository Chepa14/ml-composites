import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from core.config import *
from core.model import MLP
from utils.utils import read_csv


# =========================
# Load data
# =========================
df = read_csv(TXT_PATH)

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

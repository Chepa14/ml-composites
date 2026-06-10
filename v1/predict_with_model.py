import joblib
import numpy as np
import torch
import torch.nn as nn

# =========================
# Config: change inputs here
# =========================
MODEL_DIR = "trained_model"

# ВХІДНІ ЗНАЧЕННЯ (міняєш тут)
R = 0.50
KF = 70.0
KM = 70.0

# Якщо в тренуванні USE_PSI_AS_FEATURE=True, тоді треба задати PSI теж
USE_PSI_AS_FEATURE = False
PSI = 0.196  # приклад, якщо треба

# =========================
# Load model + scalers
# =========================
chk = torch.load(f"{MODEL_DIR}/model_mlp.pth", map_location="cpu")
feature_cols = chk["feature_cols"]
target_cols = chk["target_cols"]

x_scaler = joblib.load(f"{MODEL_DIR}/x_scaler.joblib")
y_scaler = joblib.load(f"{MODEL_DIR}/y_scaler.joblib")

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

model = MLP(in_dim=chk["in_dim"], h1=chk["h1"], h2=chk["h2"], out_dim=chk["out_dim"])
model.load_state_dict(chk["state_dict"])
model.eval()

# =========================
# Build input in correct order
# =========================
values = {"r": R, "Kf": KF, "Km": KM, "PSI": PSI}

# строгий порядок, як в тренуванні:
x = np.array([[values[c] for c in feature_cols]], dtype=np.float32)

x_s = x_scaler.transform(x)
x_t = torch.tensor(x_s, dtype=torch.float32)

with torch.no_grad():
    y_pred_s = model(x_t).numpy()

y_pred = y_scaler.inverse_transform(y_pred_s)[0]

# =========================
# Print results
# =========================
print("Input:")
for c in feature_cols:
    print(f"  {c} = {values[c]}")

print("\nPrediction:")
for name, val in zip(target_cols, y_pred):
    print(f"  {name} = {val:.6f}")

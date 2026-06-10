import os
from abc import ABC

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from config import *

torch.manual_seed(SEED)
np.random.seed(SEED)


class AbstractModel(ABC):
    ...


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

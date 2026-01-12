#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================
# Imports
# ============================================================

import os
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# ============================================================
# Configurações globais
# ============================================================

SEED = 42
BATCH_SIZE = 512
MAX_EVALS = 4

TARGET_COL = "Class"
OUTDIR = "results_lstm_full"
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Leitura dos dados
# ============================================================

def load_csv(path):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada em {path}")
    y = df[TARGET_COL].values.astype(np.float32)
    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    print(f"✔ Loaded {path} | X={X.shape}")
    return X, y


X_train, y_train = load_csv("train_full_50k.csv")
X_val,   y_val   = load_csv("validation_full_50k.csv")
X_test,  y_test  = load_csv("test_full_50k.csv")

INPUT_SIZE = X_train.shape[1]

# ============================================================
# Dataset / DataLoader
# ============================================================

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

# ============================================================
# Modelo LSTM
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden, layers, fc1, fc2, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=int(hidden),
            num_layers=int(layers),
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )

        blocks = []
        dim = int(hidden)

        if fc1 > 0:
            blocks += [nn.Linear(dim, int(fc1)), nn.ReLU()]
            dim = int(fc1)

        if fc2 > 0:
            blocks += [nn.Linear(dim, int(fc2)), nn.ReLU()]
            dim = int(fc2)

        blocks.append(nn.Linear(dim, 1))
        self.head = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,F) -> (B,1,F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.head(out)).squeeze()

# ============================================================
# Treino / Avaliação
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    y_true, y_score = [], []
    for Xb, yb in loader:
        preds = model(Xb.to(DEVICE)).cpu().numpy()
        y_true.append(yb.numpy())
        y_score.append(preds)
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    return (
        roc_auc_score(y_true, y_score),
        average_precision_score(y_true, y_score),
        y_true,
        y_score
    )

# ============================================================
# Espaço de busca Hyperopt (SEGURO)
# ============================================================

HIDDEN_CHOICES = [16, 32, 64]
LAYER_CHOICES  = [1, 2]
FC1_CHOICES    = [0, 32, 64]
FC2_CHOICES    = [0, 32]
EPOCH_CHOICES  = [3, 5]

SPACE = {
    "hidden": hp.choice("hidden", HIDDEN_CHOICES),
    "layers": hp.choice("layers", LAYER_CHOICES),
    "fc1": hp.choice("fc1", FC1_CHOICES),
    "fc2": hp.choice("fc2", FC2_CHOICES),
    "epochs": hp.choice("epochs", EPOCH_CHOICES),
    "dropout": hp.uniform("dropout", 0.1, 0.4),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(3e-3))
}

# ============================================================
# Objective Hyperopt
# ============================================================

def objective(p):

    model = LSTMModel(
        INPUT_SIZE,
        p["hidden"],
        p["layers"],
        p["fc1"],
        p["fc2"],
        p["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    criterion = nn.BCELoss()

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val, y_val)

    for _ in range(p["epochs"]):
        train_epoch(model, train_loader, optimizer, criterion)

    _, pr, _, _ = eval_model(model, val_loader)

    return {"loss": -pr, "status": STATUS_OK}

# ============================================================
# Execução Hyperopt
# ============================================================

trials = Trials()
best = fmin(
    fn=objective,
    space=SPACE,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=trials
)

def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj



with open(f"{OUTDIR}/best_params.json", "w") as f:
    #json.dump(best, f, indent=2)
    json.dump(json_safe(best), f, indent=2)

print("\n🏆 Best Hyperopt Params:", best)
print("💻 Device:", DEVICE)

# ============================================================
# Treino final
# ============================================================

final = {
    "hidden": HIDDEN_CHOICES[best["hidden"]],
    "layers": LAYER_CHOICES[best["layers"]],
    "fc1": FC1_CHOICES[best["fc1"]],
    "fc2": FC2_CHOICES[best["fc2"]],
    "epochs": EPOCH_CHOICES[best["epochs"]],
    "dropout": best["dropout"],
    "lr": best["lr"]
}

final_model = LSTMModel(
    INPUT_SIZE,
    final["hidden"],
    final["layers"],
    final["fc1"],
    final["fc2"],
    final["dropout"]
).to(DEVICE)

optimizer = torch.optim.Adam(final_model.parameters(), lr=final["lr"])
criterion = nn.BCELoss()

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val, y_val)
test_loader  = make_loader(X_test, y_test)

for _ in range(final["epochs"]):
    train_epoch(final_model, train_loader, optimizer, criterion)

torch.save(final_model.state_dict(), f"{OUTDIR}/model.pt")

# ============================================================
# Avaliação final
# ============================================================

def evaluate_full(model, loader, name):
    roc, pr, yt, ys = eval_model(model, loader)
    yp = (ys >= 0.5).astype(int)

    print(f"\n===== {name.upper()} =====")
    print(classification_report(yt, yp, digits=4))
    print(f"ROC-AUC: {roc:.6f}")
    print(f"PR-AUC : {pr:.6f}")

evaluate_full(final_model, train_loader, "train")
evaluate_full(final_model, val_loader, "validation")
evaluate_full(final_model, test_loader, "test")

print("\n✅ Pipeline finalizado com sucesso")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    brier_score_loss, matthews_corrcoef,
    precision_recall_curve, roc_curve
)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# ===================== SETUP =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTDIR = f"results/run_{timestamp}"
os.makedirs(OUTDIR, exist_ok=True)

# ===================== LOAD DATA =====================
def load_split(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y

X_train, y_train = load_split("train_full_50k.csv")
X_val, y_val     = load_split("validation_full_50k.csv")
X_test, y_test   = load_split("test_full_50k.csv")

X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).float().unsqueeze(1).to(device)
X_val_t   = torch.tensor(X_val).to(device)
y_val_t   = torch.tensor(y_val).float().unsqueeze(1).to(device)
X_test_t  = torch.tensor(X_test).to(device)
y_test_t  = torch.tensor(y_test).float().unsqueeze(1).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)

# ===================== MODEL =====================
class FlexibleFCNet(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        layers = []
        prev = input_size
        for s in layer_sizes:
            layers.append(nn.Linear(prev, s))
            layers.append(nn.ReLU())
            prev = s
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x.squeeze(1)))

# ===================== FOCAL LOSS =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

# ===================== EVALUATION =====================
def evaluate_model(model, loader):
    model.eval()
    probs, targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            if xb.ndim != 3:
                continue
            p = model(xb).detach().cpu().numpy().ravel()
            t = yb.cpu().numpy().ravel()
            probs.append(p)
            targets.append(t)

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(targets, preds)
    TN, FP, FN, TP = cm.ravel()
    frr = FN / (FN + TN + 1e-8)
    gmean = np.sqrt(
        (TP / (TP + FN + 1e-8)) *
        (TN / (TN + FP + 1e-8))
    )

    return {
        "roc_auc": roc_auc_score(targets, probs),
        "auc_pr": average_precision_score(targets, probs),
        "brier": brier_score_loss(targets, probs),
        "mcc": matthews_corrcoef(targets, preds),
        "frr": frr,
        "gmean": gmean,
        "cm": cm,
        "probs": probs,
        "targets": targets,
        "preds": preds
    }

# ===================== OBJECTIVE =====================
def objective(params):
    layers = [int(params[f"n_units_{i}"]) for i in range(params["n_layers"])]
    model = FlexibleFCNet(X_train.shape[2], layers).to(device)

    criterion = FocalLoss(alpha=params["alpha"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    train_loader = DataLoader(
        train_ds, batch_size=int(params["batch_size"]),
        shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False
    )

    losses = []
    for ep in range(int(params["epochs"])):
        print (ep,"of",int(params["epochs"]),end="\r")
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    train_metrics = evaluate_model(model, train_loader)
    val_metrics   = evaluate_model(model, val_loader)

    roc_mean = 0.5 * (train_metrics["roc_auc"] + val_metrics["roc_auc"])

    return {
        "loss": -roc_mean,
        "status": STATUS_OK,
        "metrics": {
            "roc_auc_train": train_metrics["roc_auc"],
            "roc_auc_val": val_metrics["roc_auc"],
            "loss_curve": losses
        },
        "params": params
    }

# ===================== SEARCH SPACE =====================
space = {
    "n_layers": hp.choice("n_layers", [3, 4]), #3, 4, 5
    "n_units_0": hp.quniform("n_units_0", 8, 20, 1), # 8, 128, 1
    "n_units_1": hp.quniform("n_units_1", 8, 20, 1),
    "n_units_2": hp.quniform("n_units_2", 8, 20, 1),
    "n_units_3": hp.quniform("n_units_3", 8, 20, 1),
    "n_units_4": hp.quniform("n_units_4", 8, 20, 1),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
    "batch_size": hp.choice("batch_size", [64, 128, 256]),
    "epochs": hp.choice("epochs", [10, 20, 25, 30]), #50, 100, 200, 300
    "alpha": hp.choice("alpha", [1, 2, 10, 100, 1000])
}

# ===================== RUN =====================
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=5, #50,
    trials=trials
)

with open(f"{OUTDIR}/trials.json", "w") as f:
    json.dump(trials.results, f, indent=2)

print("Hyperopt finalizado")
print("Resultados salvos em:", OUTDIR)

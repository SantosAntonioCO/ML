#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================
# Imports
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# ============================================================
# Configurações globais
# ============================================================

SEED = 42
BATCH_SIZE = 512
MAX_EVALS = 20
EPOCHS_HPO = 5
EPOCHS_FINAL = 30

TARGET_COL = "Class"
OUTDIR = "results_fc_hyperopt"
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Utils
# ============================================================

def frr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tn + 1e-12)

# ============================================================
# Leitura dos dados
# ============================================================

def load_csv(path):
    df = pd.read_csv(path)
    y = df[TARGET_COL].values.astype(np.float32)
    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    return X, y


X_train, y_train = load_csv("train_full_50k.csv")
X_val,   y_val   = load_csv("validation_full_50k.csv")
X_test,  y_test  = load_csv("test_full_50k.csv")

INPUT_SIZE = X_train.shape[1]

# ============================================================
# DataLoader
# ============================================================

def make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

# ============================================================
# Modelo FC dinâmico
# ============================================================

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, layers, dropout):
        super().__init__()

        blocks = []
        dim = input_size

        for h in layers:
            blocks.append(nn.Linear(dim, h))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            dim = h

        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()

# ============================================================
# Treino / Avaliação
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    y_true, y_score = [], []
    for xb, yb in loader:
        preds = model(xb.to(DEVICE)).cpu().numpy()
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
# Espaço Hyperopt
# ============================================================

SPACE = {
    "n_layers": hp.choice("n_layers", [3, 4, 5, 6]),
    "hidden": hp.quniform("hidden", 5, 80, 5),
    "dropout": hp.uniform("dropout", 0.0, 0.4),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-1))
}

# ============================================================
# Objective
# ============================================================

def objective(p):

    layers = [int(p["hidden"])] * int(p["n_layers"])

    model = FullyConnectedNet(
        INPUT_SIZE,
        layers,
        p["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    criterion = nn.BCELoss()

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val, y_val)

    for _ in range(EPOCHS_HPO):
        train_epoch(model, train_loader, optimizer, criterion)

    roc, pr, _, _ = eval_model(model, val_loader)

    score = 0.5 * (roc + pr)

    return {
        "loss": -score,
        "status": STATUS_OK,
        "roc_auc": roc,
        "pr_auc": pr,
        "params": {
            "layers": layers,
            "dropout": p["dropout"],
            "lr": p["lr"]
        }
    }

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

# ============================================================
# Salvar trials em CSV
# ============================================================

rows = []
for t in trials.trials:
    rows.append({
        **t["result"]["params"],
        "roc_auc": t["result"]["roc_auc"],
        "pr_auc": t["result"]["pr_auc"],
        "score": -t["result"]["loss"]
    })

df_trials = pd.DataFrame(rows)
df_trials.to_csv(f"{OUTDIR}/hyperopt_trials.csv", index=False)

# ============================================================
# Best params resolvidos
# ============================================================

best_trial = max(rows, key=lambda x: x["score"])

with open(f"{OUTDIR}/best_params.json", "w") as f:
    json.dump(best_trial, f, indent=2)

# ============================================================
# Treino final
# ============================================================

final_model = FullyConnectedNet(
    INPUT_SIZE,
    best_trial["layers"],
    best_trial["dropout"]
).to(DEVICE)

optimizer = torch.optim.Adam(final_model.parameters(), lr=best_trial["lr"])
criterion = nn.BCELoss()

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val, y_val)
test_loader  = make_loader(X_test, y_test)

for _ in range(EPOCHS_FINAL):
    train_epoch(final_model, train_loader, optimizer, criterion)

torch.save(final_model.state_dict(), f"{OUTDIR}/model_fc.pt")

# ============================================================
# Avaliação final completa
# ============================================================

def evaluate_and_save(model, loader, name):
    roc, pr, yt, ys = eval_model(model, loader)
    yp = (ys >= 0.5).astype(int)
    frr = frr_score(yt, yp)

    report = classification_report(yt, yp, digits=5)

    with open(f"{OUTDIR}/report_{name}.txt", "w") as f:
        f.write(report)
        f.write(f"\nROC-AUC: {roc:.6f}")
        f.write(f"\nPR-AUC : {pr:.6f}")
        f.write(f"\nFRR    : {frr:.6f}")

    print(f"\n===== {name.upper()} =====")
    print(report)
    print(f"ROC-AUC: {roc:.6f} | PR-AUC: {pr:.6f} | FRR: {frr:.6f}")

evaluate_and_save(final_model, train_loader, "train")
evaluate_and_save(final_model, val_loader, "validation")
evaluate_and_save(final_model, test_loader, "test")

print("\n Pipeline FC + Hyperopt finalizado com sucesso")

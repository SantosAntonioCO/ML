#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================
# Imports
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve
)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# ============================================================
# Setup
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTDIR = "results_fc_epochs"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Load preprocessed CSVs
# ============================================================

def load_csv(path):
    df = pd.read_csv(path)
    y = df["Class"].values.astype(np.float32)
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    return X, y


X_train, y_train = load_csv("train_full_50k.csv")
X_val,   y_val   = load_csv("validation_full_50k.csv")
X_test,  y_test  = load_csv("test_full_50k.csv")

INPUT_SIZE = X_train.shape[1]

# ============================================================
# DataLoader
# ============================================================

def make_loader(X, y, batch_size, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

# ============================================================
# Modelo FC
# ============================================================

class FlexibleFCNet(nn.Module):
    def __init__(self, input_size, layers):
        super().__init__()
        blocks = []
        dim = input_size
        for h in layers:
            blocks += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        blocks += [nn.Linear(dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze()

# ============================================================
# Métricas
# ============================================================

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    frr = FN / (FN + TN + 1e-12)
    gmean = np.sqrt(
        TP / (TP + FN + 1e-12) *
        TN / (TN + FP + 1e-12)
    )

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "frr": frr,
        "gmean": gmean,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_true, y_pred, digits=4, zero_division=0
        )
    }

# ============================================================
# Avaliação segura (anti-batch bug)
# ============================================================

@torch.no_grad()
def evaluate_model_safe(model, loader):
    model.eval()
    yt, ys = [], []

    for xb, yb in loader:
        preds = model(xb.to(DEVICE)).detach().cpu().numpy().reshape(-1)
        yb = yb.numpy().reshape(-1)

        if preds.shape[0] != yb.shape[0]:
            continue

        ys.append(preds)
        yt.append(yb)

    return np.concatenate(yt), np.concatenate(ys)

# ============================================================
# Treino
# ============================================================

def train_model(model, loader, optimizer, criterion, epochs):
    losses = []
    for ep in range(epochs):
        print (ep,"of",epochs,end="\r")
        model.train()
        epoch_loss = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        losses.append(float(np.mean(epoch_loss)))
    return losses

# ============================================================
# Hyperopt Objective
# ============================================================

def objective(p):

    layers = [int(p[f"n{i}"]) for i in range(p["n_layers"])]
    batch_size = int(p["batch"])
    epochs = int(p["epochs"])

    model = FlexibleFCNet(INPUT_SIZE, layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    criterion = nn.BCELoss()

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_loader(X_val, y_val, batch_size)

    train_model(model, train_loader, optimizer, criterion, epochs)

    yt_tr, ys_tr = evaluate_model_safe(model, train_loader)
    yt_va, ys_va = evaluate_model_safe(model, val_loader)

    score = 0.5 * (
        roc_auc_score(yt_tr, ys_tr) +
        roc_auc_score(yt_va, ys_va)
    )

    return {
        "loss": -score,
        "status": STATUS_OK,
        "metrics": {
            "roc_auc_train": roc_auc_score(yt_tr, ys_tr),
            "roc_auc_val": roc_auc_score(yt_va, ys_va)
        }
    }

# ============================================================
# Search Space
# ============================================================

space = {
    "n_layers": hp.choice("n_layers", [3, 4]), # 3, 4, 5, 6
    "n0": hp.quniform("n0", 5, 20, 1), #5, 80, 1
    "n1": hp.quniform("n1", 5, 20, 1),
    "n2": hp.quniform("n2", 5, 20, 1),
    "n3": hp.quniform("n3", 5, 20, 1),
    "n4": hp.quniform("n4", 5, 20, 1),
    "n5": hp.quniform("n5", 5, 20, 1),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-1)),
    "batch": hp.choice("batch", [64, 128, 256]),
    "epochs": hp.choice("epochs", [10, 20, 30]) # 10, 20, 30, 50, 80, 120
}

# ============================================================
# Hyperopt Run
# ============================================================

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10, #50
    trials=trials
)

# ============================================================
# Resolver melhores parâmetros
# ============================================================
print("Best here trials ",best)
best_params = {
    "layers": [int(best[f"n{i}"]) for i in range(best["n_layers"])],
    "lr": best["lr"],
    "batch": [64, 128, 256][best["batch"]],
    "epochs": [10, 20, 30, 50, 80, 120][best["epochs"]]
}
print("Best after trials ",best)
print("best_params",best_params)
with open(f"{OUTDIR}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

# ============================================================
# Treino final
# ============================================================

model = FlexibleFCNet(INPUT_SIZE, best_params["layers"]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
criterion = nn.BCELoss()

train_loader = make_loader(X_train, y_train, best_params["batch"], shuffle=True)
val_loader   = make_loader(X_val, y_val, best_params["batch"])
test_loader  = make_loader(X_test, y_test, best_params["batch"])

losses = train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    best_params["epochs"]
)

torch.save(model.state_dict(), f"{OUTDIR}/model.pt")

# ============================================================
# Avaliação final + gráficos
# ============================================================

def full_evaluation(model, loader, name, threshold_report=False):
    yt, ys = evaluate_model_safe(model, loader)
    metrics = compute_metrics(yt, ys)

    with open(f"{OUTDIR}/report_{name}.txt", "w") as f:
        f.write(metrics["report"])
        f.write(f"\nROC-AUC: {metrics['roc_auc']:.6f}")
        f.write(f"\nPR-AUC : {metrics['auc_pr']:.6f}")
        f.write(f"\nFRR    : {metrics['frr']:.6f}")
        f.write(f"\nBrier : {metrics['brier']:.6f}")
        f.write(f"\nMCC   : {metrics['mcc']:.6f}")
        f.write(f"\nGMean : {metrics['gmean']:.6f}")

    if threshold_report:
        with open(f"{OUTDIR}/threshold_report_test.txt", "w") as f:
            for t in np.linspace(0.05, 0.95, 19):
                yp = (ys >= t).astype(int)
                f.write(f"\n--- Threshold {t:.2f} ---\n")
                f.write(
                    classification_report(
                        yt, yp, digits=4, zero_division=0
                    )
                )

    fpr, tpr, _ = roc_curve(yt, ys)
    prec, rec, _ = precision_recall_curve(yt, ys)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC - {name}")
    plt.savefig(f"{OUTDIR}/roc_{name}.png")
    plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR - {name}")
    plt.savefig(f"{OUTDIR}/pr_{name}.png")
    plt.close()


full_evaluation(model, train_loader, "train")
full_evaluation(model, val_loader, "validation")
full_evaluation(model, test_loader, "test", threshold_report=True)

plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.savefig(f"{OUTDIR}/loss.png")
plt.close()

print(" Pipeline finalizado com epochs como hiperparâmetro.")

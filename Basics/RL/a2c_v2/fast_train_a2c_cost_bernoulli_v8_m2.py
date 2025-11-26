# fast_train_a2c_cost_bernoulli_v8.py
# Simplified V8 (fixed): uses creditcard_train.csv, creditcard_val.csv, creditcard_test.csv
# Train a compact classifier with balanced replay pool and threshold tuning
# Save model + scaler into outputs_v8/

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------
# Config filenames
# --------------------
TRAIN_CSV = "creditcard_train.csv"
VAL_CSV   = "creditcard_val.csv"
TEST_CSV  = "creditcard_test.csv"

OUTPUT_DIR = "outputs_v8"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# ReplayPool
# --------------------
class ReplayPool:
    def __init__(self, max_size, obs_dim):
        self.max_size = max_size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.labels = np.zeros(max_size, dtype=np.int64)

    def add(self, x, y):
        self.obs[self.ptr] = x
        self.labels[self.ptr] = y
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.ptr = 0
            self.full = True

    def sample_all(self):
        if self.full:
            return self.obs, self.labels
        else:
            return self.obs[:self.ptr], self.labels[:self.ptr]

# --------------------
# Model
# --------------------
class A2CNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --------------------
# helpers
# --------------------
def eval_with_best_threshold(probs, y_true):
    best_t = 0.5
    best_f1 = -1
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        f = f1_score(y_true, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1

# --------------------
# Load CSVs
# --------------------
def load_splits(train_csv=TRAIN_CSV, val_csv=VAL_CSV, test_csv=TEST_CSV):
    # Expect CSVs with same columns and 'Class'
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)
    df_test  = pd.read_csv(test_csv)

    if 'Class' not in df_train.columns:
        raise ValueError("CSV files must contain 'Class' column")

    X_train = df_train.drop(columns=['Class']).values.astype(np.float32)
    y_train = df_train['Class'].values.astype(np.int64)

    X_val = df_val.drop(columns=['Class']).values.astype(np.float32)
    y_val = df_val['Class'].values.astype(np.int64)

    X_test = df_test.drop(columns=['Class']).values.astype(np.float32)
    y_test = df_test['Class'].values.astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test

# --------------------
# Main train routine
# --------------------
def train_v8():
    # load
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    # scaler: fit on train, transform others
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # save scaler separately
    with open(os.path.join(OUTPUT_DIR, "scaler_v8.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    d = X_train.shape[1]
    print(f"Train samples: {len(X_train)}, pos={int(y_train.sum())}, neg={len(y_train)-int(y_train.sum())}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model/opt
    model = A2CNet(d).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-3)
    bce = nn.BCELoss()

    # pool: capacity = train size
    pool = ReplayPool(max_size=len(X_train), obs_dim=d)

    epochs = 200
    patience = 30
    best_f1 = -1.0
    best_epoch = 0
    wait = 0
    best_thr = 0.5

    for ep in range(1, epochs + 1):
        t0 = time.time()

        # fill pool with training data once per epoch
        for i in range(len(X_train)):
            pool.add(X_train[i], y_train[i])

        obs, labels = pool.sample_all()
        Xb = torch.tensor(obs, dtype=torch.float32, device=device)
        yb = torch.tensor(labels, dtype=torch.float32, device=device)

        model.train()
        opt.zero_grad()
        probs = model(Xb).flatten()
        loss = bce(probs, yb)
        loss.backward()
        opt.step()

        # validation: get probabilities and find best threshold
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        thr, val_f1 = eval_with_best_threshold(pv, y_val)
        val_auc = roc_auc_score(y_val, pv)

        dt = time.time() - t0
        print(f"EP {ep}/{epochs} | loss={loss.item():.4f} | val_f1={val_f1:.4f} | val_auc={val_auc:.4f} | thr={thr:.2f} | time={dt:.2f}s | best={best_f1:.4f}")

        # early stopping & save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_thr = thr
            best_epoch = ep
            wait = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_v8.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}, best epoch {best_epoch}.")
                break

    # final evaluation: load best and evaluate train/val/test with best_thr
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_v8.pth"), map_location=device))
    def final_eval(Xd, yd):
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(Xd, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        preds = (p >= best_thr).astype(int)
        print(classification_report(yd, preds, zero_division=0))
        auc = roc_auc_score(yd, p)
        return auc, p

    print("\n=== FINAL TRAIN REPORT ===")
    auc_train, p_train = final_eval(X_train, y_train)
    print("AUC train:", auc_train)

    print("\n=== FINAL VAL REPORT ===")
    auc_val, p_val = final_eval(X_val, y_val)
    print("AUC val:", auc_val)

    print("\n=== FINAL TEST REPORT ===")
    auc_test, p_test = final_eval(X_test, y_test)
    print("AUC test:", auc_test)

    # save artifacts
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_v8.pth"))
    with open(os.path.join(OUTPUT_DIR, "best_threshold_v8.txt"), "w") as f:
        f.write(f"{best_thr}\n{best_f1}\n")

    # save train/val/test probabilities for post analysis
    np.save(os.path.join(OUTPUT_DIR, "probs_train_v8.npy"), p_train)
    np.save(os.path.join(OUTPUT_DIR, "probs_val_v8.npy"), p_val)
    np.save(os.path.join(OUTPUT_DIR, "probs_test_v8.npy"), p_test)

    print("\nDone. Model + scaler saved in", OUTPUT_DIR)

if __name__ == "__main__":
    train_v8()

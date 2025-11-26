# fast_train_a2c_cost_bernoulli_v9.py
# V9: Full A2C-like trainer (based on v8) for creditcard datasets (train/val/test CSVs)
# Features:
# - LR linear warmup
# - adaptive advantage scaling (adv_scale = 1/(1+var(adv)))
# - hybrid baseline: EMA scalar + value head
# - pre-generated balanced pool (superfast)
# - focal auxiliary supervised loss
# - cost-shaped rewards
# - threshold tuning, plots (F1 vs thr, Recall vs thr), confusion matrix (test)
# - saves model checkpoint (weights only), scaler saved separately via pickle
# - safe loading with weights_only=False inside try/except

import os, time, pickle
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix, classification_report

# -------------------------
# Config
# -------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_CSV = "creditcard_train.csv"
VAL_CSV   = "creditcard_val.csv"
TEST_CSV  = "creditcard_test.csv"

OUTPUT_DIR = "outputs_v9"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "fast_a2c_v9_checkpoint.pth")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_v9.pkl")
BEST_THR_PATH = os.path.join(OUTPUT_DIR, "best_threshold_v9.txt")
LOG_CSV = os.path.join(OUTPUT_DIR, "train_logs_v9.csv")
FINAL_METRICS = os.path.join(OUTPUT_DIR, "final_metrics_v9.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# Hyperparameters (tune as needed)
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 4096
VALUE_COEF = 0.6
ENTROPY_COEF = 0.03
F1_COEF = 2.0                # focal loss weight
ADV_NORM = True
GRAD_CLIP = 1.0
EMA_ALPHA = 0.9
WARMUP_STEPS = 10            # linear warmup epochs
GAMMA = 0.99
DROPOUT = 0.0

# Balanced sampling
BATCH_POS_FRAC = 0.2
SUPERFAST = True
POOL_SIZE = 5000             # number of pre-generated batch index arrays in pool

# Cost matrix shaping
COSTS = {"TP": +5.0, "TN": +0.5, "FP": -12.0, "FN": -20.0}
REWARD_SCALE = max(abs(v) for v in COSTS.values())

# Early stopping
EARLY_STOPPING = True
PATIENCE = 25

# threshold search grid
THR_GRID = np.linspace(0.01, 0.99, 99)
REPORT_THRESHOLD = 0.5

# focal loss params
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

# -------------------------
# Helpers
# -------------------------
def append_csv_row(path, row, header=None):
    write_header = not os.path.exists(path)
    import csv
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header and header is not None:
            w.writerow(header)
        w.writerow(row)

def focal_loss_binary(logits, targets, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, eps=1e-9):
    # logits: raw logits, targets: 0/1
    p = torch.sigmoid(logits)
    targets = targets.float()
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = - w * ((1 - pt) ** gamma) * torch.log(pt + eps)
    return loss.mean()

def compute_cost_rewards(actions, labels, costs, reward_scale=1.0):
    act = actions.long(); lab = labels.long()
    rewards = torch.zeros_like(act, dtype=torch.float32)
    rewards[(act==1)&(lab==1)] = float(costs["TP"])
    rewards[(act==0)&(lab==0)] = float(costs["TN"])
    rewards[(act==1)&(lab==0)] = float(costs["FP"])
    rewards[(act==0)&(lab==1)] = float(costs["FN"])
    return rewards / reward_scale

def plot_metric_vs_threshold(probs_val, y_val, probs_test, y_test, grid, best_thr, out_prefix):
    val_f1s, val_recs, test_f1s, test_recs = [], [], [], []
    for thr in grid:
        pv = (probs_val >= thr).astype(int)
        pt = (probs_test >= thr).astype(int)
        val_f1s.append(f1_score(y_val, pv, average='macro', zero_division=0))
        val_recs.append(recall_score(y_val, pv, zero_division=0))
        test_f1s.append(f1_score(y_test, pt, average='macro', zero_division=0))
        test_recs.append(recall_score(y_test, pt, zero_division=0))

    plt.figure(figsize=(8,4))
    plt.plot(grid, val_f1s, label='val F1-macro')
    plt.plot(grid, test_f1s, label='test F1-macro')
    plt.axvline(best_thr, color='k', linestyle='--', label=f"best_thr={best_thr:.2f}")
    plt.xlabel("Threshold"); plt.ylabel("F1-macro"); plt.title("F1 vs Threshold"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{out_prefix}_f1_vs_threshold.png"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(grid, val_recs, label='val Recall')
    plt.plot(grid, test_recs, label='test Recall')
    plt.axvline(best_thr, color='k', linestyle='--', label=f"best_thr={best_thr:.2f}")
    plt.xlabel("Threshold"); plt.ylabel("Recall (pos)"); plt.title("Recall vs Threshold"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{out_prefix}_recall_vs_threshold.png"), bbox_inches="tight"); plt.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, out_path=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# -------------------------
# Model
# -------------------------
class FastA2CNetV9(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.logit = nn.Linear(64, 1)   # returns raw logit
        self.value = nn.Linear(64, 1)   # value head

    def forward(self, x):
        h = self.shared(x)
        logit = self.logit(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        val = self.value(h).squeeze(-1)
        return logit, prob, val

# -------------------------
# Pre-generated balanced pool (no nonlocal)
# -------------------------
def make_pool(idx_pos, idx_neg, pool_size, batch_size, pos_frac):
    pool = []
    for i in range(pool_size):
        n_pos = int(round(batch_size * pos_frac))
        n_pos = min(max(1, n_pos), len(idx_pos))
        n_neg = batch_size - n_pos
        n_neg = min(max(1, n_neg), len(idx_neg))
        pos_idx = np.random.choice(idx_pos, n_pos, replace=(n_pos > len(idx_pos)))
        neg_idx = np.random.choice(idx_neg, n_neg, replace=(n_neg > len(idx_neg)))
        arr = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(arr)
        pool.append(arr)
    return pool

# -------------------------
# Load CSVs and prepare tensors
# -------------------------
def load_and_prepare(train_csv, val_csv, test_csv):
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

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # save scaler separately
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # convert to torch tensors (on device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t)

# -------------------------
# Training loop
# -------------------------
def train():
    # load
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t) = load_and_prepare(TRAIN_CSV, VAL_CSV, TEST_CSV)

    N = X_train.shape[0]
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == 0)[0]
    print(f"Train samples: {N}, pos: {len(idx_pos)}, neg: {len(idx_neg)}, pos_frac={len(idx_pos)/N:.6f}")

    model = FastA2CNetV9(X_train.shape[1], dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # warmup: lr schedule function
    def adjust_lr(epoch):
        if WARMUP_STEPS <= 0: return
        if epoch <= WARMUP_STEPS:
            factor = epoch / float(max(1, WARMUP_STEPS))
            for g in optimizer.param_groups:
                g['lr'] = LR * factor

    # prepare pool
    pool = make_pool(idx_pos, idx_neg, POOL_SIZE if SUPERFAST else 1, BATCH_SIZE, BATCH_POS_FRAC)
    pool_ptr = 0

    # bookkeeping
    best_val_f1 = -np.inf
    best_epoch = -1
    patience_counter = 0
    train_reward_hist = []
    start_time = time.time()
    header = ['epoch','time_s','train_f1_approx','val_f1_best','val_f1','test_f1','val_auc','test_auc','val_recall','test_recall','val_frr','test_frr','batch_reward','adv_var','adv_scale','lr']
    if not os.path.exists(LOG_CSV):
        append_csv_row(LOG_CSV, [], header=header)

    # training epochs
    for epoch in range(1, EPOCHS+1):
        model.train()
        adjust_lr(epoch)
        epoch_t0 = time.time()

        # get batch indices from pool
        idx = pool[pool_ptr]
        pool_ptr += 1
        if pool_ptr >= len(pool):
            pool_ptr = 0
            # regenerate pool to add randomness
            pool = make_pool(idx_pos, idx_neg, POOL_SIZE if SUPERFAST else 1, BATCH_SIZE, BATCH_POS_FRAC)

        Xb = X_train_t[idx]
        yb = y_train_t[idx]

        logits, probs, vals = model(Xb)   # logits (raw), probs (sigmoid), vals

        # sample actions and compute log-probs and entropy
        bern = torch.distributions.Bernoulli(probs=probs)
        actions = bern.sample()
        logp = bern.log_prob(actions)
        entropy = bern.entropy().mean()

        # cost-shaped rewards
        rewards_vec = compute_cost_rewards(actions, yb, COSTS, reward_scale=REWARD_SCALE)
        batch_reward = rewards_vec.mean()
        train_reward_hist.append(float(batch_reward.cpu().item()))

        # EMA baseline scalar (initialize on first epoch)
        if epoch == 1:
            ema_baseline = float(batch_reward.detach().cpu().item())
        else:
            ema_baseline = EMA_ALPHA * ema_baseline + (1.0 - EMA_ALPHA) * float(batch_reward.detach().cpu().item())

        # hybrid baseline: alpha * EMA + (1-alpha) * vals.mean()
        alpha_bl = 0.8 if epoch < (EPOCHS * 0.3) else 0.5
        baseline_scalar = alpha_bl * ema_baseline + (1 - alpha_bl) * float(vals.mean().detach().cpu().item())
        baseline_tensor = torch.tensor(baseline_scalar, dtype=torch.float32, device=DEVICE)

        # advantage vector: (batch_reward - vals)
        adv_vec = (batch_reward.detach() - vals)

        # adaptive scaling based on variance
        adv_var = float(adv_vec.var(unbiased=False).detach().cpu().item())
        adv_scale = 1.0 / (1.0 + adv_var)
        adv_vec = adv_vec * adv_scale

        # normalize advantage
        if ADV_NORM:
            adv_mean = adv_vec.mean()
            adv_std = adv_vec.std(unbiased=False) + 1e-9
            adv_vec = (adv_vec - adv_mean) / adv_std

        # losses
        policy_loss = -(logp * adv_vec).mean()
        value_loss = nn.MSELoss()(vals, batch_reward.expand_as(vals))
        entropy_loss = - entropy
        focal_loss = focal_loss_binary(logits, yb, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

        loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss + F1_COEF * focal_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # evaluation on val/test using grid search for best threshold (maximize f1-macro)
        model.eval()
        with torch.no_grad():
            _, probs_val_all, _ = model(X_val_t)
            _, probs_test_all, _ = model(X_test_t)
            probs_val_np = probs_val_all.detach().cpu().numpy()
            probs_test_np = probs_test_all.detach().cpu().numpy()

            # grid search best threshold on val
            best_thr = REPORT_THRESHOLD
            best_thr_f1 = -np.inf
            for thr in THR_GRID:
                preds_val = (probs_val_np >= thr).astype(int)
                f1m = f1_score(y_val, preds_val, average='macro', zero_division=0)
                if f1m > best_thr_f1:
                    best_thr_f1 = f1m
                    best_thr = thr

            # compute metrics with best_thr
            val_preds = (probs_val_np >= best_thr).astype(int)
            val_auc = roc_auc_score(y_val, probs_val_np)
            val_recall = recall_score(y_val, val_preds, zero_division=0)
            tn = int(((y_val==0) & (val_preds==0)).sum()); fn = int(((y_val==1) & (val_preds==0)).sum())
            val_frr = (fn/(fn+tn)) if (fn+tn) > 0 else 0.0
            val_f1 = float(f1_score(y_val, val_preds, average='macro', zero_division=0))

            probs_test_np = probs_test_np
            test_preds = (probs_test_np >= best_thr).astype(int)
            test_auc = roc_auc_score(y_test, probs_test_np)
            test_recall = recall_score(y_test, test_preds, zero_division=0)
            tn = int(((y_test==0) & (test_preds==0)).sum()); fn = int(((y_test==1) & (test_preds==0)).sum())
            test_frr = (fn/(fn+tn)) if (fn+tn) > 0 else 0.0
            test_f1 = float(f1_score(y_test, test_preds, average='macro', zero_division=0))

        epoch_time = time.time() - epoch_t0
        total_elapsed = time.time() - start_time
        lr_now = optimizer.param_groups[0]['lr'] if 'optimizer' in locals() else LR

        # log row
        train_f1_approx = float(f1_score(yb.detach().cpu().numpy().astype(int), (probs.detach().cpu().numpy()>=REPORT_THRESHOLD).astype(int), average='macro', zero_division=0))
        row = [epoch, round(total_elapsed,3), train_f1_approx, float(best_val_f1 if best_val_f1!=-np.inf else 0.0), val_f1, test_f1, float(val_auc), float(test_auc), float(val_recall), float(test_recall), float(val_frr), float(test_frr), float(batch_reward.cpu().item()), adv_var, adv_scale, lr_now]
        append_csv_row(LOG_CSV, row)

        # checkpoint & early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # save checkpoint containing tensors only (no scaler)
            ck = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'best_thr': best_thr,
                  'epoch': epoch}
            torch.save(ck, CHECKPOINT_PATH)
            with open(BEST_THR_PATH, "w") as f:
                f.write(f"{best_thr}\n{best_val_f1}\n")
        else:
            patience_counter += 1

        if epoch % PRINT_EVERY == 0:
            print(f"EP {epoch}/{EPOCHS} | loss={loss.item():.4f} | train_f1_approx={train_f1_approx:.4f} | val_f1={val_f1:.4f} | val_auc={val_auc:.4f} | adv_var={adv_var:.4f} | adv_scale={adv_scale:.4f} | time_epoch={epoch_time:.2f}s | best_val_f1={best_val_f1:.4f}")

        if EARLY_STOPPING and (patience_counter >= PATIENCE):
            print(f"Early stopping at epoch {epoch} (best_val_f1={best_val_f1:.4f} at epoch {best_epoch})")
            break

    # -------------------------
    # Final evaluation: load best checkpoint safely and scaler
    # -------------------------
    # load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler_loaded = pickle.load(f)

    # load checkpoint (weights_only=False for PyTorch >=2.6) inside try/except for compatibility
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        best_thr_saved = ck.get('best_thr', REPORT_THRESHOLD)
    else:
        best_thr_saved = REPORT_THRESHOLD

    model.eval()
    with torch.no_grad():
        _, probs_train, _ = model(X_train_t)
        _, probs_val, _ = model(X_val_t)
        _, probs_test, _ = model(X_test_t)

    probs_train_np = probs_train.detach().cpu().numpy()
    probs_val_np = probs_val.detach().cpu().numpy()
    probs_test_np = probs_test.detach().cpu().numpy()

    # final metrics and classification reports
    def final_eval(probs_np, y_arr, thr):
        preds = (probs_np >= thr).astype(int)
        rep = classification_report(y_arr, preds, zero_division=0)
        auc = roc_auc_score(y_arr, probs_np)
        rec = recall_score(y_arr, preds, zero_division=0)
        tn = int(((y_arr==0) & (preds==0)).sum()); fn = int(((y_arr==1) & (preds==0)).sum())
        frr = (fn/(fn+tn)) if (fn+tn)>0 else 0.0
        f1m = float(f1_score(y_arr, preds, average='macro', zero_division=0))
        return {"report": rep, "auc": auc, "recall": rec, "frr": frr, "f1_macro": f1m, "preds": preds}

    train_metrics = final_eval(probs_train_np, y_train, best_thr_saved)
    val_metrics = final_eval(probs_val_np, y_val, best_thr_saved)
    test_metrics = final_eval(probs_test_np, y_test, best_thr_saved)

    print("\n=== FINAL TRAIN REPORT ===")
    print(train_metrics['report'])
    print("\n=== FINAL VAL REPORT ===")
    print(val_metrics['report'])
    print("\n=== FINAL TEST REPORT ===")
    print(test_metrics['report'])

    # write final metrics file
    with open(FINAL_METRICS, "w") as f:
        f.write(f"BEST_THRESHOLD:{best_thr_saved}\n\n")
        f.write("=== TRAIN ===\n"); f.write(train_metrics['report'] + "\n")
        f.write("=== VAL ===\n"); f.write(val_metrics['report'] + "\n")
        f.write("=== TEST ===\n"); f.write(test_metrics['report'] + "\n")

    # plots: F1/Recall vs threshold
    plot_metric_vs_threshold(probs_val_np, y_val, probs_test_np, y_test, THR_GRID, best_thr_saved, "valtest_v9")

    # confusion matrix on test (absolute and normalized)
    cm = confusion_matrix(y_test, test_metrics['preds'])
    plot_confusion_matrix(cm, classes=['0','1'], normalize=False, title='Confusion matrix (test) - counts', out_path=os.path.join(PLOTS_DIR, "cm_test_counts_v9.png"))
    plot_confusion_matrix(cm, classes=['0','1'], normalize=True, title='Confusion matrix (test) - normalized', out_path=os.path.join(PLOTS_DIR, "cm_test_norm_v9.png"))

    # additional plots: reward history and val_f1 across epochs
    try:
        import pandas as pd
        log_df = pd.read_csv(LOG_CSV)
        plt.figure(figsize=(8,4)); plt.plot(train_reward_hist); plt.title("Train batch reward history (v9)"); plt.savefig(os.path.join(PLOTS_DIR, "train_reward_v9.png"), bbox_inches="tight"); plt.close()
        plt.figure(figsize=(8,4)); plt.plot(log_df.index+1, log_df['val_f1']); plt.title("Val F1 per epoch (v9)"); plt.xlabel("Epoch"); plt.ylabel("Val F1"); plt.grid(True); plt.savefig(os.path.join(PLOTS_DIR, "val_f1_epoch_v9.png"), bbox_inches="tight"); plt.close()
    except Exception:
        pass

    # save final artifacts
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, "model_weights_v9.pth"))
    print("Done. Checkpoint saved to:", CHECKPOINT_PATH)
    print("Scaler saved to:", SCALER_PATH)
    print("Plots saved to:", PLOTS_DIR)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    start_time = time.time()
    train()

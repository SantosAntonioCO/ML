"""
train_a2c_cost_bernoulli_f1.py
A2C training with Bernoulli policy and F1-macro differential reward.
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, f1_score, roc_auc_score
import pandas as pd
import gymnasium as gym
import gym_fraud_rl
from gym_fraud_rl.envs.gym_fraud_env import FraudEnv
print ("check 1")
# -------------------------
# Config
# -------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "creditcard_train.csv"
VAL_CSV   = "creditcard_val.csv"
TEST_CSV  = "creditcard_test.csv"

OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "a2c_f1_model.pth")
LOG_CSV = os.path.join(OUTPUT_DIR, "train_logs_f1.csv")

# Hyperparams
LR = 1e-3
GAMMA = 0.99
EPISODES = 30
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

# F1 differential scale (tune if signal too small)
F1_SCALE = 100.0

# threshold to produce hard labels for final reports
CLASS_THRESHOLD = 0.5

# -------------------------
# Utils
# -------------------------
def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def append_log(csv_path, row, header=None):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header and header is not None:
            w.writerow(header)
        w.writerow(row)

def compute_returns(rewards, last_value, gamma):
    R = last_value
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return np.array(returns, dtype=np.float32)
print ("check 2")
# -------------------------
# Wrapper: F1-differential reward
# -------------------------
class F1DiffWrappedEnv:
    """
    Wrap FraudEnv to produce reward = scaled * (f1_macro(t) - f1_macro(t-1))
    Keeps history of labels/preds within episode.
    """
    def __init__(self, csv_path, scaler=None, shuffle=False):
        self.env = FraudEnv(csv_path, scaler=scaler, shuffle=shuffle)
        self.n_features = self.env.n_features
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_history()

    def reset_history(self):
        self.labels_hist = []
        self.preds_hist = []
        self.prev_f1 = 0.0

    def reset(self, seed=None):
        self.reset_history()
        obs, info = self.env.reset(seed=seed)
        return obs, {}

    def step(self, action):
        # action: int 0/1
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        label = int(info["label"])
        pred = int(action)

        # update history
        self.labels_hist.append(label)
        self.preds_hist.append(pred)

        # compute f1 current and differential
        try:
            f1_curr = f1_score(self.labels_hist, self.preds_hist, average="macro", zero_division=0)
        except Exception:
            f1_curr = 0.0

        diff = f1_curr - self.prev_f1
        reward = float(F1_SCALE * diff)
        self.prev_f1 = f1_curr

        return obs, reward, terminated, truncated, {"label": label, "action": pred, "f1": f1_curr}

# -------------------------
# Model: Bernoulli policy + value
# -------------------------
class A2CBernoulliNet(nn.Module):
    def __init__(self, obs_dim, hidden=(32,16)): #hidden=(128,64)
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.logit = nn.Linear(in_dim, 1)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        h = self.shared(x)
        logit = self.logit(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        val = self.value(h).squeeze(-1)
        if squeeze:
            prob = prob.squeeze(0)
            val = val.squeeze(0)
        return prob, val

# -------------------------
# Data / scaler / envs
# -------------------------
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)
print("check 3")
X_train = train_df.drop(columns=["Class"]).values.astype(np.float32)
scaler = StandardScaler().fit(X_train)

env_train = F1DiffWrappedEnv(TRAIN_CSV, scaler=scaler, shuffle=True)
env_val   = F1DiffWrappedEnv(VAL_CSV,   scaler=scaler, shuffle=False)
env_test  = F1DiffWrappedEnv(TEST_CSV,  scaler=scaler, shuffle=False)

obs_dim = env_train.n_features
print("check 4")
# -------------------------
# Model / optimizer
# -------------------------
model = A2CBernoulliNet(obs_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
print("check 5")
# -------------------------
# Evaluation (produces probs & labels; used for AUC and reports)
# -------------------------
def evaluate_env_probs(env, model, device, threshold=CLASS_THRESHOLD):
    model.eval()
    state, _ = env.reset()
    done = False
    labels, preds, probs = [], [], []
    total_reward = 0.0

    with torch.no_grad():
        while not done:
            s = torch.tensor(state, dtype=torch.float32, device=device)
            prob, _ = model(s)
            p1 = float(prob.item())
            action = 1 if p1 >= threshold else 0
            next_state, reward, terminated, truncated, info = env.step(action)

            labels.append(info["label"])
            preds.append(action)
            probs.append(p1)
            total_reward += reward

            state = next_state
            done = terminated or truncated

    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    recall = recall_score(labels, preds, zero_division=0)
    # FRR
    tn = np.sum((labels == 0) & (preds == 0))
    fn = np.sum((labels == 1) & (preds == 0))
    FRR = (fn / (fn + tn)) if (fn + tn) > 0 else 0.0
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    report_text = classification_report(labels, preds, zero_division=0)
    return {"reward": float(total_reward), "recall": float(recall), "FRR": float(FRR),
            "AUC": float(auc), "labels": labels, "preds": preds, "probs": probs, "report_text": report_text}
print("check 5")
# -------------------------
# Training loop
# -------------------------
train_rewards_history = []
header = ['ep','train_reward','val_reward','val_recall','val_frr','val_auc','test_reward','test_recall','test_frr','test_auc']
if not os.path.exists(LOG_CSV):
    append_log(LOG_CSV, [], header=header)
print("check 6 start train")
for ep in range(1, EPISODES+1):
    print (ep," ========of  ",EPISODES+1)
    model.train()
    state, _ = env_train.reset()
    done = False
    log_probs = []
    values = []
    rewards = []
    entropies = []
    ep_reward = 0.0

    while not done:
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        prob, value = model(s)
        bern = torch.distributions.Bernoulli(probs=prob)
        action = bern.sample()
        log_prob = bern.log_prob(action)
        entropy = bern.entropy()

        a = int(action.item())
        next_state, reward, terminated, truncated, info = env_train.step(a)
        done = terminated or truncated
        

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy) 

        ep_reward += reward
        state = next_state 

    returns = compute_returns(rewards, 0.0, GAMMA)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    values = torch.stack(values).squeeze()
    log_probs = torch.stack(log_probs).squeeze()
    entropies = torch.stack(entropies).squeeze()

    advantage = returns - values.detach()
    policy_loss = -(log_probs * advantage).mean()
    value_loss = (returns - values).pow(2).mean()
    entropy_loss = -entropies.mean()
    loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_rewards_history.append(ep_reward)

    # evaluate
    val_res = evaluate_env_probs(env_val, model, DEVICE)
    test_res = evaluate_env_probs(env_test, model, DEVICE)

    row = [ep, float(ep_reward),
           float(val_res['reward']), float(val_res['recall']), float(val_res['FRR']), float(val_res['AUC']),
           float(test_res['reward']), float(test_res['recall']), float(test_res['FRR']), float(test_res['AUC'])]
    append_log(LOG_CSV, row)

    print(f"EP {ep}/{EPISODES} | train_reward={ep_reward:.3f} | val_recall={val_res['recall']:.4f} | val_AUC={val_res['AUC']:.4f} | test_recall={test_res['recall']:.4f} | test_AUC={test_res['AUC']:.4f}")

    if ep % 5 == 0:
        torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, MODEL_PATH)
print("check 7 pos train")
torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, MODEL_PATH)

# Final eval & save reports
train_res = evaluate_env_probs(env_train, model, DEVICE)
val_res   = evaluate_env_probs(env_val, model, DEVICE)
test_res  = evaluate_env_probs(env_test, model, DEVICE)
print("check 8")
print("\n=== FINAL REPORT TRAIN ===")
print(train_res['report_text'])
print("\n=== FINAL REPORT VAL ===")
print(val_res['report_text'])
print("\n=== FINAL REPORT TEST ===")
print(test_res['report_text'])

with open(os.path.join(OUTPUT_DIR, "final_metrics_f1diff.txt"), "w") as f:
    f.write("=== TRAIN ===\n")
    f.write(train_res['report_text'] + "\n")
    f.write(f"Reward: {train_res['reward']}\nRecall: {train_res['recall']}\nFRR: {train_res['FRR']}\nAUC: {train_res['AUC']}\n\n")
    f.write("=== VAL ===\n")
    f.write(val_res['report_text'] + "\n")
    f.write(f"Reward: {val_res['reward']}\nRecall: {val_res['recall']}\nFRR: {val_res['FRR']}\nAUC: {val_res['AUC']}\n\n")
    f.write("=== TEST ===\n")
    f.write(test_res['report_text'] + "\n")
    f.write(f"Reward: {test_res['reward']}\nRecall: {test_res['recall']}\nFRR: {test_res['FRR']}\nAUC: {test_res['AUC']}\n\n")

# Plots
plt.figure(figsize=(8,4))
plt.plot(train_rewards_history)
plt.title("Train reward per episode (F1 diff)")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "train_rewards_f1diff.png"), bbox_inches="tight")
plt.close()

plt.figure()
plt.bar(["Train","Val","Test"], [train_res['recall'], val_res['recall'], test_res['recall']])
plt.title(f"Recall (threshold {CLASS_THRESHOLD:.2f})")
plt.savefig(os.path.join(PLOTS_DIR, "recall_f1diff.png"), bbox_inches="tight")
plt.close()

plt.figure()
plt.bar(["Train","Val","Test"], [train_res['FRR'], val_res['FRR'], test_res['FRR']])
plt.title("FRR (FN/(FN+TN))")
plt.savefig(os.path.join(PLOTS_DIR, "frr_f1diff.png"), bbox_inches="tight")
plt.close()

print("Done. Model, logs and plots saved to", OUTPUT_DIR)

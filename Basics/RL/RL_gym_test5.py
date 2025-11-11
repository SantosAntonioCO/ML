"""
train dqn fraudv0
# Code based on https://github.com/purvasingh96/gym-fraud/tree/master
Trains a DQN agent in the Fraud-v0 environment (gym_fraud) using Gymnasium + PyTorch.
It shows graphs: reward per episode, ROC curve, PR curve, confusion matrix, and classification report.

Requirements
python=3.11  pip install torch numpy pandas scikit-learn matplotlib tqdm gymnasium 
numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 matplotlib==3.8.4 gymnasium==0.29.1
torch                2.5.1
torchaudio           2.5.1
torchvision          0.20.1
tqdm                 4.67.1

Run this within the environment that contains your gym_fraud package (pip install -e . already done).
"""

import os
import time
import random
import math
import collections
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import gymnasium as gym
import gym_fraud  # garante registro de Fraud-v0
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------
# Settings 
# -------------------------
SEED = 42
NUM_EPISODES = 50           # 500 number of training episodes
MAX_STEPS_PER_EP = 10000   #1000000 Limit (safety) per episode; the broadcast env ends automatically.
BATCH_SIZE = 64
REPLAY_CAPACITY = 3000 #30000
MIN_REPLAY_SIZE = 200  #2000
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE_EVERY = 100   #1000 training steps to update the target network
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50000
MODEL_CHECKPOINT = "dqn_fraud_checkpoint.pth"
RESULTS_DIR_ROOT = "results"
SAVE_PLOTS = True            # save plots 
PRINT_EVERY = 3             #10 print status for each N episodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Utilities
# -------------------------
def make_results_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(RESULTS_DIR_ROOT) / f"dqn_fraud_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def preprocess_obs(obs):
    """Converts a pandas.Series to a numpy float32 array, or returns the existing array.."""
    if hasattr(obs, "values"):
        arr = obs.values.astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32)
    return arr

def epsilon_by_step(step):
    """Linear-ish decay from EPS_START to EPS_END across EPS_DECAY_STEPS"""
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_END + (EPS_START - EPS_END) * (1 - (step / EPS_DECAY_STEPS))

# -------------------------
# Replay buffer
# -------------------------
Transition = collections.namedtuple("Transition", ["s", "a", "r", "s2", "done"])
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append(Transition(s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Q-network (30 -> 20 -> 5 -> hidden -> outputs)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden1=20, hidden2=5, n_actions=2):
        super().__init__()
        # layer: 30->20->5 (hidden)
        # output: n_actions (Q for each action)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Final evaluation function (sweeps through the entire dataset of the environment, eps=0)
# -------------------------
def evaluate_policy(policy_net, device):
    """
    Execute a complete run of the Fraud-v0 environment with epsilon=0 (deterministic).
    Collect true labels and predicted actions for metrics (ROC, PR, confusion matrix).
    """
    env = gym.make("Fraud-v0")
    res = env.reset()
    if isinstance(res, tuple):
        obs, info = res
    else:
        obs, info = res, {}
    obs = preprocess_obs(obs)
    y_true = []
    y_pred = []

    done = False
    while True:
        s_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            q = policy_net(s_t)
            action = int(torch.argmax(q, dim=1).item())

        # If info contains label from reset, handle; else rely on step info
        # Step
        step_res = env.step(action)
        if len(step_res) == 5:
            obs2, reward, terminated, truncated, info = step_res
            done_flag = terminated or truncated
        else:
            obs2, reward, done_flag, info = step_res

        # info should contain true_label per env implementation
        true_label = info.get("true_label")
        # If not provided, try to compute from env (if env exposes labels)
        if true_label is None and hasattr(env, "labels") and hasattr(env, "current_index"):
            # note: current_index has already been incremented in env.step() in this implementation
            idx = env.current_index - 1
            if idx >= 0 and idx < len(env.labels):
                true_label = int(env.labels[idx])
        if true_label is None:
            # fallback: cannot evaluate reliably
            break

        y_true.append(int(true_label))
        y_pred.append(int(action))

        if done_flag:
            break

        obs = preprocess_obs(obs2)

    env.close()
    return y_true, y_pred

# -------------------------
# Training
# -------------------------
def train():
    results_dir = make_results_dir()
    print("Results ->", results_dir)

    env = gym.make("Fraud-v0")
    # sample obs to get dimension
    res = env.reset()
    if isinstance(res, tuple):
        obs0, info = res
    else:
        obs0, info = res, {}
    obs0 = preprocess_obs(obs0)
    state_dim = obs0.shape[0]
    n_actions = env.action_space.n

    policy_net = QNetwork(input_dim=state_dim, hidden1=20, hidden2=5, n_actions=n_actions).to(DEVICE)
    target_net = QNetwork(input_dim=state_dim, hidden1=20, hidden2=5, n_actions=n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    replay = ReplayBuffer(REPLAY_CAPACITY)

    # Pre-fill replay with random transitions to stabilize training
    print("Populating replay buffer (random policy)...")
    steps = 0
    while len(replay) < MIN_REPLAY_SIZE:
        res = env.reset()
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}
        obs = preprocess_obs(obs)
        done = False
        while not done and len(replay) < MIN_REPLAY_SIZE:
            a = env.action_space.sample()
            step_res = env.step(a)
            if len(step_res) == 5:
                obs2, reward, terminated, truncated, info = step_res
                done_flag = terminated or truncated
            else:
                obs2, reward, done_flag, info = step_res
            obs2 = preprocess_obs(obs2)
            replay.push(obs, int(a), float(reward), obs2, bool(done_flag))
            obs = obs2
            steps += 1
    print(f"Replay size = {len(replay)} ready. Starting training...")

    total_steps = 0
    train_losses = []
    rewards_per_episode = []
    best_val_auc = 0.0
    target_update_counter = 0

    # Main training loop
    for ep in range(1, NUM_EPISODES + 1):
        print("ep",ep," to ", NUM_EPISODES + 1)
        res = env.reset()
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}
        obs = preprocess_obs(obs)
        ep_reward = 0.0

        done = False
        step_in_ep = 0
        while not done and step_in_ep < MAX_STEPS_PER_EP:
            eps = epsilon_by_step(total_steps)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                s_t = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    qvals = policy_net(s_t)
                    action = int(torch.argmax(qvals, dim=1).item())

            step_res = env.step(action)
            if len(step_res) == 5:
                obs2, reward, terminated, truncated, info = step_res
                done_flag = terminated or truncated
            else:
                obs2, reward, done_flag, info = step_res

            obs2 = preprocess_obs(obs2)
            replay.push(obs, int(action), float(reward), obs2, bool(done_flag))

            obs = obs2
            ep_reward += float(reward)
            total_steps += 1
            step_in_ep += 1
            target_update_counter += 1

            # train step
            if len(replay) >= BATCH_SIZE:
                batch = replay.sample(BATCH_SIZE)
                s_batch = torch.tensor(np.stack(batch.s)).float().to(DEVICE)     # (B, state_dim)
                a_batch = torch.tensor(batch.a).long().unsqueeze(1).to(DEVICE)  # (B,1)
                r_batch = torch.tensor(batch.r).float().unsqueeze(1).to(DEVICE) # (B,1)
                s2_batch = torch.tensor(np.stack(batch.s2)).float().to(DEVICE)
                done_batch = torch.tensor(batch.done).float().unsqueeze(1).to(DEVICE)

                q_values = policy_net(s_batch).gather(1, a_batch)  # (B,1)
                with torch.no_grad():
                    q_next = target_net(s2_batch).max(dim=1, keepdim=True)[0]
                    target = r_batch + (1.0 - done_batch) * GAMMA * q_next

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                train_losses.append(loss.item())

            # update target
            if target_update_counter >= TARGET_UPDATE_EVERY:
                target_net.load_state_dict(policy_net.state_dict())
                target_update_counter = 0

            if done_flag:
                break

        rewards_per_episode.append(ep_reward)

        # periodic logging
        if ep % PRINT_EVERY == 0 or ep == 1:
            avg_recent = np.mean(rewards_per_episode[-PRINT_EVERY:])
            avg_loss = np.mean(train_losses[-200:]) if train_losses else 0.0
            print(f"EP {ep:4d} | total_steps {total_steps:7d} | eps {eps:.3f} | ep_reward {ep_reward:.3f} | avg_reward(last{PRINT_EVERY}) {avg_recent:.3f} | avg_loss {avg_loss:.6f}")

        # periodic evaluation & checkpoint
        if ep % 50 == 0:
            # evaluate deterministically over dataset
            y_true, y_pred = evaluate_policy(policy_net, DEVICE)
            try:
                auc_roc = roc_auc_score(y_true, y_pred)
            except Exception:
                auc_roc = 0.0
            # save model checkpoint
            ckpt_path = results_dir / f"checkpoint_ep{ep}.pth"
            torch.save({
                "episode": ep,
                "model_state": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay_size": len(replay)
            }, ckpt_path)

            print(f">>> checkpoint saved: {ckpt_path} | eval AUC-ROC: {auc_roc:.4f}")

    # end training
    print("Training finished. Total steps:", total_steps)

    # final save
    final_model_path = results_dir / MODEL_CHECKPOINT
    torch.save({
        "episode": NUM_EPISODES,
        "model_state": policy_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay_size": len(replay)
    }, final_model_path)
    print("Saved final model to", final_model_path)

    # -------------------------
    # Final evaluation and metrics
    # -------------------------
    print("Running final deterministic evaluation...")
    y_true, y_pred = evaluate_policy(policy_net, DEVICE)
    if len(y_true) == 0:
        print("No labels collected during evaluation — aborting metrics.")
        return

    # Metrics
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc_roc = float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("\n=== Final Evaluation Metrics ===")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    # Save report
    (results_dir / "metrics.txt").write_text(
        f"AUC-ROC: {auc_roc:.6f}\nAUC-PR: {auc_pr:.6f}\n\nConfusion matrix:\n{cm}\n\nClassification report:\n{report}\n"
    )

    # -------------------------
    # Plots
    # -------------------------
    if SAVE_PLOTS:
        # rewards plot
        plt.figure(figsize=(8,4))
        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("Reward per episode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "reward_per_episode.png", dpi=200)
        plt.close()

        # PR curve
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, label=f"AUC-PR={auc_pr:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "precision_recall.png", dpi=200)
        plt.close()

        # ROC-like (note: we used binary preds so it's discrete)
        # For visualization we'll create a simple ROC by treating preds as scores
        # If needed, adapt policy to output probability-like scores
        plt.figure(figsize=(6,6))
        # compute simple ROC from sklearn by using preds as scores (not ideal)
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, label=f"AUC-ROC={auc_roc:.4f}")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(results_dir / "roc_curve.png", dpi=200)
        except Exception as e:
            print("Could not plot ROC:", e)
        plt.close()

        # confusion matrix
        plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["pred0","pred1"])
        plt.yticks(tick_marks, ["true0","true1"])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png", dpi=200)
        plt.close()

    print("All artifacts saved to:", results_dir)
    env.close()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    results_dir = make_results_dir()
    train()

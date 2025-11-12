# DQN_fraud_notebook.py
# Jupyter-friendly script (cells separated by `# %%`) to train a DQN on the
# Fraud-v0 environment from the gym-fraud repository (originally written for gym).
# It is step-by-step and includes:
#  - compatibility shim (gym / gymnasium)
#  - ReplayBuffer implementation
#  - QNetwork definition
#  - training loop with epsilon-greedy, target network and replay
#  - evaluate_policy() that collects y_true/y_pred for the entire dataset
#  - final metrics: ROC-AUC, PR curve, confusion matrix and classification_report
#  - plotting and saving artifacts
#
# Save this file and open it in a Jupyter Notebook (File -> Open -> select file)
# or paste each cell into notebook cells. Adjust parameters at the top if desired.

# %%
# Cell 1: Imports, compatibility shim and config
import os
import sys
import time
import random
import math
import collections
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# try to import gym (repo uses gym). If not available, fall back to gymnasium and
# provide a compatibility alias so existing code can use `import gym`.
try:
    import gym
except Exception:
    import gymnasium as gym
    # expose as 'gym' for older code expecting that name
    sys.modules['gym'] = gym

from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report

# --------------------
# Configuration (edit as needed)
SEED = 42
NUM_EPISODES = 300          # number of training episodes
BATCH_SIZE = 64
REPLAY_CAPACITY = 30000
MIN_REPLAY_SIZE = 2000
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE_EVERY = 1000  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50000
MAX_STEPS_PER_EP = 1_000_000  # safety limit
RESULTS_DIR = Path("results")
SAVE_PLOTS = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("Device:", DEVICE)

# %%
# Cell 2: Utility functions

def make_results_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = RESULTS_DIR / f"dqn_fraud_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preprocess_obs(obs):
    # convert pandas.Series -> numpy float32, or return numpy array
    if hasattr(obs, "values"):
        return obs.values.astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


def epsilon_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_END + (EPS_START - EPS_END) * (1 - (step / EPS_DECAY_STEPS))

# %%
# Cell 3: Replay buffer and QNetwork
Transition = collections.namedtuple("Transition", ["s","a","r","s2","done"])

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

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden1=20, hidden2=5, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# %%
# Cell 4: evaluate_policy - deterministic pass over environment to collect y_true/y_pred

def evaluate_policy(policy_net, device):
    env = gym.make("Fraud-v0")
    res = env.reset()
    if isinstance(res, tuple):
        obs, info = res
    else:
        obs, info = res, {}
    obs = preprocess_obs(obs)

    y_true = []
    y_pred = []

    while True:
        s_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            q = policy_net(s_t)
            action = int(torch.argmax(q, dim=1).item())

        step_res = env.step(action)
        if len(step_res) == 5:
            obs2, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            obs2, reward, done, info = step_res

        # try to obtain true label from info; fallback to env labels if available
        true_label = None
        if isinstance(info, dict):
            true_label = info.get("true_label")
        if true_label is None and hasattr(env, 'labels') and hasattr(env, 'current_index'):
            idx = env.current_index - 1
            if 0 <= idx < len(env.labels):
                true_label = int(env.labels[idx])
        if true_label is None:
            # cannot collect labels -> break
            break

        y_true.append(int(true_label))
        y_pred.append(int(action))

        if done:
            break
        obs = preprocess_obs(obs2)

    env.close()
    return y_true, y_pred

# %%
# Cell 5: Training function (full DQN loop)

def train_dqn(num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, results_dir=None):
    if results_dir is None:
        results_dir = make_results_dir()
    print("Results folder:", results_dir)

    env = gym.make("Fraud-v0")
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

    # warm-up replay buffer with random policy
    print("Filling replay buffer with random transitions...")
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
    print("Replay buffer filled:", len(replay))

    total_steps = 0
    train_losses = []
    rewards_per_episode = []
    target_update_counter = 0

    for ep in trange(1, num_episodes+1, desc="Training episodes"):
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

            # training step
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                s_batch = torch.tensor(np.stack(batch.s)).float().to(DEVICE)
                a_batch = torch.tensor(batch.a).long().unsqueeze(1).to(DEVICE)
                r_batch = torch.tensor(batch.r).float().unsqueeze(1).to(DEVICE)
                s2_batch = torch.tensor(np.stack(batch.s2)).float().to(DEVICE)
                done_batch = torch.tensor(batch.done).float().unsqueeze(1).to(DEVICE)

                q_values = policy_net(s_batch).gather(1, a_batch)
                with torch.no_grad():
                    q_next = target_net(s2_batch).max(dim=1, keepdim=True)[0]
                    target = r_batch + (1.0 - done_batch) * GAMMA * q_next

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                train_losses.append(loss.item())

            if target_update_counter >= TARGET_UPDATE_EVERY:
                target_net.load_state_dict(policy_net.state_dict())
                target_update_counter = 0

            if done_flag:
                break

        rewards_per_episode.append(ep_reward)

        # periodic checkpoint/eval every 50 episodes
        if ep % 50 == 0:
            ckpt = results_dir / f"checkpoint_ep{ep}.pth"
            torch.save({"episode": ep, "model_state": policy_net.state_dict()}, ckpt)
            print(f"Saved checkpoint {ckpt} | ep_reward {ep_reward:.3f}")

    # end training
    print("Training finished. Total steps:", total_steps)
    final_model = results_dir / "dqn_fraud_final.pth"
    torch.save({"episode": num_episodes, "model_state": policy_net.state_dict()}, final_model)

    env.close()

    # return artifacts
    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "rewards_per_episode": rewards_per_episode,
        "train_losses": train_losses,
        "results_dir": results_dir
    }

# %%
# Cell 6: Run training (this cell executes the train function)

results_dir = make_results_dir()
artifacts = train_dqn(num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, results_dir=results_dir)

# %%
# Cell 7: Final evaluation and metrics

policy = artifacts['policy_net']
results_dir = artifacts['results_dir']

print("Running deterministic evaluation to collect predictions...")
y_true, y_pred = evaluate_policy(policy, DEVICE)

if len(y_true) == 0:
    print("Evaluation did not return labels. Check that env.info contains 'true_label' or env.labels is available.")
else:
    auc_roc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("AUC-ROC:", auc_roc)
    print("AUC-PR:", auc_pr)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

    # save metrics
    (results_dir / "metrics.txt").write_text(
        f"AUC-ROC: {auc_roc:.6f}\nAUC-PR: {auc_pr:.6f}\n\nConfusion matrix:\n{cm}\n\nClassification report:\n{report}\n"
    )

# %%
# Cell 8: Plots (reward curve, PR, ROC, confusion matrix)

if len(artifacts['rewards_per_episode']) > 0:
    rewards = artifacts['rewards_per_episode']
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.title('Total reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.grid(True)
    if SAVE_PLOTS:
        plt.savefig(results_dir / 'reward_per_episode.png', dpi=200)
    plt.show()

# PR curve
if len(y_true) > 0:
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f'AUC-PR={auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.grid(True)
    if SAVE_PLOTS:
        plt.savefig(results_dir / 'precision_recall.png', dpi=200)
    plt.show()

    # ROC
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUC-ROC={auc_roc:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend()
        plt.grid(True)
        if SAVE_PLOTS:
            plt.savefig(results_dir / 'roc_curve.png', dpi=200)
        plt.show()
    except Exception as e:
        print('ROC plot error:', e)

    # Confusion matrix
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['pred0','pred1'])
    plt.yticks(tick_marks, ['true0','true1'])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center',
                     color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if SAVE_PLOTS:
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=200)
    plt.show()

# save classification report too
if len(y_true) > 0:
    (results_dir / 'classification_report.txt').write_text(report)

print('Artifacts saved to', results_dir)

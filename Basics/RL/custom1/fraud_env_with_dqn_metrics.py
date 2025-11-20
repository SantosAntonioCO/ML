# fraud_env_with_dqn_metrics.py
"""
FraudEnv (gymnasium) + DQN training skeleton with episode metrics (recall, classification report, precision, f1, AUC)
Atualizado:
 - Recompensa configurada (Opção A: agressivo contra FN)
 - Coleta de métricas por episódio
 - Gráficos gerados ao final: recall, f1, reward total, loss médio por episódio
 - Salva plots em results/plots (opcional)
"""

import os
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# -------------------------
# Reward scheme (Opção A — agressivo contra FN)
# -------------------------
REWARD_TP = +8.0   # True Positive
REWARD_FP = -3.0   # False Positive
REWARD_FN = -12.0  # False Negative (missed fraud)
REWARD_TN = +1.0   # True Negative

# -------------------------
# FraudEnv (gymnasium)
# -------------------------
class FraudEnv(gym.Env):
    """
    Fraud environment based on creditcard.csv (adapted for Gymnasium).
    State: feature vector (n_features,)
    Action: 0 = not_fraud, 1 = fraud

    step returns: obs, reward, terminated, truncated, info
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        csv_path: str = None,
        max_steps: int = 1000,
        apply_scaler: bool = False,
        scaler: Optional[StandardScaler] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # CSV path (by default it looks for ./dataset/creditcard.csv)
        if csv_path is None:
            csv_path = os.path.join(os.getcwd(), "dataset", "creditcard.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found. Adjust csv_path or include it.")

        # Load CSV
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

        # Features and labels
        X = self.df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
        y = self.df["Class"].to_numpy(dtype=np.int64)

        # Optional scaler (fit on whole dataset or user-supplied)
        if apply_scaler:
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(X)  # ideally fit on TRAIN split externally
            else:
                self.scaler = scaler
            X = self.scaler.transform(X).astype(np.float32)
        else:
            self.scaler = None

        self.features = X
        self.labels = y

        # Actions: 0 (not_fraud), 1 (fraud)
        self.ACTION_LOOKUP = {0: "not_fraud", 1: "fraud"}
        self.action_space = spaces.Discrete(len(self.ACTION_LOOKUP))

        # Observations: features vector (Box)
        n_features = self.features.shape[1]
        low = np.full((n_features,), -np.finfo(np.float32).max, dtype=np.float32)
        high = np.full((n_features,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state
        self.current_state_index = 0
        self.turns = 0
        self.sum_rewards = 0.0
        self.episode_over = False
        self.max_steps = int(max_steps)

        # RNG
        self._np_random = np.random.default_rng(seed)

        # Start state
        self.observation = self._get_random_initial_state()

    # ----------------------
    # Gymnasium API
    # ----------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: int (0 or 1)
        Returns: obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"invalid action: {action}"

        self.turns += 1
        self.last_action = int(action)

        # calculate reward (based on current state index)
        reward = self._get_reward(action)

        terminated = False
        truncated = False

        # Move to next state
        try:
            next_state = self._get_next_state()
        except IndexError:
            # If for some reason random index logic fails, treat as end
            terminated = True
            next_state = self.observation  # keep last state

        self.observation = next_state

        # Termination only by max steps (truncation) or natural end
        if self.turns >= self.max_steps:
            truncated = True

        info = {}
        return self.observation, float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Modern Gymnasium reset: returns (obs, info)
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.turns = 0
        self.sum_rewards = 0.0
        self.episode_over = False
        self.current_state_index = int(self._np_random.integers(0, len(self.features)))
        self.observation = self.features[self.current_state_index]
        info = {}
        return self.observation, info

    def render(self, mode="human"):
        idx = self.current_state_index
        lbl = int(self.labels[idx])
        print(f"[FraudEnv] idx={idx} label={lbl} sum_rewards={self.sum_rewards:.2f} turns={self.turns}")

    def close(self):
        pass

    # ----------------------
    # Auxiliary methods
    # ----------------------
    def _take_action(self, action_index: int) -> int:
        assert action_index < len(self.ACTION_LOOKUP)
        self.last_action = int(action_index)
        return self.last_action

    def _get_random_initial_state(self) -> np.ndarray:
        nrand = int(self._np_random.integers(0, len(self.features)))
        self.current_state_index = nrand
        return self.features[nrand]

    def _get_reward(self, predicted_action: int) -> float:
        """
        Reward scheme defined by constants at top of file.
        """
        labelled_action = int(self.labels[self.current_state_index])

        if predicted_action == 1 and labelled_action == 1:
            reward = REWARD_TP  # True Positive
        elif predicted_action == 1 and labelled_action == 0:
            reward = REWARD_FP  # False Positive
        elif predicted_action == 0 and labelled_action == 1:
            reward = REWARD_FN  # False Negative
        else:
            reward = REWARD_TN  # True Negative

        self.sum_rewards += reward
        return float(reward)

    def _get_next_state(self) -> np.ndarray:
        # Choose next sample randomly to avoid walking deterministically through CSV
        new_state_index = int(self._np_random.integers(0, len(self.features)))
        # ensure we always change index (optional)
        if new_state_index == self.current_state_index and len(self.features) > 1:
            new_state_index = (new_state_index + 1) % len(self.features)
        self.current_state_index = int(new_state_index)
        return self.features[self.current_state_index]

    def seed(self, seed: Optional[int] = None):
        self._np_random = np.random.default_rng(seed)
        return [seed]

    # ----------------------
    # Compatibility helpers
    # ----------------------
    def step_legacy(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Compatibility wrapper for older Gym code that expects:
        next_state, reward, done, info
        done = terminated or truncated
        """
        obs, reward, terminated, truncated, info = self.step(action)
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info

    def current_label(self) -> int:
        """Return label for the current state index (useful when collecting y_true before stepping)."""
        return int(self.labels[self.current_state_index])

# -------------------------
# DQN skeleton and training showing metrics collection
# -------------------------

class DQNNet(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_outputs),  # output: Q(s,a) for each action
        )

    def forward(self, x):
        return self.net(x)

# Replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

# Utility: compute episode-level metrics
def compute_metrics(y_true, y_pred):
    metrics = {}
    if len(y_true) == 0:
        return metrics
    try:
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
        # AUC only if both classes present in y_true
        if len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_pred))
            except Exception:
                metrics['auc'] = None
        else:
            metrics['auc'] = None

        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception as e:
        metrics['error'] = str(e)
    return metrics

def train_dqn_with_metrics(
    csv_path: str,
    n_episodes: int = 200,
    max_steps_per_episode: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 200,
    target_update: int = 10,
    memory_capacity: int = 10000,
    device: str = "cpu",
    moving_avg_window: int = 20,
    log_loss: bool = False,
    save_plots: bool = True,
    plots_dir: str = "results/plots",
):
    device = torch.device(device)
    env = FraudEnv(csv_path=csv_path, max_steps=max_steps_per_episode, apply_scaler=True)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQNNet(n_states, n_actions).to(device)
    target_net = DQNNet(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(capacity=memory_capacity)
    steps_done = 0

    recall_window = deque(maxlen=moving_avg_window)
    f1_window = deque(maxlen=moving_avg_window)
    precision_window = deque(maxlen=moving_avg_window)

    # Lists to store per-episode metrics (for plotting)
    recall_list = []
    f1_list = []
    precision_list = []
    reward_list = []
    loss_list = []

    def select_action(state_np):
        nonlocal steps_done
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1.0 * steps_done / eps_decay)
        steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                s = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
                qvals = policy_net(s)  # shape (1, n_actions)
                action = int(qvals.max(1)[1].item())
                return action
        else:
            return int(random.randrange(n_actions))

    def optimize_model():
        if len(memory) < batch_size:
            return None
        transitions = memory.sample(batch_size)
        batch = Transition(*transitions)

        # Convert to tensors
        state_batch = torch.from_numpy(np.vstack(batch.state)).float().to(device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)

        non_final_mask = torch.tensor([not d for d in batch.done], device=device, dtype=torch.bool)
        if non_final_mask.any().item():
            non_final_next_states = torch.from_numpy(
                np.vstack([s for s, d in zip(batch.next_state, batch.done) if not d])
            ).float().to(device)
        else:
            non_final_next_states = None

        # Q values for actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute expected Q values
        next_state_values = torch.zeros(batch_size, 1, device=device)
        if non_final_next_states is not None:
            next_q = target_net(non_final_next_states).max(1)[0].detach().unsqueeze(1)
            next_state_values[non_final_mask] = next_q

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        return float(loss.item())

    # Training episodes
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_rewards = 0.0
        y_true_episode = []
        y_pred_episode = []
        losses = []

        for t in range(max_steps_per_episode):
            true_label = env.current_label()
            action = select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)

            # store transition
            memory.push(state, action, next_state, reward, done)
            loss_val = optimize_model()
            if loss_val is not None:
                losses.append(loss_val)

            # Collect prediction and true label for metrics
            y_true_episode.append(true_label)
            y_pred_episode.append(int(action))

            state = next_state
            episode_rewards += reward

            if done:
                break

        metrics = compute_metrics(np.array(y_true_episode), np.array(y_pred_episode))
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        precision = metrics.get('precision', 0.0)

        recall_window.append(recall)
        f1_window.append(f1)
        precision_window.append(precision)

        # Save per-episode values for plotting
        recall_list.append(recall)
        f1_list.append(f1)
        precision_list.append(precision)
        reward_list.append(episode_rewards)
        avg_loss = np.mean(losses) if losses else np.nan
        loss_list.append(avg_loss)

        # Print episode summary
        loss_str = f" loss={avg_loss:.4f}" if (not np.isnan(avg_loss) and log_loss) else ""
        print(f"Episode {i_episode:4d} | steps={len(y_true_episode):3d} | reward_sum={episode_rewards:.2f} | recall={recall:.4f} | f1={f1:.4f} | precision={precision:.4f}{loss_str}")

        # Update target network
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print moving averages at useful intervals
        if i_episode % max(1, n_episodes // 10) == 0 or i_episode <= 10:
            avg_recall = np.nanmean(recall_list[-moving_avg_window:]) if len(recall_list) > 0 else 0.0
            avg_f1 = np.nanmean(f1_list[-moving_avg_window:]) if len(f1_list) > 0 else 0.0
            avg_precision = np.nanmean(precision_list[-moving_avg_window:]) if len(precision_list) > 0 else 0.0
            print(f"  >>> Moving averages (last {min(len(recall_list), moving_avg_window)} eps): recall={avg_recall:.4f}, f1={avg_f1:.4f}, precision={avg_precision:.4f}")

    print("Training complete.")

    # -------------------------
    # Plots
    # -------------------------
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        # Recall per episode
        plt.figure()
        plt.plot(np.arange(1, len(recall_list)+1), recall_list)
        plt.title("Recall por episódio")
        plt.xlabel("Episódio")
        plt.ylabel("Recall")
        plt.grid(True)
        recall_path = os.path.join(plots_dir, "recall_per_episode.png")
        plt.savefig(recall_path)
        plt.close()

        # F1 per episode
        plt.figure()
        plt.plot(np.arange(1, len(f1_list)+1), f1_list)
        plt.title("F1 por episódio")
        plt.xlabel("Episódio")
        plt.ylabel("F1")
        plt.grid(True)
        f1_path = os.path.join(plots_dir, "f1_per_episode.png")
        plt.savefig(f1_path)
        plt.close()

        # Reward per episode
        plt.figure()
        plt.plot(np.arange(1, len(reward_list)+1), reward_list)
        plt.title("Reward total por episódio")
        plt.xlabel("Episódio")
        plt.ylabel("Reward total")
        plt.grid(True)
        reward_path = os.path.join(plots_dir, "reward_per_episode.png")
        plt.savefig(reward_path)
        plt.close()

        # Loss per episode (avg)
        plt.figure()
        plt.plot(np.arange(1, len(loss_list)+1), loss_list)
        plt.title("Loss médio por episódio")
        plt.xlabel("Episódio")
        plt.ylabel("Loss médio (SmoothL1)")
        plt.grid(True)
        loss_path = os.path.join(plots_dir, "loss_per_episode.png")
        plt.savefig(loss_path)
        plt.close()

        print(f"Plots salvos em: {os.path.abspath(plots_dir)}")
        print(f" - {recall_path}")
        print(f" - {f1_path}")
        print(f" - {reward_path}")
        print(f" - {loss_path}")

    return policy_net, target_net, memory

# -------------------------
# If run as script, demonstrate usage
# -------------------------
if __name__ == "__main__":
    csv_path = os.path.join(os.getcwd(), "dataset", "creditcard.csv")
    if not os.path.exists(csv_path):
        print("creditcard.csv not found in ./dataset. Please place it there or adjust csv_path variable.")
    else:
        policy, target, memory = train_dqn_with_metrics(
            csv_path=csv_path,
            n_episodes=50,
            max_steps_per_episode=1000,
            batch_size=128,
            moving_avg_window=10,
            device="cpu",
            log_loss=False,
            save_plots=True,
        )

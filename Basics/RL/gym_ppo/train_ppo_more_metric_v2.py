import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gym_fraud_ppo.envs.gym_fraud_ppo import CreditCardFraudEnv
from sklearn.metrics import recall_score, f1_score, accuracy_score

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

env = CreditCardFraudEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy = PolicyNetwork(obs_dim, act_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

num_episodes = 50
all_rewards = []
all_recalls = []
all_f1s = []
all_accuracies = []

for episode in range(num_episodes):
    print(episode, " of ", num_episodes)
    obs, _ = env.reset()
    done = False
    log_probs = []
    rewards = []
    actions = []
    labels = []

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        obs, reward, done, _, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        actions.append(action.item())
        labels.append(env.labels[env.current_step - 1])

    returns = compute_returns(rewards)
    returns = torch.tensor(returns, dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    loss = -torch.sum(log_probs * returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Métricas
    total_reward = sum(rewards)
    recall = recall_score(labels, actions, zero_division=0)
    f1 = f1_score(labels, actions, zero_division=0)
    accuracy = accuracy_score(labels, actions)

    all_rewards.append(total_reward)
    all_recalls.append(recall)
    all_f1s.append(f1)
    all_accuracies.append(accuracy)

    print(f"Episode {episode+1}: Reward={total_reward}, Recall={recall:.2f}, F1={f1:.2f}, Accuracy={accuracy:.2f}")

# Gráficos
plt.plot(all_rewards, label="Reward")
plt.plot(all_recalls, label="Recall")
plt.plot(all_f1s, label="F1 Score")
plt.plot(all_accuracies, label="Accuracy")
plt.legend()
plt.title("PPO Training Performance")
plt.xlabel("Episode")
plt.ylabel("Metric")
plt.show()

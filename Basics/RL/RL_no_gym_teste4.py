import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================
# 1. Reading and preparing data
# ======================
data = pd.read_csv("creditcard.csv")
X = data.drop("Class", axis=1).values
y = data["Class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão treino/teste
n = len(X)
train_size = int(0.8 * n)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# ======================
# 2. RL ENV custom
# ======================
class FraudEnv(gym.Env):
    def __init__(self, X, y):
        super(FraudEnv, self).__init__()
        self.X = X
        self.y = y
        self.n = len(X)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)  # 0 = não fraude, 1 = fraude
        self.reset()

    def reset(self, seed=None, options=None):
        self.idx = 0
        self.correct = 0
        self.total = 0
        return self.X[self.idx], {}

    def step(self, action):
        reward = 1.0 if action == self.y[self.idx] else -1.0
        done = self.idx >= self.n - 1
        self.idx += 1
        if not done:
            obs = self.X[self.idx]
        else:
            obs = np.zeros_like(self.X[0])
        return obs, reward, done, False, {}

# ======================
# 3. Neural Net (DQN)
# ======================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# 4. Begining
# ======================
env = FraudEnv(X_train, y_train)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(obs_dim, act_dim).to(device)
target_net = DQN(obs_dim, act_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Replay buffer
memory = []
MEM_SIZE = 5000
BATCH_SIZE = 64
GAMMA = 0.95
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
EPSILON = 1.0

rewards_log = []

# ======================
# 5. Training Loop
# ======================
EPISODES = 30
for ep in tqdm(range(EPISODES), desc="Treinando DQN"):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)
    total_reward = 0

    for t in range(env.n):
        # Escolha de ação ε-greedy
        if random.random() < EPSILON:
            action = random.randrange(act_dim)
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                action = torch.argmax(q_values).item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).to(device)

        memory.append((state, action, reward, next_state, done))
        if len(memory) > MEM_SIZE:
            memory.pop(0)

        state = next_state
        total_reward += reward

        # Treinamento por amostragem
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            s, a, r, s2, d = zip(*batch)
            s = torch.stack(s)
            a = torch.LongTensor(a).unsqueeze(1).to(device)
            r = torch.FloatTensor(r).unsqueeze(1).to(device)
            s2 = torch.stack(s2)
            d = torch.BoolTensor(d).to(device)

            q_values = policy_net(s).gather(1, a)
            with torch.no_grad():
                next_q = target_net(s2).max(1)[0].unsqueeze(1)
                target = r + GAMMA * next_q * (~d)

            loss = criterion(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    rewards_log.append(total_reward)
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    target_net.load_state_dict(policy_net.state_dict())

print("Treinamento concluído.")

# ======================
# 6. Eval
# ======================
env_test = FraudEnv(X_test, y_test)
state, _ = env_test.reset()
state = torch.FloatTensor(state).to(device)

y_true, y_pred = [], []
for t in range(env_test.n):
    with torch.no_grad():
        q_values = policy_net(state)
        action = torch.argmax(q_values).item()

    y_true.append(env_test.y[env_test.idx])
    y_pred.append(action)

    next_state, _, done, _, _ = env_test.step(action)
    state = torch.FloatTensor(next_state).to(device)
    if done:
        break

# ======================
# 7. Metrics
# ======================
auc_roc = roc_auc_score(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_pred)
auc_pr = auc(recall, precision)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print("\n=== Resultados de Avaliação ===")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")
print("Matriz de confusão:\n", cm)
print("\nRelatório:\n", report)

# ======================
# 8. Plots
# ======================
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(rewards_log)
plt.title("Recompensa média por episódio")
plt.xlabel("Episódio")
plt.ylabel("Reward Total")

plt.subplot(1, 3, 2)
plt.plot(recall, precision)
plt.title(f"Precision-Recall Curve (AUC={auc_pr:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.subplot(1, 3, 3)
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.tight_layout()
plt.show()

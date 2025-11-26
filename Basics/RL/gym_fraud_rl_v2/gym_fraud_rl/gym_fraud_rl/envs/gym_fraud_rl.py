import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class FraudEnv(gym.Env):
    def __init__(self, csv_path):
        super().__init__()

        # Carregar dataset específico
        data = pd.read_csv(csv_path)

        self.features = data.drop(columns=["Class"]).values
        self.labels = data["Class"].values
        self.n_samples, self.n_features = self.features.shape

        # Espaços
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0 = não fraude, 1 = fraude

        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = self.features[self.current_step]
        return obs, {}

    def step(self, action):
        label = self.labels[self.current_step]
        reward = 1 if action == label else -1

        self.current_step += 1
        terminated = self.current_step >= self.n_samples

        obs = (
            self.features[self.current_step]
            if not terminated else np.zeros(self.n_features)
        )

        return obs, reward, terminated, False, {"label": label, "action": action}

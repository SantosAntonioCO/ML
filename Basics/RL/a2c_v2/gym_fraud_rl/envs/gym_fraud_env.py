import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class FraudEnv(gym.Env):
    """
    FraudEnv reads a CSV with a 'Class' column (0 or 1) and numeric features.
    Constructor:
        FraudEnv(csv_path, scaler=None, shuffle=False)
    If scaler is provided, it should implement .transform(X) like sklearn's StandardScaler.
    If shuffle=True the sample order is shuffled on reset.
    Each episode iterates the whole dataset once.
    """
    metadata = {"render_modes": []}

    def __init__(self, csv_path, scaler=None, shuffle=False):
        super().__init__()
        self.csv_path = csv_path
        self.scaler = scaler
        self.shuffle = shuffle
        self._load_data()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = not fraud, 1 = fraud

        self.current_step = 0
        self.order = np.arange(self.n_samples)

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        # Expect last column named 'Class'
        if 'Class' not in df.columns:
            raise ValueError("CSV must have a 'Class' column")
        X = df.drop(columns=["Class"]).values.astype(np.float32)
        y = df["Class"].values.astype(np.int64)
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        self.features = X
        self.labels = y
        self.n_samples, self.n_features = self.features.shape

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.shuffle:
            np.random.shuffle(self.order)
        obs = self.features[self.order[self.current_step]]
        return obs, {}

    def step(self, action):
        """
        Returns (obs, reward, terminated, truncated, info)
        The base reward is simply +1 if action == label else -1 (kept for compatibility).
        We'll use wrappers to provide task-specific reward shaping (F1 differential).
        """
        idx = self.order[self.current_step]
        label = int(self.labels[idx])
        base_reward = 1.0 if int(action) == label else -1.0

        self.current_step += 1
        terminated = self.current_step >= self.n_samples

        if terminated:
            obs = np.zeros(self.n_features, dtype=np.float32)
        else:
            obs = self.features[self.order[self.current_step]]

        info = {"label": label, "action": int(action), "base_reward": base_reward}
        return obs, float(base_reward), terminated, False, info

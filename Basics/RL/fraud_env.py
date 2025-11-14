# gym_fraud/envs/fraud_env.py
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class FraudEnv(gym.Env):
    """
    Ambiente de fraude baseado em creditcard.csv (adaptado para Gymnasium).
    Observação: cada estado é o vetor de features da linha do CSV (shape = (n_features,)).
    Ações: 0 = not_fraud, 1 = fraud
    Step returns: obs, reward, terminated, truncated, info
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, csv_path: str = None, max_steps: int = 1000):
        super().__init__()

        # Caminho do csv (por padrão procura ./dataset/creditcard.csv)
        if csv_path is None:
            csv_path = os.path.join(os.getcwd(), "dataset", "creditcard.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo '{csv_path}' não encontrado. Ajuste csv_path ou coloque o arquivo.")

        # Carrega o CSV (apenas uma vez)
        df = pd.read_csv(csv_path)
        # Guarda DataFrame e numpy array de features/labels
        self.df = df.reset_index(drop=True)
        # features como float32 numpy array
        self.features = self.df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
        self.labels = self.df["Class"].to_numpy(dtype=np.int64)

        # Ações: 0 (not_fraud), 1 (fraud)
        self.ACTION_LOOKUP = {0: "not_fraud", 1: "fraud"}
        self.action_space = spaces.Discrete(len(self.ACTION_LOOKUP))

        # Observations: vetor das features (Box)
        n_features = self.features.shape[1]
        # Inferir ranges por coluna (usamos -inf..+inf para simplicidade) — se preferir, calcule min/max
        low = np.full((n_features,), -np.finfo(np.float32).max, dtype=np.float32)
        high = np.full((n_features,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Estado interno
        self.current_state_index = 0
        self.turns = 0
        self.sum_rewards = 0.0
        self.episode_over = False
        self.max_steps = int(max_steps)

        # RNG
        self._np_random = np.random.default_rng()

        # Inicializa estado
        self.observation = self._get_random_initial_state()

    # ----------------------
    # Gymnasium API
    # ----------------------
    def step(self, action):
        """
        action: int (0 ou 1)
        Returns: obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Ação inválida: {action}"

        self.turns += 1
        # registra ação (padrão: apenas armazena)
        self.last_action = int(action)

        # calcula recompensa
        reward = self._get_reward(action)

        # avança estado
        terminated = False
        truncated = False

        # tenta obter próximo estado; se chegar ao final do dataset, termina episódio (terminated=True)
        try:
            next_state = self._get_next_state()
        except IndexError:
            # fim do dataset -> terminado
            terminated = True
            next_state = self.observation  # manter o último estado

        self.observation = next_state

        # condição de término extra (max steps ou soma de recompensas)
        if self.turns >= self.max_steps:
            truncated = True
        if self.sum_rewards > 2.0:
            terminated = True

        info = {}
        return self.observation, reward, bool(terminated), bool(truncated), info

    def reset(self, *, seed=None, options=None):
        """
        Reset moderno do Gymnasium: retorna (obs, info)
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
        # implementação simples: imprime índice e label
        idx = self.current_state_index
        lbl = int(self.labels[idx])
        print(f"[FraudEnv] idx={idx} label={lbl} sum_rewards={self.sum_rewards:.2f} turns={self.turns}")

    def close(self):
        pass

    # ----------------------
    # Métodos auxiliares
    # ----------------------
    def _take_action(self, action_index):
        assert action_index < len(self.ACTION_LOOKUP)
        self.last_action = int(action_index)
        return self.last_action

    def _get_random_initial_state(self):
        nrand = int(self._np_random.integers(0, len(self.features)))
        self.current_state_index = nrand
        return self.features[nrand]

    def _get_reward(self, predicted_action):
        """
        Reward simples: +1 se classificação correta, -1 se incorreta
        """
        labelled_action = int(self.labels[self.current_state_index])
        reward = 1.0 if labelled_action == int(predicted_action) else -1.0
        self.sum_rewards += reward
        return float(reward)

    def _get_next_state(self):
        new_state_index = self.current_state_index + 1
        # se exceder, levanta IndexError para que step() trate como episódio terminado
        if new_state_index >= len(self.features):
            raise IndexError("Reached end of dataset.")
        self.current_state_index = int(new_state_index)
        return self.features[self.current_state_index]

    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        return [seed]

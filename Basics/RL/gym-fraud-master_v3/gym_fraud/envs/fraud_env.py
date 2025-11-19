# gym_fraud/envs/fraud_env.py
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Opcional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False


class FraudEnv(gym.Env):
    """
    Fraud environment baseado no creditcard.csv (versão Gymnasium).
    Agora com:
    - divisão treino/teste controlada
    - n_nf_train / n_f_train / n_nf_test / n_f_test
    - opção de SMOTE apenas no treino
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        csv_path: str = None,
        max_steps: int = 1000,
        # seleção personalizada
        n_nf_train=None,
        n_f_train=None,
        n_nf_test=None,
        n_f_test=None,
        use_smote=False,
    ):
        super().__init__()

        # -----------------------------------------------------
        #  LOCALIZAÇÃO DO CSV
        # -----------------------------------------------------
        if csv_path is None:
            csv_path = os.path.join(os.getcwd(), "dataset", "creditcard.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")

        df = pd.read_csv(csv_path).reset_index(drop=True)
        df_nf = df[df["Class"] == 0]
        df_f = df[df["Class"] == 1]

        # -----------------------------------------------------
        #  AMOSTRAGEM PERSONALIZADA (TREINO / TESTE)
        # -----------------------------------------------------
        # treinamento
        if n_nf_train is None:
            df_nf_train = df_nf
        else:
            df_nf_train = df_nf.sample(n=min(n_nf_train, len(df_nf)), random_state=42)

        if n_f_train is None:
            df_f_train = df_f
        else:
            df_f_train = df_f.sample(n=min(n_f_train, len(df_f)), random_state=42)

        # teste
        if n_nf_test is None:
            df_nf_test = df_nf
        else:
            df_nf_test = df_nf.sample(n=min(n_nf_test, len(df_nf)), random_state=42)

        if n_f_test is None:
            df_f_test = df_f
        else:
            df_f_test = df_f.sample(n=min(n_f_test, len(df_f)), random_state=42)

        df_train = pd.concat([df_nf_train, df_f_train]).sample(frac=1, random_state=42)
        df_test  = pd.concat([df_nf_test, df_f_test]).sample(frac=1, random_state=42)

        # -----------------------------------------------------
        #  Aplicar SMOTE apenas no TREINO
        # -----------------------------------------------------
        if use_smote:
            if not HAS_SMOTE:
                raise RuntimeError("SMOTE solicitado mas imblearn não instalado.")

            print("🔧 Aplicando SMOTE no conjunto de treino...")
            X_train = df_train.drop(columns=["Class"]).values.astype(np.float32)
            y_train = df_train["Class"].values.astype(np.int64)

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            df_train = pd.DataFrame(X_res)
            df_train["Class"] = y_res

        # -----------------------------------------------------
        #  DEFINIR FEATURES / LABELS
        # -----------------------------------------------------
        self.df_train = df_train.reset_index(drop=True)
        self.features_train = self.df_train.drop(columns=["Class"]).to_numpy(dtype=np.float32)
        self.labels_train = self.df_train["Class"].to_numpy(dtype=np.int64)

        self.df_test = df_test.reset_index(drop=True)
        self.features_test = self.df_test.drop(columns=["Class"]).to_numpy(dtype=np.float32)
        self.labels_test = self.df_test["Class"].to_numpy(dtype=np.int64)

        # Ambiente usa somente treino
        self.features = self.features_train
        self.labels = self.labels_train

        # -----------------------------------------------------
        #  ACTION SPACE / OBSERVATION SPACE
        # -----------------------------------------------------
        self.ACTION_LOOKUP = {0: "not_fraud", 1: "fraud"}
        self.action_space = spaces.Discrete(2)

        n_features = self.features.shape[1]
        low = np.full((n_features,), -np.finfo(np.float32).max, dtype=np.float32)
        high = np.full((n_features,),  np.finfo(np.float32).max, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # -----------------------------------------------------
        #  STATE INTERNOS
        # -----------------------------------------------------
        self.current_state_index = 0
        self.turns = 0
        self.sum_rewards = 0.0
        self.episode_over = False
        self.max_steps = int(max_steps)
        self._np_random = np.random.default_rng()

        self.observation = self._get_random_initial_state()

    # =========================================================
    #  GYM API
    # =========================================================
    def step(self, action):
        assert self.action_space.contains(action)

        self.turns += 1
        self.last_action = int(action)
        reward = self._get_reward(action)

        terminated = False
        truncated = False

        try:
            next_state = self._get_next_state()
        except IndexError:
            terminated = True
            next_state = self.observation

        self.observation = next_state

        if self.turns >= self.max_steps:
            truncated = True
        if self.sum_rewards > 2.0:
            terminated = True

        info = {"label": int(self.labels[self.current_state_index])}
        return self.observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.turns = 0
        self.sum_rewards = 0.0
        self.episode_over = False

        self.current_state_index = int(self._np_random.integers(0, len(self.features)))
        self.observation = self.features[self.current_state_index]

        info = {"label": int(self.labels[self.current_state_index])}
        return self.observation, info

    def render(self):
        idx = self.current_state_index
        lbl = int(self.labels[idx])
        print(f"[FraudEnv] idx={idx} label={lbl} sum_rewards={self.sum_rewards:.2f} turns={self.turns}")

    def close(self):
        pass

    # =========================================================
    #  AUXILIARY
    # =========================================================
    def _get_random_initial_state(self):
        idx = int(self._np_random.integers(0, len(self.features)))
        self.current_state_index = idx
        return self.features[idx]

    def _get_reward(self, predicted):
        label = int(self.labels[self.current_state_index])
        reward = 1.0 if predicted == label else -1.0
        self.sum_rewards += reward
        return reward

    def _get_next_state(self):
        idx = self.current_state_index + 1
        if idx >= len(self.features):
            raise IndexError
        self.current_state_index = idx
        return self.features[idx]

    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        return [seed]

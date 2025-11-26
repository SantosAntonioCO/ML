# gym_fraud_rl package initializer: registers the env and exposes FraudEnv
from gymnasium.envs.registration import register

register(
    id="CreditCardFraud-v0",
    entry_point="gym_fraud_rl.envs.gym_fraud_env:FraudEnv",
)

from gym_fraud_rl.envs.gym_fraud_env import FraudEnv

__all__ = ["FraudEnv"]


"""Fraud Gym Enviornments."""

#from gym_fraud.envs.fraud_env import FraudEnv

from gymnasium import register

# registra o ambiente localmente: fraud-v0
register(
    id="fraud-v0",
    entry_point="gym_fraud.envs.fraud_env:FraudEnv",
)

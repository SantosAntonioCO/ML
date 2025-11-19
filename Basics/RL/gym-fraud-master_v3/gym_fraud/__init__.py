import logging
import gymnasium as gym
#from gym.envs.registration import register#

#register(
#    id='fraud-v0',
#    entry_point='gym_fraud.envs:FraudEnv',
#)


# gym_fraud/__init__.py
from gymnasium.envs.registration import register

# Registro do ambiente
register(
    id="Fraud-v0",
    entry_point="gym_fraud.envs:FraudEnv",
)

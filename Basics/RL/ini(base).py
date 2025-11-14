#import logging
#from gym.envs.registration import register
# Pacote gym_fraud
#from . import envs

#register(
#    id='fraud-v0',
#    entry_point='gym_fraud.envs:FraudEnv',
#)


from gymnasium.envs.registration import register
from . import envs

# Registra o ambiente fraud-v0
register(
    id="fraud-v0",
    entry_point="gym_fraud.envs.fraud_env:FraudEnv",
)

#import logging
#from gym.envs.registration import register
# Pacote gym_fraud
#from . import envs

#register(
#    id='fraud-v0',
#    entry_point='gym_fraud.envs:FraudEnv',
#)

from gym_fraud_ppo.envs.gym_fraud_ppo import CreditCardFraudEnv 

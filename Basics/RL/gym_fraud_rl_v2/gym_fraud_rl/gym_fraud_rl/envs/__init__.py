from gymnasium.envs.registration import register

register(
    id="CreditCardFraud-v0",
    entry_point="gym_fraud_rl.envs.gym_fraud_rl:FraudEnv",
)

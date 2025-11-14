#!/usr/bin/env python
#from setuptools import setup

#setup(name='gym_fraud',
#      version='0.0.1',
#      install_requires=['gym']
#)


#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="gym_fraud",
    version="0.0.2",
    description="Fraud detection RL environment using Gymnasium",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "pandas>=2.0.0",
        "numpy>=1.24",
    ],
    entry_points={
        "gymnasium.envs": [
            "FraudEnv = gym_fraud.envs.fraud_env:FraudEnv",
        ],
    },
)

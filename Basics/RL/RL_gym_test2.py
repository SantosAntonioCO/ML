# Code based on https://github.com/purvasingh96/gym-fraud/tree/master
import gymnasium as gym
import gym_fraud

env = gym.make("Fraud-v0")
obs, info = env.reset()
print("Primeira observação:", obs[:5], "...")
print("Espaço de observação:", env.observation_space)
print("Espaço de ação:", env.action_space)

obs, reward, terminated, truncated, info = env.step(1)
print("Reward:", reward, "| Terminated:", terminated)

#conda create -n gymfraud_py311 python=3.11 -y
#conda activate gymfraud_py311
#pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 matplotlib==3.8.4 gymnasium==0.29.1
#--------------------------------
#Make substitutionon top:

#import gym
#por:
#import gymnasium as gym
#------------------------------
#Error
#e . Obtaining RL/gym-fraud-master/gym-fraud Preparing metadata (setup.py) ... error error: subprocess-exited-with-error
#Solution
#Open file:
#gym-fraud-master/gym-fraud/setup.py
#The file maybe starts like:

#from setuptools import setup

#setup(
#    name='gym_fraud',
#    version='0.0.1',
#    install_requires=['gym']
#)
#Change code to this:
#from setuptools import setup, find_packages

#setup(
#    name="gym_fraud",
#    version="0.0.1",
#    packages=find_packages(include=["gym_fraud", "gym_fraud.*"]),
#    install_requires=[
#        "gymnasium>=0.29.1",
#        "numpy>=1.23.5",
#        "pandas>=1.5.3",
#        "scikit-learn>=1.2.2",
#        "matplotlib>=3.7.1"
#    ],
#)
#--------------------------------------

#Steps to correct ModuleNotFoundError: No module named 'gym'
#fast option (recommended): create alias “gym” → “gymnasium”

#Open file:

# ..RL\gym-fraud-master\gym-fraud\gym_fraud\__init__.py


#Add these lines at the begining file (before any import):

#import sys
#import gymnasium as gym
#sys.modules["gym"] = gym

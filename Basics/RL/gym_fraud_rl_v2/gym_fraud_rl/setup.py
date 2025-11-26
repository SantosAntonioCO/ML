from setuptools import setup, find_packages
#pip install -e .
setup(
    name="gym_fraud_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0"
    ],
    description="Ambiente personalizado de detecção de fraude para RL com DQN em Gymnasium e PyTorch",
    author="Git",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

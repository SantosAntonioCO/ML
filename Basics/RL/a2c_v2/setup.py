from setuptools import setup, find_packages
#pip install -e .
setup(
    name="gym_fraud_rl",
    version="0.1.0",
    description="Gymnasium environment for credit-card fraud (with F1-macro RL training example)",
    author="You",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.22",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "matplotlib>=3.5",
        "torch>=1.12"
    ],
    include_package_data=True,
    python_requires=">=3.8",
)

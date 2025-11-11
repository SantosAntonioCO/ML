from setuptools import setup, find_packages

setup(
    name="gym_fraud",
    version="0.0.1",
    packages=find_packages(include=["gym_fraud", "gym_fraud.*"]),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scikit-learn>=1.2.2",
        "matplotlib>=3.7.1"
    ],
)

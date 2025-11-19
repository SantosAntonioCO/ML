#!/usr/bin/env python
#from setuptools import setup

#setup(name='gym_fraud',
#      version='0.0.1',
#      install_requires=['gym']
#)



from setuptools import setup, find_packages

setup(
    name="gym_fraud",
    version="0.3.0",
    description="Custom Gymnasium environment for fraud detection using the creditcard.csv dataset.",
    author="Test",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    include_package_data=True,
)

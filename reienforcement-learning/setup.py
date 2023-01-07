#!/usr/bin/env python

from setuptools import setup

setup(
    name="reinforcement-learning",
    version="1.0",
    description="A package for reinforcement learning projects",
    author="Shiva Reddy",
    author_email="shiva.reddy37@gmail.com",
    install_requires=[
        "numpy==1.19.3",
        "tensorflow==2.6.0",
        "opencv-python",
        "matplotlib",
        "sklearn",
        "argh",
        "gym-super-mario-bros==7.3.0",
        "nes_py",
        "stable-baselines3[extra]"
    ],
)

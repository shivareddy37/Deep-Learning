#!/usr/bin/env python
# ------------------------------------------------------------------------------
# setup.py
# ------------------------------------------------------------------------------

from setuptools import setup, find_packages
import os
from glob import glob

setup(
    name="midas",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.5.3",
        "numpy==1.21.6",
        "opencv-contrib-python==4.6.0.66",
        "opencv-python==4.6.0.66",
        "timm==0.6.12",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "imutils==0.5.4",
    ],
)

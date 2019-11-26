from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This mini-project aims to estimate 2D skeleton data from 3D skeleton data. More specifically 2D skeleton data for IR frames from the kinect v2. Exact formulas exist, but we are too lazy to find them ourselves. The motivation behind this work is that the NTU RGB+D dataset provides 2D IR skeleton data, while the PKU MMD dataset does not. And we need them.',
    author='Alban Main de Boissiere',
    license='MIT',
)

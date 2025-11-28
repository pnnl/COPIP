#!/bin/bash

# load baskerville modules
module purge
module load baskerville
module load bask-apps/live
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
module load scikit-learn/1.3.1-gfbf-2023a

# mk dir in current folder - only needed on first run!
# mkdir SEGP_venv

# create the virtual environment - only needed on first run!
# python -m venv --system-site-packages SEGP_venv

# activate virtual environment
source SEGP_venv/bin/activate

# install Python software (not included in Baskerville modules) - only needed on first run!
# pip install gpytorch
# pip install seaborn

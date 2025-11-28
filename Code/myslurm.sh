#!/bin/bash
#
# [myslurm.sh]
#
#SBATCH --qos=turing
#SBATCH --account=vjgo8416-lurienet # project name.
#SBATCH --time=00-15:00:00
#SBATCH --job-name=exp
#SBATCH --output=Outputs/%x-%j.out # sets job-name (%x) - output (%j) as out and stats filename.
#SBATCH --gpus=1 # number of gpus to request
#SBATCH --cpus-per-gpu=36 # recommended number of cpus per gpu (specific to Baskerville) - equivalent to a whole node.
#SBATCH --nodes=1 # ensures gpus are on one node - not always neccessary
#SBATCH --constraint=a100_80 # use this command to specify the GPU type. Remove if large memory isn't required.

# Create/load environment
source make_venv.sh

# call script
# python DataGeneration/DataGeneration_Particle.py
# python SEGP/Train_GP.py
# python SEGP/Evaluate_GP.py
python SEGP_VAE/Train_VAE.py

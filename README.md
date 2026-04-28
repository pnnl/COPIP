# SEGP-VAE
This code acompanies the papers [Stability Enhanced Gaussian Process Variational Autoencoders](https://arxiv.org/pdf/2604.09331)

### Authors:
* Carl R Richardson (carl.richardson@eng.ox.ac.uk)
* Jichen Zhang (jichen.zhang@eng.ox.ac.uk)
* Ethan King (ethan.king@pnnl.gov)
* Ján Drgoˇna (jdrgona1@jh.edu.)

## Prerequisites
All the code is written in Python and predominantly in PyTorch. This must be installed along with several other standard libraries such as Numpy, Matplotlib, etc. For easy use, please use the same file structure as below.

## Overview
The repository is organised as follows:
* `Code`
  * `DataGeneration`
    *  `DataGeneration_Particle.py` : Script for generating video and trajectory data of the particle spiralling in the plane.
  * `SEGP`
    * `SEGP.py` : SEGP model class.
    * `Train_GP.py` : Script for training the SEGP.
    * `Evaluate_GP.py` : Script for plotting figures used to evaluate GP.
  * `SEGP_VAE`
    * `VAE.py` : VAE encoder and decoder classes.
    * `Train_VAE.py` : Script for training the VAE.
  * `Utils`
    * `Utils.py` : Script containing miscellaneous utility functions.
 * `Data` : Directory for storing datasets.
 * `Models` : Directory for storing models and associated files.

## To replicate results in paper
- Download dependencies.
- Add root directory to relevant scripts.
- Generate data using `DataGeneration_Particle.py` copying data parameters specified in paper. This will be stored in the `Data` directory.
- Run `Train_VAE.py` setting following parameter setting from paper. Returned modelled will be stored in `Models`.

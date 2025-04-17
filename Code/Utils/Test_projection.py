import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

root = '/bask/projects/v/vjgo8416-lurienet/SEGP/'

# Directory SEGP model is stored.
GP_path = root + 'Code/SEGP/'

# Directory VAE model is stored.
VAE_path = root + 'Code/SEGP_VAE/'

# Directory utility functions are stored.
utils_path = root + 'Code/Utils/'

# Add directories to path.
if GP_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(GP_path)

if VAE_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(VAE_path)

if utils_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(utils_path)

# Import custom files.
import SEGP
from VAE import VAEEncoder, VAEDecoder
from Train_VAE import get_dataloaders
from Utils import plot_latents, MSE_projection



def main():

    # Hardware settings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print("Device in use is: ", device)

    # Directory where data is stored.
    dataset_number = 3
    data_path = root + 'Data/Dataset{0}'.format(dataset_number)

    # Directory where model is stored.
    model_path = root + 'Models/'
    model_name = 'SEGP_VAE'
    exp_no = 23
    model_idx = 211
    model_dir = model_path + model_name + '/Exp_{:03d}/'.format(exp_no)

    # Import data.
    data_setup = np.load(data_path + '/data_setup.pkl', allow_pickle=True)
    T = torch.from_numpy( np.load(data_path + '/T.npy') ).to(device)
    dT = torch.from_numpy( np.load(data_path + '/dT.npy') ).to(device)
    mean_U = torch.from_numpy( np.load(data_path + '/mean_U.npy') ).unsqueeze(1).to(device)
    mean_dU = torch.from_numpy( np.load(data_path + '/mean_dU.npy') ).unsqueeze(1).to(device)
    dZ = torch.from_numpy( np.load(data_path + '/dZ.npy') ).to(device)
    vid_batch = torch.from_numpy( np.load(data_path + '/vid_batch.npy') ).to(device)

    M, Q, N, d, _ = vid_batch.shape
    K = T.shape[0]

    print('M = {0} \t Q = {1} \t N = {2} \t d = {3} \t K = {4}'.format(M, Q, N, d, K))

    # Instantiate model.
    m = data_setup['m']
    n = data_setup['n']
    p = data_setup['p']
    lt = data_setup['lt']
    tmax = data_setup['tmax']
    mean_x0 = torch.tensor([data_setup['mean_r'], data_setup['mean_theta']])
    covar_x0 = data_setup['sigma'] * torch.eye(n)
    covar_noise = data_setup['var_noise'] * torch.eye(m*N, dtype=torch.double)
    h_dim = 500
    d = data_setup['d']
    A_train = None # torch.tensor([[-l, 0.0], [0.0, 0.0]])
    B_train = torch.tensor([[0.0], [1.0]])
    C_train = torch.eye(m)
    D_train = torch.zeros(m,p)

    GP = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A_train, B_train, C_train, D_train).to(device)
    enc = VAEEncoder(d*d, h_dim, m).to(device)
    dec = VAEDecoder(m, h_dim, d*d).to(device)


    # Load model.
    GP.load_state_dict( torch.load(model_dir + 'GP/epoch{:03d}.pt'.format(model_idx), map_location=device, weights_only=True) )
    enc.load_state_dict( torch.load(model_dir + 'Encoder/epoch{:03d}.pt'.format(model_idx), map_location=device, weights_only=True) )
    dec.load_state_dict( torch.load(model_dir + 'Decoder/epoch{:03d}.pt'.format(model_idx), map_location=device, weights_only=True) )
    GP.eval()
    enc.eval()
    dec.eval()

    # Model outputs.
    rand_idx = np.random.randint(low=0, high=M)
    print('Random index was', rand_idx)

    mean_lhood, var_lhood = enc(vid_batch[rand_idx].float()) # both (bs, N, m)
    mean_prior, covar_prior = GP(T, dT, tmax, mean_U, mean_dU) # (N, m) and (m*N, m*N)
    mean_post, covar_post = GP.posterior(var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed(), mean_lhood, mean_prior, covar_prior) # (bs, N, m), (bs, mN, mN)
    samples = GP.sample(mean_post, covar_post) # (bs, N, m, n_samples=3)
    samples = samples.mean(dim=3) # (bs, N, m)
    p_theta_logits = dec(samples).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d)
    p_theta = torch.sigmoid(p_theta_logits) # shape = (bs, N, d, d)

    with torch.no_grad():
        rot_mean_post, W, MSE, rot_covar_post = MSE_projection(mean_post.cpu().numpy(), dZ[rand_idx].cpu().numpy(), covar_post.cpu().numpy() ) # torch
        torch.save(W, model_dir + 'W.pt' )
        print('MSE was', MSE)

        # Plot.
        nplots = 3
        string = 'MSE_scale_test'
        plot_latents(model_dir + 'Plots', string, vid_batch[rand_idx].unflatten(dim=0, sizes=(1,Q)), dZ[rand_idx].unflatten(dim=0, sizes=(1,Q)), dT, tmax, nplots, recon_batch=p_theta, recon_traj=rot_mean_post, recon_covar=rot_covar_post)

    
    return 0



if __name__ == '__main__':
    main()

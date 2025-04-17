# Imports.
import sys
import torch
import numpy as np


root = '/bask/projects/v/vjgo8416-lurienet/SEGP/'

# Directories of SEGP, VAE and Utils.
GP_path = root + 'Code/SEGP/'
VAE_path = root + 'Code/SEGP_VAE/'
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
from Utils import plot_latents, scale_mean, scale_covar



def main():

    # Hardware settings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print("Device in use is: ", device)

    # Directory where data is stored.
    dataset_number = 2
    data_path = root + 'Data/Dataset{0}'.format(dataset_number)

    # Directory where models are stored.
    model_path = root + 'Models/'
    model_name = 'SEGP_VAE'
    exp_no = 14
    model_idx = 199
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
    rand_batch = np.random.randint(low=0, high=M)
    
    print('rand batch', rand_batch)

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
    
    l = data_setup['l']
    A_train = torch.tensor([[-l, 0.0], [0.0, 0.0]])
    B_train = torch.tensor([[0.0], [1.0]])
    C_train = torch.eye(m)
    D_train = torch.zeros(m,p)

    GP = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A_train, B_train, C_train, D_train).to(device)
    enc = VAEEncoder(d*d, h_dim, m).to(device)
    dec = VAEDecoder(m, h_dim, d*d).to(device)

    # Load trained VAE.
    enc.load_state_dict( torch.load(model_dir + 'Encoder/epoch{:03d}.pt'.format(model_idx),
                                      map_location=device, weights_only=True) )
    dec.load_state_dict( torch.load(model_dir + 'Decoder/epoch{:03d}.pt'.format(model_idx),
                                      map_location=device, weights_only=True) )
    enc.eval()
    dec.eval()

    # No scaling test.
    mean_lhood, var_lhood = enc(vid_batch[rand_batch].float() ) # (bs, N, m)
    mean_prior, covar_prior = GP(T, dT, tmax, mean_U, mean_dU) # (N,m), (mN, mN)
    mean_post, covar_post = GP.posterior(var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed(), mean_lhood, mean_prior, covar_prior)
    samples = GP.sample(mean_post, covar_post) # (bs, N, m, n_samples=3)
    samples = samples.mean(dim=3) # (bs, N, m)
    p_theta_logits = dec(samples).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d)
    p_theta = torch.sigmoid(p_theta_logits) # shape = (bs, N, d, d)

    nplots = 3
    string = 'unscaled'
    with torch.no_grad():
        plot_latents(model_dir + 'Plots', string, vid_batch[rand_batch].unflatten(dim=0, sizes=(1,Q)), dZ[rand_batch].unflatten(dim=0, sizes=(1,Q)), dT, tmax, nplots, recon_batch=p_theta, recon_traj=mean_post, recon_covar=covar_post)

    # Scaling vectors.
    dZ0_max = torch.max(dZ[:,:,:,0])
    dZ1_max = torch.max(dZ[:,:,:,1])
    down_scaling_vec = torch.tensor([1/dZ0_max, 1/dZ1_max])
    up_scaling_vec = torch.tensor([dZ0_max, dZ1_max])
    
    print('down scaling vec', down_scaling_vec)
    print('up scaling vec:', up_scaling_vec)
    
    # scaling test
    mean_lhood, var_lhood = enc(vid_batch[rand_batch].float() ) # (bs, N, m)
    mean_prior, covar_prior = GP(T, dT, tmax, mean_U, mean_dU) # (N,m), (mN, mN)

    mean_lhood_scaled = scale_mean(down_scaling_vec, mean_lhood) # (bs, N, m)
    mean_prior_scaled = scale_mean(down_scaling_vec, mean_prior) # (N, m)
    covar_lhood_scaled = scale_covar(down_scaling_vec, var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed() ) # (bs, mN, mN)
    covar_prior_scaled = scale_covar(down_scaling_vec, covar_prior) # (mN, mN)

    mean_post_scaled, covar_post_scaled = GP.posterior(covar_lhood_scaled, mean_lhood_scaled, mean_prior_scaled, covar_prior_scaled)
    
    mean_post = scale_mean(up_scaling_vec, mean_post_scaled) # (bs, N, m)
    covar_post = scale_covar(up_scaling_vec, covar_post_scaled) # (bs, mN, mN)

    samples = GP.sample(mean_post, covar_post) # (bs, N, m, n_samples=3)
    samples = samples.mean(dim=3) # (bs, N, m)
    p_theta_logits = dec(samples).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d)
    p_theta = torch.sigmoid(p_theta_logits) # shape = (bs, N, d, d)  

    nplots = 3
    string1 = 'scaled'
    string2 = 'scaled_unscaled'
    dZ_scaled = torch.zeros(M,Q,N,m)
    dZ_scaled[:,:,:,0] = down_scaling_vec[0] * dZ[:,:,:,0]
    dZ_scaled[:,:,:,1] = down_scaling_vec[1] * dZ[:,:,:,1]
    with torch.no_grad():
        plot_latents(model_dir + 'Plots', string1, vid_batch[rand_batch].unflatten(dim=0, sizes=(1,Q)), dZ_scaled[rand_batch].unflatten(dim=0, sizes=(1,Q)), dT, tmax, nplots, recon_batch=p_theta, recon_traj=mean_post_scaled, recon_covar=covar_post_scaled)
        plot_latents(model_dir + 'Plots', string2, vid_batch[rand_batch].unflatten(dim=0, sizes=(1,Q)), dZ[rand_batch].unflatten(dim=0, sizes=(1,Q)), dT, tmax, nplots, recon_batch=p_theta, recon_traj=mean_post, recon_covar=covar_post)
    
    
    return 0



if __name__ == '__main__':
    main()

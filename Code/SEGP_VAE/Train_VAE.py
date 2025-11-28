

"""
Script for training the VAE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import sys
import os
import time
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
from Utils import plot_latents, scale_mean, scale_covar, MSE_projection



def get_dataloaders(vid_batch:torch.tensor, dY:torch.tensor, batches:int, bs:int, test_split:float, seed:int, device):
    """
    Function for splitting data into train and test dataloaders. Latent states corresponding to
    test loader data are also returned for plotting purposes.
    args:
        vid_batch: data to split with shape = (M, Q, N, d, d).
        dY: latent states to split in same way for plots (M, Q, N, m).
        batches: number of batches.
        bs: batch size.
        test_split: ratio of batches to reserve for testing.
        seed: random seed.
        device: hardware device.
    returns:
        train_loader: dataloader with shape = (batches - N_test, bs, N, d, d).
        test_loader: dataloader with shape = (N_test, bs, N, d, d).
        dY_test: latent states corresponding to test loader; shape = (N_test, bs, N, m).
    """

    M, Q, N, d, _ = vid_batch.shape
    m = dY.shape[3]

    vid_batch = vid_batch.view(M*Q, N, d, d) # shape = (M * Q, N, d, d)
    dY = dY.view(M*Q, N, m) # shape = (M * Q, N, m)

    # reshape into shapes = (batches, bs, N, d, d), (batches, bs, N, m)
    if batches * bs != M * Q:
        print('M =', M)
        print('Q =', Q)
        raise Exception("batches * bs != M * Q!")
    else:
        vid_batch = vid_batch.view(batches, bs, N, d, d)
        dY = dY.view(batches, bs, N, m)

    # passing in the same seed will result in the same idx.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    idx = torch.randperm(vid_batch.shape[0], generator=generator)
    
    # shuffle
    vid_batch = vid_batch[idx]
    dY = dY[idx]

    # split into train and test sets.
    N_test = int(batches * test_split)
    if N_test < 1:
        raise Exception("No batches reserved for testing!")

    # passing in the same seed will result in the same test_idx.
    rng = np.random.default_rng(seed)
    test_idx = rng.integers(low=0, high=batches, size=N_test)

    train_idx = np.delete( np.arange(0, batches), test_idx)

    test_loader = DataLoader(vid_batch[test_idx], batch_size=1, shuffle=False)
    train_loader = DataLoader(vid_batch[train_idx], batch_size=1, shuffle=False)
    dY_test = dY[test_idx]

    return train_loader, test_loader, dY_test



class ELBO(nn.Module):
    """
    Class for computing the two terms of the ELBO objective: reconstruction term and KL divergence, 
    each averaged across a batch of data with shape (bs, N, m).
    """

    def __init__(self):
        super().__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.register_buffer('pi', torch.tensor(math.pi) )



    def recon(self, vid_batch:torch.tensor, p_theta_logits:torch.tensor):
        """
        Computes the average reconstruction term across a batch of data.
        args:
            vid_batch: Observed images (bs, N, d, d).
            p_theta_logits: Logits of Bernouli distribution for each black/white pixel of image (bs, N, d, d).
        returns:
            recon_av: average reconstruction term (scalar).
        """

        bs, N, d, _  = p_theta_logits.shape

        recon = -self.BCE(p_theta_logits, vid_batch) # shape = (bs, N, d, d)
        recon = recon.sum( dim=(3,2) ) # summed over image. shape = (bs, N)
        recon = recon.mean(dim=1) # averaged over time. shape = (bs)

        recon_av = recon.mean()

        return recon_av



    def kl_divergence(self, mean_post:torch.tensor, covar_post:torch.tensor, 
                      mean_prior:torch.tensor, covar_prior:torch.tensor):
        """
        Compute the average KL divergence (across a batch) between two multivariate normal distributions
        q(z) = N(mean_post, covar_post) and p(z) = N(mean_prior, covar_prior) using the Cholesky decomposition.

        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Introduction_and_context
        
        args:
            mean_post: Mean of the variational posterior distribution q(z) with shape (bs, N, m).
            covar_post: Covariance matrix of the variational posterior distribution q(z) with shape (bs, m*N, m*N).
            mean_prior: Mean of the prior distribution p(z) with shape (N, m).
            covar_prior: Covariance matrix of the prior distribution p(z) with shape (m*N, m*N).
        returns:
            kl_div_av (scalar).
        """
        
        bs, N, m = mean_post.shape
        
        # Perform Cholesky decomposition to get lower triangular matrices L_p and L_q.
        L_prior = torch.linalg.cholesky(covar_prior, upper=False) # shape = (m*N, m*N).
        L_post = torch.linalg.cholesky(covar_post, upper=False) # shape = (bs, m*N, m*N).
        
        # Solve covar_post * alpha = covar_prior for alpha.
        alpha = torch.cholesky_solve(covar_prior.repeat(bs,1,1), L_post) # shape = (bs , m*N, m*N).
        
        # Compute the trace term: Tr(covar_post^{-1} * covar_prior) = Tr(alpha).
        term1 = torch.sum( torch.diagonal(alpha, dim1=1, dim2=2), dim=1)  # shape = (bs).

        term2 = torch.tensor(N * m)  # Dimensionality of the Gaussian distributions.
        term2 = term2.repeat(bs) # shape = (bs).

        # Solve covar_post * alpha = (mean_post - mean_prior) for alpha.
        mean_post = torch.transpose(mean_post, 1, 2).reshape(bs, m*N).unsqueeze(2) # shape = (bs, m*N, 1).
        mean_prior = torch.transpose( mean_prior.repeat(bs, 1, 1), 1, 2).reshape(bs, m*N).unsqueeze(2) # shape = (bs, m*N, 1)
        alpha = torch.cholesky_solve(mean_post - mean_prior, L_post) # shape = (bs , m*N, 1).

        # Compute the quadratic term: (mean_post - mean_prior)^T * covar_post^-1 * (mean_post - mean_prior).
        term3 = torch.bmm((mean_post - mean_prior).transpose(1,2), alpha) # shape = (bs, 1, 1).
        term3 = term3.squeeze(2).squeeze(1) # shape = (bs).
        
        # Compute the log determinant term using Cholesky factors.
        term4 = torch.sum( torch.log( torch.diagonal(L_post, dim1=1, dim2=2) ), dim=1 ) # shape = (bs).
        term4 = term4 - torch.sum( torch.log( torch.diagonal(L_prior.repeat(bs,1,1), dim1=1, dim2=2) ), dim=1 ) # shape = (bs).
        term4 = 2 * term4 # shape = (bs).
        
        # check for infs.
        if torch.any( torch.isinf(term1) ):
            print('KL div: term1 contains infs!')

        if torch.any( torch.isinf(term3) ):
            print('KL div: term3 contains infs!')

        if torch.any( torch.isinf(term4) ):
            print('KL div: term4 contains infs!')
        
        # Compute the KL divergence across the batch.
        kl_div = 0.5 * (term1 - term2 + term3 + term4) # shape = (bs).

        kl_div_av = kl_div.mean()
        
        return kl_div_av


    def forward(self, vid_batch:torch.tensor, p_theta_logits:torch.tensor, mean_prior:torch.tensor, 
                covar_prior:torch.tensor, mean_post:torch.tensor, covar_post:torch.tensor, kl_weight:float):
        """
        Computes ELBO objective.
        args:
            vid_batch: Observed images (bs, N, d, d).
            p_theta_logits: Logits of Bernouli distribution for each black/white pixel of image (bs, N, d, d).
            mean_prior: SEGP mean function (N, m).
            covar_prior: SEGP covariance matrix (m*N, m*N).
            mean_post: SEGP posterior mean function (bs, N, m).
            covar_post: SEGP posterior covariance matrix (bs, m*N, m*N).
            kl_weight: weight for KL term of the ELBO objective. 
        returns:
            elbo_obj: ELBO objective (scalar).
        """

        self.recon_obj = self.recon(vid_batch, p_theta_logits) # should always be negative.
        
        self.kl_obj = - self.kl_divergence(mean_post, covar_post, mean_prior, covar_prior) # should always be negative.
        
        self.elbo_obj = self.recon_obj + ( kl_weight * self.kl_obj )

        return self.elbo_obj



class KLAnnealingScheduler():
    """
    Class for computing the weight of the KL divergence term (in the ELBO objective) to be used at each epoch.
    args:
        initial_beta: initial weight of KL term.
        final_beta: final weight of KL term.
        warm_up: number of epochs to linearly increase beta from initial_beta to final_beta. After warm_up, beta = final_beta.
        start_warm_up: epoch to begin the warm up period.
    """
    
    def __init__(self, initial_beta:float, final_beta:float, warm_up:int, start_warm_up:int):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.warm_up = warm_up
        self.start_warm_up = start_warm_up
        self.beta = initial_beta

    def step(self, epoch:int):

        if epoch < self.start_warm_up: # Keep beta constant before warm_up.
            self.beta = self.initial_beta
        elif epoch >= self.start_warm_up and epoch < self.start_warm_up + self.warm_up: # Linearly increase beta from initial_beta to final_beta.
            self.beta = self.initial_beta + (self.final_beta - self.initial_beta) * ( (epoch - self.start_warm_up) / self.warm_up )
        else:
            self.beta = self.final_beta  # Keep beta constant after warm_up.

        return self.beta



def train(T, dT, tmax, mean_U, mean_dU, train_loader, test_loader, max_epoch, enc, GP, dec, optimizer, 
          elbo, kl_sched, CL_factor, model_dir, device, dY_test, ds_vec, us_vec, min_epoch=0, stats=None):

    """
    Training loop.
    args:           
        T: "Continuous" time points for computing integrals in GP, shape = (K). 
        dT: Sampled time points to compute the prior mean and covariance matrix, shape = (N).
        tmax: Maximum time point.
        mean_U: "Continuous" mean function of U used in GP integrals, shape = (K,p).
        mean_dU: Discretised mean function of U corresponding to dT, shape = (N,p).
        train_loader: Dataloader for training set. Each batch has shape = (1, bs, N, m).
        test_loader: Dataloader for test set. Each batch has shape = (1, bs, N, m).
        max_epoch: Epoch which training will terminate at.
        enc: Encoder.
        GP: Gaussian Process.
        dec: Decoder.
        optimizer: Chosen optimizer.
        elbo: ELBO objective function.
        kl_sched: Annealing schedule for weighting the KL divergence term of ELBO loss.
        CL_factor: Factor to reduce the length of videos by.
        model_dir: Path to where models and data are stored.
        device: Hardware in use.
        dY_test: Ground truth latent states of test set (_, bs, N, m).
        ds_vec: Tensor containing down scaling terms for each of the m latent state dimensions shape = (m).
        us_vec: Tensor containing up scaling terms for each of the m latent state dimensions shape = (m).
        min_epoch: Epoch which training will start at.
        stats: Existing stats dataframe.
    returns:
        enc, GP, dec: Final version of the model.
    """

    if CL_factor > 1 or CL_factor < 0 :
        raise Exception("CL_factor must satisfy 0 =< CL_factor =< 1.")
    
    for epoch in range(min_epoch, max_epoch):
            
        # training loop
        enc.train()
        GP.train()
        dec.train()

        # train losses for each batch train_loader
        batch_elbo_train = []
        batch_recon_train = []
        batch_KL_train = []

        for batch, vid_batch in enumerate(train_loader):

            optimizer.zero_grad()

            vid_batch = vid_batch.squeeze(0).to(device).float() # shape = (bs, N, d, d)
            bs, N, d, _ = vid_batch.shape
            K  = mean_U.shape[0]
            
            # curriculum learning schedule
            N = int(CL_factor * N)
            K = int(CL_factor * K)
            tmax_cl = CL_factor * tmax

            vid_batch = vid_batch[:,:N]

            ##### model outputs #####
            mean_lhood, var_lhood = enc(vid_batch) # both (bs, N, m)
            mean_prior, covar_prior = GP(T[:K], dT[:N], tmax_cl, mean_U[:K], mean_dU[:N]) # (N, m) and (m*N, m*N) - same for all items in batch
            
            # map prior and likelihood to scaled space.
            mean_lhood_scaled = scale_mean(ds_vec, mean_lhood) # (bs, N, m) 
            mean_prior_scaled = scale_mean(ds_vec, mean_prior) # (N, m)
            covar_lhood_scaled = scale_covar(ds_vec, var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed() ) # (bs, mN, mN)
            covar_prior_scaled = scale_covar(ds_vec, covar_prior) # (mN, mN)
            
            # Compute posterior in scaled space.
            mean_post_scaled, covar_post_scaled = GP.posterior(covar_lhood_scaled, mean_lhood_scaled, mean_prior_scaled, covar_prior_scaled) # (bs, N, m), (bs, m*N, m*N)
            
            # Map posterior back to unscaled space.
            mean_post = scale_mean(us_vec, mean_post_scaled) # (bs, N, m)
            covar_post = scale_covar(us_vec, covar_post_scaled) # (bs, mN, mN)

            # draw samples from the unscaled space
            samples = GP.sample(mean_post, covar_post) # (bs, N, m, n_samples=3)
            samples = samples.mean(dim=3) # (bs, N, m)
            
            p_theta_logits = dec(samples).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d)

            ##### compute loss (reconstruction in unscaled space and KL in scaled space) #####
            kl_weight = kl_sched.step(epoch)
            elbo_train = - elbo( vid_batch, p_theta_logits, mean_prior_scaled, covar_prior_scaled, mean_post_scaled, covar_post_scaled, kl_weight) # weighted elbo
            recon_train = - elbo.recon_obj.item() # unweighted recon
            KL_train = - elbo.kl_obj.item() # unweighted KL

            ##### optimise #####
            elbo_train.backward()
            optimizer.step()

            batch_elbo_train.append( elbo_train.item() )
            batch_recon_train.append(recon_train)
            batch_KL_train.append(KL_train)

        # testing loop
        enc.eval()
        GP.eval()
        dec.eval()

        # test losses for each batch in test_loader
        batch_elbo_test = []
        batch_recon_test = []
        batch_KL_test = []

        with torch.no_grad():
          for batch, vid_batch in enumerate(test_loader): 

              vid_batch = vid_batch.squeeze(0).to(device).float() # shape = (bs, N, d, d)
              vid_batch = vid_batch[:,:N]
              
              ##### model outputs #####
              mean_lhood, var_lhood = enc(vid_batch) # both (bs, N, m)
              mean_prior, covar_prior = GP(T[:K], dT[:N], tmax_cl, mean_U[:K], mean_dU[:N]) # (N, m) and (m*N, m*N) - same for all items in batch

              # map prior and likelihood to scaled space.
              mean_lhood_scaled = scale_mean(ds_vec, mean_lhood) # (bs, N, m) 
              mean_prior_scaled = scale_mean(ds_vec, mean_prior) # (N, m)
              covar_lhood_scaled = scale_covar(ds_vec, var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed() ) # (bs, mN, mN)
              covar_prior_scaled = scale_covar(ds_vec, covar_prior) # (mN, mN)

              # Compute posterior in scaled space.
              mean_post_scaled, covar_post_scaled = GP.posterior(covar_lhood_scaled, mean_lhood_scaled, mean_prior_scaled, covar_prior_scaled) # (bs, N, m), (bs, m*N, m*N)

              # Map posterior back to unscaled space.
              mean_post = scale_mean(us_vec, mean_post_scaled) # (bs, N, m)
              covar_post = scale_covar(us_vec, covar_post_scaled) # (bs, mN, mN)

              samples = GP.sample(mean_post, covar_post) # (bs, N, m, n_samples=3)
              samples = samples.mean(dim=3) # (bs, N, m)
              p_theta_logits = dec(samples).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d)

              ##### compute loss (reconstruction in unscaled space and KL in scaled space) #####
              kl_weight = kl_sched.step(epoch)
              elbo_test = - elbo( vid_batch, p_theta_logits, mean_prior_scaled, covar_prior_scaled, mean_post_scaled, covar_post_scaled, kl_weight) # weighted elbo
              recon_test = - elbo.recon_obj.item() # unweighted recon
              KL_test = - elbo.kl_obj.item() # unweighted KL

              batch_elbo_test.append( elbo_test.item() )
              batch_recon_test.append(recon_test)
              batch_KL_test.append(KL_test)

        if epoch % 10 == 0 or epoch == max_epoch:
            
            # mean of Bernoulli for each pixel, averaged over n_samples.
            p_theta = torch.sigmoid(p_theta_logits) # shape = (bs, N, d, d)
            
            # # Project mean_post, covar_post from last test batch onto corresponding batch from Y_test.
            # proj_mean_post, W, MSE, proj_covar_post = MSE_projection(mean_post.cpu().numpy(), dY_test[batch,:,:N,:].cpu().numpy(), covar_post.cpu().numpy() )
            # torch.save(W, model_dir +  'W/epoch{:03d}.pt'.format(epoch) )
            
            nplots = 3
            string = 'latents_{:04d}'.format(epoch) # 'proj_latents_{:04d}'.format(epoch)
            plot_latents(model_dir + 'Data/Plots', string, vid_batch.unflatten(dim=0, sizes=(1,bs)), dY_test[batch,:,:N,:].unflatten(dim=0, sizes=(1,bs)), dT[:N], tmax_cl, nplots, recon_batch=p_theta, recon_traj=mean_post, recon_covar=covar_post)

        # save model, optimizer.
        checkpoint = {
                        'epoch': epoch,
                          'enc': enc.state_dict(),
                           'GP': GP.state_dict(),
                          'dec': dec.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
        torch.save(checkpoint, model_dir + 'Checkpoints/epoch{:04d}.pth'.format(epoch) )

        # save stats averaged over the batch.
        stats_path = model_dir + 'Data/stats.csv'
        if epoch == 0:
            stats = pd.DataFrame({
                'epoch':[epoch],
                'elbo train':[np.mean(batch_elbo_train)],
                'recon train':[np.mean(batch_recon_train)],
                'kl train':[np.mean(batch_KL_train)],
                'elbo test':[np.mean(batch_elbo_test)],
                'recon test':[np.mean(batch_recon_test)],
                'kl test':[np.mean(batch_KL_test)]
            })
            stats.to_csv(stats_path, index=False)
        else:
            # append to existing stats dataframe.
            stats_new = pd.DataFrame({
                'epoch':epoch,
                'elbo train':[np.mean(batch_elbo_train)],
                'recon train':[np.mean(batch_recon_train)],
                'kl train':[np.mean(batch_KL_train)],
                'elbo test':[np.mean(batch_elbo_test)],
                'recon test':[np.mean(batch_recon_test)],
                'kl test':[np.mean(batch_KL_test)]
            })
            stats = pd.concat([stats, stats_new], ignore_index=True)
            stats.to_csv(stats_path, index=False) # overwrites stats.


        if device.type == 'cuda':
          torch.cuda.empty_cache()

    return enc, GP, dec



def main():
    
    # Hardware settings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print("Device in use is: ", device)

    # Directory for storing checkpoints, plots, stats, etc.
    model_path = root + 'Models/'
    model_name = 'SEGP_VAE'
    exp_no = 1
    model_dir = model_path + model_name + '/Exp_{:03d}/'.format(exp_no)

    # Flag indicating if a checkpoint is being loaded in, followed by the epoch to load.
    Load = False
    Load_epoch = 0 # does nothing if Load is False.
    
    # If Load is True, these are the only extra parameters to be set. If False, more parameters below need to be set.
    max_epoch = 100
    CL_factor = 1.0 # curriculum learning factor for simplifying problem.

    # Directory where data is stored.
    if Load:
        print('Loading dataset settings!')
        train_settings = np.load(model_dir + 'Data/train_settings.pkl', allow_pickle=True)
        dataset_number = train_settings['dataset_number']
    else:
        dataset_number = 6
    
    data_path = root + 'Data/Dataset{0}'.format(dataset_number)

    if Load:
        print("Loading from model_dir!")
    elif not Load and os.path.isdir(model_dir):
        raise Exception("File already exists and not loading model! Not overwriting.")
    else:
        os.makedirs(model_dir)
        os.makedirs(model_dir + 'Checkpoints/')
        os.makedirs(model_dir + 'W/') 
        os.makedirs(model_dir + 'Data/Plots/')

    # Import data.
    data_setup = np.load(data_path + '/data_setup.pkl', allow_pickle=True)
    T = torch.from_numpy( np.load(data_path + '/T.npy') ).to(device)
    dT = torch.from_numpy( np.load(data_path + '/dT.npy') ).to(device)
    tmax = data_setup['tmax']
    mean_U = torch.from_numpy( np.load(data_path + '/mean_U.npy') ).unsqueeze(1).to(device)
    mean_dU = torch.from_numpy( np.load(data_path + '/mean_dU.npy') ).unsqueeze(1).to(device)
    dY = torch.from_numpy( np.load(data_path + '/dZ.npy') ).to(device)
    vid_batch = torch.from_numpy( np.load(data_path + '/vid_batch.npy') ).to(device)

    M, Q, N, d, _ = vid_batch.shape
    K = T.shape[0]
 
    # Scaling vectors.
    dY0_max = torch.max(dY[:,:,:,0])
    dY1_max = torch.max(dY[:,:,:,1])
    down_scaling_vec = torch.tensor([1/dY0_max, 1/dY1_max])
    up_scaling_vec = torch.tensor([dY0_max, dY1_max])

    # Setup data loaders.
    if Load:
        print('Loading dataloader settings!')
        batches = train_settings['batches']
        bs = train_settings['bs']
        test_split = train_settings['test_split']
        seed = train_settings['seed']
    else:
        batches = 100
        bs = 600
        test_split = 10/100
        seed = 42
    
    train_loader, test_loader, dY_test = get_dataloaders(vid_batch, dY, batches, bs, test_split, seed, device) 

    del vid_batch, dY # save memory.
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    
    # Instantiate model.
    if Load:
        print('Loading model settings!')
        model_settings = np.load(model_dir + 'Data/model_settings.pkl', allow_pickle=True)
        m = model_settings['m']
        n = model_settings['n']
        p = model_settings['p']
        lt = model_settings['lt']
        mean_x0 = model_settings['mean_x0']
        covar_x0 = model_settings['covar_x0']
        d = model_settings['d']
        h_dim = model_settings['h_dim']
        A = model_settings['A']
        B = model_settings['B']
        C = model_settings['C']
        D = model_settings['D']
    else:
        m = data_setup['m']
        n = data_setup['n']
        p = data_setup['p']
        lt = data_setup['lt']
        mean_x0 = torch.tensor([data_setup['mean_r'], data_setup['mean_theta']])
        covar_x0 = data_setup['sigma'] * torch.eye(n)
        d = data_setup['d']
        h_dim = 500
    
        l = data_setup['l']
        A = torch.tensor([[l, 0.0], [0.0, 0.0]])
        B = torch.tensor([[0.0], [1.0]])
        C = torch.eye(m)
        D = torch.zeros(m,p)


    GP = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A, B, C, D).to(device)
    enc = VAEEncoder(d*d, h_dim, m).to(device)
    dec = VAEDecoder(m, h_dim, d*d).to(device)


    # Instantiate optimizer, loss, KL annealing scheduler.
    if Load:
        lr = train_settings['lr'] # manually drop lr as needed.
        wd = train_settings['wd']
        initial_beta = train_settings['initial_beta']
        final_beta = train_settings['final_beta']
        warm_up = train_settings['warm_up']
        start_warm_up = train_settings['start_warm_up']
    else:
        lr = 9e-4 # manually drop lr as needed.
        wd = 1e-5 # weight decay.
        initial_beta = 0.7 # initial weight of KL divergence in ELBO loss.
        final_beta = 0.6 # final weight of KL divergence in ELBO loss.
        warm_up = 0 # number of epochs to linearly increase beta over.
        start_warm_up = 0 # epoch to start linearly increasing beta.

    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),lr=lr, weight_decay=wd)
    elbo = ELBO()
    kl_sched = KLAnnealingScheduler(initial_beta, final_beta, warm_up, start_warm_up)
     
    if Load:
        print('Loading in model, optimizer and previous stats!')
        checkpoint = torch.load(model_dir + 'Checkpoints/epoch{:04d}.pth'.format(Load_epoch), weights_only=True)
        min_epoch = checkpoint['epoch'] + 1
        enc.load_state_dict(checkpoint['enc'])
        GP.load_state_dict(checkpoint['GP'])
        dec.load_state_dict(checkpoint['dec'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        stats = pd.read_csv(model_dir + 'Data/stats.csv')
        if min_epoch >= max_epoch:
            raise Exception("min_epoch >= max_epoch.")
        
    # Save model and training settings. This will just overwrite if loading in a checkpoint.
    model_settings = {'m':m, 'n':n, 'p':p, 'lt':lt, 'mean_x0':mean_x0, 'covar_x0':covar_x0, 
                      'A':A, 'B':B, 'C':C, 'D':D, 'd':d, 'h_dim':h_dim}
    
    train_settings = {'dataset_number':dataset_number, 'batches':batches, 'bs':bs, 'test_split':test_split, 'seed':seed, 'lr':lr, 'wd':wd, 
            'max_epoch':max_epoch, 'initial_beta':initial_beta, 'final_beta':final_beta, 'warm_up':warm_up, 'start_warm_up':start_warm_up}

    with open(model_dir + 'Data/model_settings.pkl', 'wb') as f:
        pickle.dump(model_settings, f)
        f.close()
    with open(model_dir + 'Data/train_settings.pkl', 'wb') as f:
        pickle.dump(train_settings, f)
        f.close()

    enc_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    GP_params = sum(p.numel() for p in GP.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print('enc count =', enc_params) 
    print('GP count = ', GP_params)
    print('dec count =', dec_params)

    start_time = time.time()

    # Train model.
    if Load:
        enc, GP, dec = train(T, dT, tmax,  mean_U, mean_dU, train_loader, test_loader, max_epoch, enc, GP, dec, 
                             optimizer, elbo, kl_sched, CL_factor, model_dir, device, dY_test, down_scaling_vec, 
                             up_scaling_vec, min_epoch=min_epoch, stats=stats)
    else:
        enc, GP, dec = train(T, dT, tmax,  mean_U, mean_dU, train_loader, test_loader, max_epoch, enc, GP, dec, 
                             optimizer, elbo, kl_sched, CL_factor, model_dir, device, dY_test, down_scaling_vec, 
                             up_scaling_vec)


    exe_time = time.time() - start_time
    print( "--- %s seconds ---" % (exe_time) )

    return 0



if __name__ == '__main__':
    main()




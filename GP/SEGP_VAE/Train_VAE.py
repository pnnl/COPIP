import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from torch.utils.data import DataLoader
import numpy as np
import math
import sys
import os
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
from Utils import plot_latents, lr_scheduler


# Hardware settings.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Device in use is: ", device)



def get_dataloaders(vid_batch:torch.tensor, dY:torch.tensor,
                      batches:int, bs:int, test_split:float, idx=None, test_idx=None):
  """
  Function for splitting data into train and test dataloaders. Latent states corresponding to
  test loader data are also returned for plotting purposes.
  args:
         vid_batch: data to split with shape = (M, Q, N, d, d).
                dY: latent states to split in same way for plots (M, Q, N, m).
           batches: number of batches.
                bs: batch size.
        test_split: ratio of batches to reserve for testing.
               idx: indices for repeating a particular shuffle of vid_batch, dY.
          test_idx: indices for repeating a particular train/test split.
  returns:
      train_loader: dataloader with shape = (batches - N_test, bs, N, d, d).
       test_loader: dataloader with shape = (N_test, bs, N, d, d).
           dY_test: latent states corresponding to test loader; shape = (N_test, bs, N, m).
               idx: indices for repeating a particular shuffle of vid_batch, dY.
          test_idx: indices for repeating a particular train/test split.
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

  if idx == None:
      idx = torch.randperm(vid_batch.shape[0])

  # shuffle
  vid_batch = vid_batch[idx]
  dY = dY[idx]

  # split into train and test sets.
  N_test = int(batches * test_split)
  if N_test < 1:
      raise Exception("No batches reserved for testing!")

  if test_idx == None:
      test_idx = np.random.randint(low=0, high=batches, size=N_test)

  train_idx = np.delete( np.arange(0, batches), test_idx)

  test_loader = DataLoader(vid_batch[test_idx], batch_size=1, shuffle=False)
  train_loader = DataLoader(vid_batch[train_idx], batch_size=1, shuffle=False)
  dY_test = dY[test_idx]

  return train_loader, test_loader, dY_test, idx, test_idx



class ELBO(nn.Module):
    """
    Class for computing the two terms of the ELBO objective: reconstruction term and KL divergence, 
    each averaged across a batch of data with shape (bs, N, m).
    args:
          weights: 2d tensor containing weights for each term of the ELBO objective.
    """

    def __init__(self, weights:torch.tensor):
        super().__init__()
        self.weights = weights
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.register_buffer('pi', torch.tensor(math.pi) )



    def recon(self, vid_batch:torch.tensor, p_theta_logits:torch.tensor):
        """
        Computes the average reconstruction term across a batch of data.
        args:
                 vid_batch: Observed images (bs, N, d, d).
            p_theta_logits: Logits of Bernouli distribution for each black/white pixel of image (bs, N, d, d, n_samples).
        returns:
                  recon_av: average reconstruction term (scalar).
        """

        bs, N, d, _, n_samples = p_theta_logits.shape

        recon = -self.BCE(p_theta_logits,
                         vid_batch.unsqueeze(4).repeat(1,1,1,1,n_samples)) # shape = (bs, N, d, d, n_samples)
        recon = recon.mean(dim=4) # shape = (bs, N, d, d)
        recon = recon.sum( dim=(3,2,1) ) # shape = (bs)

        recon_av = recon.mean()

        return recon_av



    def kl_divergence(self, mean_post:torch.tensor, covar_post:torch.tensor, 
                      mean_prior:torch.tensor, covar_prior:torch.tensor):
        """
        Compute the average KL divergence (across a batch) between two multivariate normal distributions
        q(z) = N(mean_post, covar_post) and p(z) = N(mean_prior, covar_prior) using the Cholesky decomposition.

        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Introduction_and_context
        
        Args:
        - mean_post: Mean of the variational posterior distribution q(z) with shape (bs, N, m).
        - covar_post: Covariance matrix of the variational posterior distribution q(z) with shape (bs, m*N, m*N).
        - mean_prior: Mean of the prior distribution p(z) with shape (N, m).
        - covar_prior: Covariance matrix of the prior distribution p(z) with shape (m*N, m*N).
        
        Returns:
        - kl_div_av (scalar).
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
                covar_prior:torch.tensor, mean_post:torch.tensor, covar_post:torch.tensor):
        """
        Computes ELBO objective.
        args:
                  vid_batch: Observed images (bs, N, d, d).
             p_theta_logits: Logits of Bernouli distribution for each black/white pixel of image (bs, N, d, d, n_samples).
                 mean_prior: SEGP mean function (N, m).
                covar_prior: SEGP covariance matrix (m*N, m*N).
                  mean_post: SEGP posterior mean function (bs, N, m).
                 covar_post: SEGP posterior covariance matrix (bs, m*N, m*N).
        returns:
                   elbo_obj: ELBO objective (scalar).
        """

        self.recon_obj = self.weights[0] * self.recon(vid_batch, p_theta_logits) # should always be negative.
        self.kl_obj = - self.weights[1] * self.kl_divergence(mean_post, covar_post, mean_prior, covar_prior) # should always be negative.

        self.elbo_obj = self.recon_obj + self.kl_obj

        return self.elbo_obj



def train(dT:torch.tensor, tmax, mean_U, mean_dU, train_loader, test_loader,
          max_epoch:int, enc, GP, dec, optimizer, elbo, decay:float, decay_epochs:list,
          model_dir:str, device, Y_test:torch.tensor, min_epoch=0,
          mean_train_elbo_losses=[], mean_train_recon_losses=[],mean_train_KL_losses=[],
          mean_test_elbo_losses=[], mean_test_recon_losses=[],mean_test_KL_losses=[]):

    """
    Training loop.
    args:
                    dT: Sampled time points to compute the prior mean and covariance matrix.
                  tmax: Maximum time point.
                mean_U: "Continuous" mean function of U used in GP integrals.
               mean_dU: Discretised mean function of U corresponding to dT.
          train_loader: Dataloader for training set. Each batch has shape = (1, bs, N, m).
           test_loader: Dataloader for test set. Each batch has shape = (1, bs, N, m).
             max_epoch: Epoch which training will terminate at.
                   enc: Encoder.
                    GP: Gaussian Process.
                   dec: Decoder.
             optimizer: Chosen optimizer.
                  elbo: ELBO objective function.
                 decay: Scalar to multiply lr by.
          decay_epochs: List containing the epochs which the lr should be cut at.
             model_dir: Path to where models and data are stored.
                device: Hardware in use.
                Y_test: Ground truth latent states of test set (bs, N, m).
    optional:
              min_epoch: Epoch which training will start at.
 mean_train_elbo_losses: List containing elbo training loss at each epoch from disrupted run.
mean_train_recon_losses: List containing recon training loss at each epoch from disrupted run.
   mean_train_KL_losses: List containing KL training loss at each epoch from disrupted run.
  mean_test_elbo_losses: List containing elbo test loss at each epoch from disrupted run.
 mean_test_recon_losses: List containing recon test loss at each epoch from disrupted run.
    mean_test_KL_losses: List containing KL test loss at each epoch from disrupted run.

    returns:
           enc, GP, dec: Final version of the model.
    """

    if len(mean_train_elbo_losses) == 0:
        mean_train_elbo_losses = []
        mean_train_recon_losses = []
        mean_train_kl_losses = []

    if len(mean_test_elbo_losses) == 0:
        mean_test_elbo_losses = []
        mean_test_recon_losses = []
        mean_test_KL_losses = []

    for epoch in range(min_epoch, max_epoch):

        # training loop
        enc.train()
        GP.train()
        dec.train()

        batch_elbo_losses = []
        batch_recon_losses = []
        batch_KL_losses = []

        for batch, vid_batch in enumerate(train_loader):

            optimizer.zero_grad()
            vid_batch = vid_batch.squeeze(0).to(device).float() # shape = (bs, N, d, d)
            bs, N, d, _ = vid_batch.shape

            ##### model outputs #####
            mean_lhood, var_lhood = enc(vid_batch) # both (bs, N, m)
            mean_prior, covar_prior = GP(dT, tmax, mean_U, mean_dU) # (N, m) and (m*N, m*N) - same for all items in batch
            
            # var_lhood is converted to diagonal covariance matrix with shape = (bs, m*N, m*N)
            # and outputs have shape = (bs, N, m) & (bs, m*N, m*N)
            mean_post, covar_post = GP.posterior(dT, dT, tmax, tmax, mean_U, mean_U, mean_dU, mean_dU,
                                                 var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed(), mean_lhood)

            samples = GP.sample_posterior() # (bs, N, m, n_samples=3)
            p_theta_logits = dec( samples.transpose(2,3) ).transpose(2,3).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d, n_samples)

            ##### compute loss #####
            loss_elbo = - elbo( vid_batch, p_theta_logits, mean_prior, covar_prior, mean_post, covar_post)

            loss_recon = - elbo.recon_obj.item()

            loss_KL = - elbo.kl_obj.item()


            ##### optimise #####
            loss_elbo.backward()
            optimizer.step()

            batch_elbo_losses.append( loss_elbo.item() )
            batch_recon_losses.append(loss_recon)
            batch_KL_losses.append(loss_KL)

        mean_train_elbo_losses.append( np.mean(batch_elbo_losses) )
        mean_train_recon_losses.append( np.mean(batch_recon_losses) )
        mean_train_KL_losses.append( np.mean(batch_KL_losses) )

        optimizer = lr_scheduler(epoch, optimizer, decay, decay_epochs)

        # testing loop
        enc.eval()
        GP.eval()
        dec.eval()

        batch_elbo_losses = []
        batch_recon_losses = []
        batch_KL_losses = []

        with torch.no_grad():
          for batch, vid_batch in enumerate(test_loader):

              vid_batch = vid_batch.squeeze(0).to(device).float() # shape = (bs, N, d, d)

              ##### model outputs #####
              mean_lhood, var_lhood = enc(vid_batch) # both (bs, N, m)
              mean_prior, covar_prior = GP(dT, tmax, mean_U, mean_dU) # (N, m) and (m*N, m*N) - same for all items in batch

              # var_lhood is converted to diagonal covariance matrix with shape = (bs, m*N, m*N)
              # and outputs have shape = (bs, N, m) & (bs, m*N, m*N)
              mean_post, covar_post = GP.posterior(dT, dT, tmax, tmax, mean_U, mean_U, mean_dU, mean_dU,
                                                   var_lhood.transpose(1, 2).flatten(start_dim=1, end_dim=2).diag_embed(), mean_lhood)

              samples = GP.sample_posterior() # (bs, N, m, n_samples=3)
              p_theta_logits = dec( samples.transpose(2,3) ).transpose(2,3).unflatten(dim=2, sizes=(d,d)) # (bs, N, d, d, n_samples)

              ##### compute loss #####
              loss_elbo = - elbo( vid_batch, p_theta_logits, mean_prior, covar_prior, mean_post, covar_post)

              loss_recon = - elbo.recon_obj.item()

              loss_KL = - elbo.kl_obj.item()

              batch_elbo_losses.append( loss_elbo.item() )
              batch_recon_losses.append(loss_recon)
              batch_KL_losses.append(loss_KL)

        mean_test_elbo_losses.append( np.mean(batch_elbo_losses) )
        mean_test_recon_losses.append( np.mean(batch_recon_losses) )
        mean_test_KL_losses.append( np.mean(batch_KL_losses) )

        print('Iter %d/%d - Train ELBO Loss: %.3f - Recon Loss: %.3f - KL Loss: %.3f'
          % (epoch + 1, max_epoch, mean_train_elbo_losses[epoch],
             mean_train_recon_losses[epoch], mean_train_KL_losses[epoch]) )

        print('Iter %d/%d - Test ELBO Loss: %.3f - Recon Loss: %.3f - KL Loss: %.3f'
          % (epoch + 1, max_epoch, mean_test_elbo_losses[epoch],
             mean_test_recon_losses[epoch], mean_test_KL_losses[epoch]) )


        if epoch % 10 == 0:
            # mean of Bernoulli for each pixel, averaged over n_samples.
            p_theta = torch.sigmoid(p_theta_logits).mean(dim=4) # shape = (bs, N, d, d)

            nplots = 3
            fig = plt.gcf()
            string = 'latents_{0}'.format(epoch+1)
            temp = int(bs/2)
            plot_latents(model_dir + 'Plots', string, vid_batch.unflatten(dim=0, sizes=(2,temp)), Y_test.unflatten(dim=0, sizes=(2,temp)), dT, tmax, nplots, recon_batch=p_theta, recon_traj=mean_post, recon_covar=covar_post)
            fig.tight_layout()
            plt.show()


        # save models
        enc_path = model_dir + 'Encoder/epoch{:03d}.pt'.format(epoch+1)
        GP_path = model_dir + 'GP/epoch{:03d}.pt'.format(epoch+1)
        dec_path = model_dir + 'Decoder/epoch{:03d}.pt'.format(epoch+1)

        try:
              torch.save(enc.state_dict(), enc_path)
              torch.save(GP.state_dict(), GP_path)
              torch.save(dec.state_dict(), dec_path)
        except:
              os.makedirs(model_dir + 'Encoder/')
              os.makedirs(model_dir + 'GP/')
              os.makedirs(model_dir + 'Decoder/')
              torch.save(enc.state_dict(), enc_path)
              torch.save(GP.state_dict(), GP_path)
              torch.save(dec.state_dict(), dec_path)

        # save stats.
        stats_path = model_dir + 'stats.csv'
        stats = pd.DataFrame()
        stats["train elbo loss"] = mean_train_elbo_losses
        stats["train recon loss"] = mean_train_recon_losses
        stats["train KL loss"] = mean_train_KL_losses
        stats["test elbo loss"] = mean_test_elbo_losses
        stats["test recon loss"] = mean_test_recon_losses
        stats["test KL loss"] = mean_test_KL_losses
        stats.to_csv(stats_path, index=False)

        if device.type == 'cuda':
          torch.cuda.empty_cache()

    return enc, GP, dec



def main():

    # Directory where data is stored.
    dataset_number = 1
    data_path = root + 'Data/Dataset{0}'.format(dataset_number)

    # Directory for storing models.
    model_path = root + 'Models/'
    model_name = 'SEGP_VAE'
    exp_no = 1
    model_dir = model_path + model_name + '/Exp_{:03d}/'.format(exp_no)

    if os.path.isdir(model_dir):
        raise Exception("File already exists! Not overwriting.")
    else:
        os.makedirs(model_dir)
        os.makedirs(model_dir + 'Plots/')

    # import data.
    data_setup = np.load(data_path + '/data_setup.pkl', allow_pickle=True)
    dT = torch.from_numpy( np.load(data_path + '/dT.npy') ).to(device)
    mean_U = torch.from_numpy( np.load(data_path + '/mean_U.npy') ).unsqueeze(1).to(device)
    mean_dU = torch.from_numpy( np.load(data_path + '/mean_dU.npy') ).unsqueeze(1).to(device)
    dZ = torch.from_numpy( np.load(data_path + '/dZ.npy') ).to(device)
    vid_batch = torch.from_numpy( np.load(data_path + '/vid_batch.npy') ).to(device)

    M, Q, N, d, _ = vid_batch.shape

    print('M = {0} \t Q = {1} \t N = {2} \t d = {3}'.format(M, Q, N, d))

    # setup data loaders.
    batches = 20
    bs = 320
    test_split = 1/20
    train_loader, test_loader, Z_test, idx, test_idx = get_dataloaders(vid_batch, dZ, batches, bs, test_split)  

    with open(model_dir + 'idx.npy', 'wb') as f:
      np.save(f, idx.cpu())

    with open(model_dir + 'test_idx.npy', 'wb') as f:
        np.save(f, test_idx)

    with open(model_dir + 'Z_test.npy', 'wb') as f:
        np.save(f, Z_test.cpu().numpy() )

    del vid_batch, idx, test_idx
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    
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
    C_train = torch.eye(m) # torch.tensor([[1, 0.0], [0.0, 1/scale]])
    D_train = torch.zeros(m,p)

    GP = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A_train, B_train, C_train, D_train).to(device)
    enc = VAEEncoder(d*d, h_dim, m).to(device)
    dec = VAEDecoder(m, h_dim, d*d).to(device)

    model_settings = {'m':m, 'n':n, 'p':p, 'lt':lt, 'mean_x0':mean_x0, 'covar_x0':covar_x0,
                  'A_train':A_train, 'B_train':B_train, 'C_train':C_train, 'D_train':D_train,
                    'd':d, 'h_dim':h_dim}


    # Instantiate optimiser.
    lr = 1e-3
    wd = 1e-5 # weight decay
    decay = 1e-1 # scalar to multiply lr by at decay_epochs
    decay_epochs = [] # epochs to perform lr cut
    
    # optimizer = torch.optim.Adam(list(enc.parameters()) + list(GP.parameters()) + list(dec.parameters()),
    #                             lr=lr, weight_decay=wd)
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=wd)

    # Training settings.
    min_epoch = 0
    max_epoch = 10000

    # Instantiate loss - weighted ELBO.
    elbo_weights = torch.tensor([1.0, 1.0]) # weight of (recon, KL) terms.
    elbo = ELBO(elbo_weights)

    train_settings = {'lr':lr, 'wd':wd, 'decay':decay, 'decay_epochs':decay_epochs, 'min_epoch':min_epoch, 
                      'max_epoch':max_epoch, 'elbo_weights':elbo_weights}

    with open(model_dir + 'model_settings.pkl', 'wb') as f:
        pickle.dump(model_settings, f)
        f.close()
    with open(model_dir + 'train_settings.pkl', 'wb') as f:
        pickle.dump(train_settings, f)
        f.close()

    enc_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    GP_params = sum(p.numel() for p in GP.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print( 'parameter count =',  enc_params + GP_params + dec_params)

    start_time = time.time()
    print( "--- %s seconds ---" % (start_time) )

    # Train model.
    enc, GP, dec = train(dT, tmax,  mean_U, mean_dU, train_loader, test_loader, max_epoch,
                              enc, GP, dec, optimizer, elbo, decay, decay_epochs, model_dir,
                                device, Z_test[0])


    exe_time = time.time() - start_time
    print( "--- %s seconds ---" % (exe_time) )


    train_settings = {'lr':lr, 'wd':wd, 'decay':decay, 'decay_epochs':decay_epochs, 'min_epoch':min_epoch, 
                      'max_epoch':max_epoch, 'elbo_weights':elbo_weights, 'exe_time':exe_time}

    with open(model_dir + 'train_settings.pkl', 'wb') as f:
        pickle.dump(train_settings, f)
        f.close()

    return 0



if __name__ == '__main__':
    main()

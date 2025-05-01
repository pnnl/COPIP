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
GP_path = root + 'Code/SEGP'

# Directory utility functions are stored.
utils_path = root + 'Code/Utils'

# Add directories to path.
if GP_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(GP_path)

if utils_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(utils_path)

# Import custom files.
import SEGP
from Utils import scale_mean, scale_covar



def get_dataloaders(dY:torch.tensor, batches:int, bs:int, test_split:float, seed:int, device):
  """
  Function for splitting data into train and test dataloaders.
  args:
                  dY: latent states to split into train and test sets (M, Q, N, m).
            batches: number of batches.
                  bs: batch size.
          test_split: ratio of batches to reserve for testing.
                seed: random seed.
              device: hardware device.

  returns:
        train_loader: dataloader with shape = (batches - N_test, bs, N, m).
        test_loader: dataloader with shape = (N_test, bs, N, m).
  """

  M, Q, N, m = dY.shape

  dY = dY.view(M*Q, N, m) # shape = (M * Q, N, m)

  # reshape into shape = (batches, bs, N, m)
  if batches * bs != M * Q:
      print('M =', M)
      print('Q =', Q)
      raise Exception("batches * bs != M * Q!")
  else:
      dY = dY.view(batches, bs, N, m)

  # passing in the same seed will result in the same idx.
  generator = torch.Generator(device=device)
  generator.manual_seed(seed)
  idx = torch.randperm(dY.shape[0], generator=generator)

  # shuffle
  dY = dY[idx]

  # split into train and test sets.
  N_test = int(batches * test_split)
  if N_test < 1:
      raise Exception("No batches reserved for testing!")

  # passing in the same seed will result in the same test_idx.
  rng = np.random.default_rng(seed)
  test_idx = rng.integers(low=0, high=batches, size=N_test)

  train_idx = np.delete( np.arange(0, batches), test_idx)

  test_loader = DataLoader(dY[test_idx], batch_size=1, shuffle=False)
  train_loader = DataLoader(dY[train_idx], batch_size=1, shuffle=False)

  return train_loader, test_loader



class MLL(nn.Module):
    """
    Class for computing the log marginal likelihood (mll) for a batch of data with shape (bs, N, m).
    """

    def __init__(self):
        super().__init__()
        self.pi = torch.tensor(math.pi)



    def forward(self, obs:torch.tensor, mean:torch.tensor, covar:torch.tensor,
                covar_noise:torch.tensor):
        """
        Computes mll objective.
        args:
                  obs: Observed data (bs, N, m).
                 mean: SEGP mean function (N, m).
                covar: SEGP covariance matrix (m*N, m*N).
          covar_noise: Covariance matrix for the measurement noise (m*N, m*N).
        returns:
               mll_av: log marginal likelihood averaged over the batch.
        """

        bs, N, m = obs.shape

        total_covar = covar + covar_noise # shape = (m*N, m*N)  
        L = torch.linalg.cholesky(total_covar, upper=False) # shape = (m*N, m*N)

        # Reshape obs and mean.
        obs = torch.transpose(obs, 1, 2).reshape(bs, m*N).unsqueeze(2) # shape = (bs, m*N, 1)
        mean = torch.transpose( mean.repeat(bs, 1, 1), 1, 2).reshape(bs, m*N).unsqueeze(2) # shape = (bs, m*N, 1)
        
        # Solve L * alpha = obs - mean for alpha.
        alpha = torch.cholesky_solve(obs - mean, L.repeat(bs, 1, 1) ) # shape = (bs , m*N, 1)

        # Compute quadratic term.
        term1 = -0.5 * torch.bmm((obs-mean).transpose(1,2), alpha) # shape = (bs, 1, 1)
        term1 = term1.squeeze(2).squeeze(1) # shape = (bs)

        # Compute determinant term.
        term2 = - torch.sum(torch.log(torch.diagonal(L) ) )
        term2 = term2.repeat(bs,1).squeeze(1) # shape = (bs)

        # Compute constant term.
        term3 = -0.5 * N * m * torch.log(2*self.pi)
        term3 = term3.repeat(bs,1).squeeze(1) # shape = (bs)

        # check for infs.
        if torch.any( torch.isinf(term1) ):
            print('term1 contains infs!')

        if torch.any( torch.isinf(term2) ):
            print('term2 contains infs!')

        if torch.any( torch.isinf(term3) ):
            print('term3 contains infs!')
            
        mll = term1 + term2 + term3
        mll_av = mll.sum() / bs

        return mll_av



def train(T:torch.tensor, dT:torch.tensor, tmax:float, mean_U:torch.tensor, mean_dU:torch.tensor, covar_noise:torch.tensor,
          train_loader, test_loader, max_epoch:int, model, optimizer, mll, model_dir:str, device, ds_vec, us_vec):
    """
    Training loop.
    args:           
                     T: "Continuous" time points for computing integrals in GP, shape = (K). 
                    dT: Sampled time points to compute the prior mean and covariance matrix, shape = (N).
                  tmax: Maximum time point.
                mean_U: "Continuous" mean function of U used in GP integrals, shape = (K,p).
               mean_dU: Discretised mean function of U corresponding to dT, shape = (N,p).
           covar_noise: Covariance of noise, shape = (mN, mN).
          train_loader: Dataloader for training set. Each batch has shape = (1, bs, N, m).
           test_loader: Dataloader for test set. Each batch has shape = (1, bs, N, m).
             max_epoch: Epoch which training will terminate at.
                 model: Gaussian Process.
             optimizer: Chosen optimizer.
                   mll: MLL objective function.
             model_dir: Path to where models and data are stored.
                device: Hardware in use.
                ds_vec: Tensor containing down scaling terms for each of the m latent state dimensions shape = (m).
                us_vec: Tensor containing up scaling terms for each of the m latent state dimensions shape = (m).
    returns:
                 model: Final version of the GP.
    """

    covar_noise_scaled = scale_covar(ds_vec, covar_noise) # (mN, mN)

    for epoch in range(max_epoch):
      
      print('epoch {}'.format(epoch) )

      # training loop
      model.train()
      batch_train = []

      for batch, y in enumerate(train_loader):
          
          y = y.to(device)    

          optimizer.zero_grad()

          mean, covar = model(T, dT, tmax, mean_U, mean_dU)

          # scale for training.
          mean_scaled = scale_mean(ds_vec, mean) # (N, m)
          covar_scaled = scale_covar(ds_vec, covar) # (mN, mN)
          y_scaled = scale_mean(ds_vec, y[0]) # (N, m)

          loss = -mll(y_scaled, mean_scaled, covar_scaled, covar_noise_scaled)

          loss.backward()
          optimizer.step()
          
          batch_train.append(loss.item())


      # testing loop
      model.eval()
      batch_test = []

        
      with torch.no_grad():
          for batch, y in enumerate(test_loader): 

              y = y.to(device)
              
              mean, covar = model(T, dT, tmax, mean_U, mean_dU)
          
              # scale for training.
              mean_scaled = scale_mean(ds_vec, mean) # (N, m)
              covar_scaled = scale_covar(ds_vec, covar) # (mN, mN)
              y_scaled = scale_mean(ds_vec, y[0]) # (N, m)

              loss = -mll(y_scaled, mean_scaled, covar_scaled, covar_noise_scaled)
 
              batch_test.append(loss.item())

      
      print('Saving model and stats!')

      # save model, optimizer.
      checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
      
      torch.save(checkpoint, model_dir + 'Checkpoints/epoch{:04d}.pth'.format(epoch) )
      
      # save stats averaged over the batch.
      stats_path = model_dir + 'Data/stats.csv'
      if epoch == 0:
          stats = pd.DataFrame({
                'epoch':[epoch],
                'mll train':[np.mean(batch_train)],
                'mll test':[np.mean(batch_test)]
                 })
          stats.to_csv(stats_path, index=False)
      else:
          # append to existing stats dataframe.
          stats_new = pd.DataFrame({
              'epoch':epoch,
                'mll train':[np.mean(batch_train)],
                'mll test':[np.mean(batch_test)]
            })
          stats = pd.concat([stats, stats_new], ignore_index=True)
          stats.to_csv(stats_path, index=False) # overwrites stats.


      if device.type == 'cuda':
          torch.cuda.empty_cache()

    return model



def main():

  # Hardware settings.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.set_default_device(device)
  print("Device in use is: ", device)

  # Directory where data is stored.
  dataset_number = 4 
  data_path = root + 'Data/Dataset{0}'.format(dataset_number)

  # Directory for storing checkpoints, plots, stats, etc.
  model_path = root + 'Models/'
  model_name = 'SEGP'
  exp_no = 5
  model_dir = model_path + model_name + '/Exp_{:03d}/'.format(exp_no)

  if os.path.isdir(model_dir):
      raise Exception("File already exists! Not overwriting.")
  else:
      os.makedirs(model_dir)
      os.makedirs(model_dir + 'Checkpoints/')
      os.makedirs(model_dir + 'Data/Plots/')

  # Import data.
  data_setup = np.load(data_path + '/data_setup.pkl', allow_pickle=True)
  T = torch.from_numpy( np.load(data_path + '/T.npy') ).to(device)
  dT = torch.from_numpy( np.load(data_path + '/dT.npy') ).to(device)
  tmax = data_setup['tmax']
  mean_U = torch.from_numpy( np.load(data_path + '/mean_U.npy') ).unsqueeze(1).to(device)
  mean_dU = torch.from_numpy( np.load(data_path + '/mean_dU.npy') ).unsqueeze(1).to(device)
  dY = torch.from_numpy( np.load(data_path + '/dZ.npy') ).to(device)
  dYn = torch.from_numpy( np.load(data_path + '/dZn.npy') ).to(device)
  
  M, Q, N, m = dYn.shape

  print('M = {0} \t Q = {1} \t N = {2} \t m = {3}'.format(M, Q, N, m))

  # Scaling vectors.
  dY0_max = torch.max(dYn[:,:,:,0])
  dY1_max = torch.max(dYn[:,:,:,1])
  down_scaling_vec = torch.tensor([1/dY0_max, 1/dY1_max])
  up_scaling_vec = torch.tensor([dY0_max, dY1_max])
  
  # setup  data loaders.
  batches = 10
  bs = 1000
  test_split = 1/10
  seed = 42
  train_loader, test_loader = get_dataloaders(dYn, batches, bs, test_split, seed, device)

  # Instantiate model.
  m = data_setup['m']
  n = data_setup['n']
  p = data_setup['p']
  lt = data_setup['lt']
  mean_x0 = torch.tensor([data_setup['mean_r'], data_setup['mean_theta']])
  covar_x0 = data_setup['sigma'] * torch.eye(n)
  covar_noise = data_setup['var_noise'] * torch.eye(m*N)
  covar_noise = covar_noise.to(device)

  # l = data_setup['l']
  A = None # torch.tensor([[-l, 0.0], [0.0, 0.0]])
  B = None # torch.tensor([[0.0], [1.0]])
  C = None # torch.eye(m)
  D = torch.zeros(m,p)

  model = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A, B, C, D).to(device)

  # Instantiate optimiser, loss.
  lr = 1e-1
  wd = 1e-5 # weight decay
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  mll = MLL()

  max_epoch = 50

  # Save model and training settings. This will just overwrite if loading in a checkpoint.
  model_settings = {'m':m, 'n':n, 'p':p, 'lt':lt, 'mean_x0':mean_x0, 'covar_x0':covar_x0, 
                      'A':A, 'B':B, 'C':C, 'D':D}
    
  train_settings = {'dataset_number':dataset_number, 'batches':batches, 'bs':bs, 'test_split':test_split,
            'seed':seed, 'lr':lr, 'wd':wd, 'max_epoch':max_epoch}

  with open(model_dir + 'Data/model_settings.pkl', 'wb') as f:
    pickle.dump(model_settings, f)
    f.close()
  with open(model_dir + 'Data/train_settings.pkl', 'wb') as f:
    pickle.dump(train_settings, f)
    f.close()

  # Train model.
  print( 'parameter count =', sum(p.numel() for p in model.parameters() if p.requires_grad) )
  start_time = time.time()
    
  model = train(T, dT, tmax, mean_U, mean_dU, covar_noise, train_loader, test_loader, max_epoch, model, 
                optimizer, mll, model_dir, device, down_scaling_vec, up_scaling_vec)
  
  exe_time = time.time() - start_time
  print( "--- %s seconds ---" % (exe_time) )

  return 0



if __name__ == '__main__':
    main()

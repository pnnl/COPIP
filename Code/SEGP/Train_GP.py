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
from Utils import lr_scheduler, scale_data

# Hardware settings.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Device in use is: ", device)



def get_dataloaders(data:torch.tensor, M:int, Q:int, N:int, m:int, batches:int,
                    bs:int, test_split:float):
  """
  Function for splitting data into train and test dataloaders.
  args:
              data: data to split with shape = (M, Q, N, m).
                 M: number of input signals.
                 Q: number of initial conditions.
                 N: number of discrete time points.
                 m: number of latent states.
           batches: number of batches.
                bs: batch size.
        test_split: ratio of batches to reserve for testing.
  returns:
      train_loader: dataloader with shape = (batches - N_test, bs, N, m).
       test_loader: dataloader with shape = (N_test, bs, N, m).
  """

  data = data.view(M*Q, N, m) # shape = (M * Q, N, m)

  # shuffle
  idx = torch.randperm(data.shape[0]).cpu()
  data = data[idx]

  # reshape into shape = (batches, bs, N, m)
  if batches * bs != M * Q:
    print('M =', M)
    print('Q =', Q)
    raise Exception("batches * bs != M * Q!")
  else:
    data = data.view(batches, bs, N, m)


  # split into train and test sets.
  N_test = int(batches * test_split)
  if N_test < 1:
    raise Exception("No batches reserved for testing!")
  else:
    test_idx = np.random.randint(low=0, high=batches, size=N_test)
    train_idx = np.delete( np.arange(0, batches), test_idx)
    test_set = data[test_idx]
    train_set = data[train_idx]

  test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
  train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

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



def train(dT:torch.tensor, tmax:float, mean_U:torch.tensor, mean_dU:torch.tensor, covar_noise:torch.tensor,
          train_loader, test_loader, max_epoch:int, model, optimizer, mll, decay:float, decay_epochs:list,
          model_dir:str, device, min_epoch=0, mean_train_losses=[], mean_test_losses=[]):
    """
    Training loop.
    args:
                    dT: Sampled time points to compute the prior mean and covariance matrix.
                  tmax: Maximum time point.
                mean_U: "Continuous" mean function of U used in GP integrals.
               mean_dU: Discretised mean function of U corresponding to dT.
           covar_noise: Covariance matrix for the measurement noise (m*N, m*N).
          train_loader: Dataloader for training set. Each batch has shape = (1, bs, N, m).
           test_loader: Dataloader for test set. Each batch has shape = (1, bs, N, m).
             max_epoch: Epoch which training will terminate at.
                 model: Instantiation of the chosen model.
             optimizer: Chosen optimizer.
                   mll: log marginal likelihood objective function.
                 decay: Scalar to multiply lr by.
          decay_epochs: List containing the epochs which the lr should be cut at.
             model_dir: Path to where models and data are stored.
                device: Hardware in use.
    optional:
             min_epoch: Epoch which training will start at.
     mean_train_losses: List containing training loss at each epoch from disrupted run.
      mean_test_losses: List containing test loss at each epoch from disrupted run.

    returns:
                model: Final version of the model.
    mean_train_losses: List containing updated training losses.
     mean_test_losses: List containing updated test losses.
    """

    if len(mean_train_losses) == 0:
        mean_train_losses = []

    if len(mean_test_losses) == 0:
        mean_test_losses = []

    for epoch in range(min_epoch, max_epoch):

        # training loop
        model.train()
        batch_losses = []

        for batch, y in enumerate(train_loader):

            y = y.to(device)
            optimizer.zero_grad()
            mean, covar = model(dT, tmax, mean_U, mean_dU)
            loss = -mll(y[0], mean, covar, covar_noise)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            print('batch {} processed!'.format(batch) )

        mean_train_losses.append( np.mean(batch_losses) )

        optimizer = lr_scheduler(epoch, optimizer, decay, decay_epochs)

        # testing loop
        model.eval()
        batch_losses = []

        print('Start testing loop!')
        
        with torch.no_grad():
          for batch, y in enumerate(test_loader):
              y = y.to(device)
              mean, covar = model(dT, tmax, mean_U, mean_dU)
              loss = -mll(y[0], mean, covar, covar_noise)
              batch_losses.append(loss.item())

          mean_test_losses.append( np.mean(batch_losses) )

        print('Iter %d/%d - Train Loss: %.3f - Test Loss: %.3f' % (epoch + 1, max_epoch, mean_train_losses[epoch], mean_test_losses[epoch]))


        # save models
        model_path = model_dir + 'epoch{:03d}.pt'.format(epoch+1)
        torch.save(model.state_dict(), model_path)

        # save stats.
        stats_path = model_dir + 'stats.csv'
        stats = pd.DataFrame()
        stats["train loss"] = mean_train_losses
        stats["test loss"] = mean_test_losses
        stats.to_csv(stats_path, index=False)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model, mean_train_losses, mean_test_losses



def main():

    # Directory where data is stored.
    dataset_number = 1 
    data_path = root + 'Data/Dataset{0}'.format(dataset_number)

    # Directory for storing models.
    model_path = root + 'Models/'

    # Training loop.

    model_name = 'SEGP'
    exp_no = 2
    model_dir = model_path + model_name + '/Exp_{:03d}/'.format(exp_no)

    if os.path.isdir(model_dir):
        raise Exception("File already exists! Not overwriting.")
    else:
        os.makedirs(model_dir)

    # import data
    data_setup = np.load(data_path + '/data_setup.pkl', allow_pickle=True)
    dZn = torch.from_numpy( np.load(data_path + '/dZn.npy') )
    M, Q, N, n = dZn.shape

    print('M = {0} \t Q = {1} \t N = {2} \t n = {3}'.format(M, Q, N, n))
    
    # scale theta by scale.
    scale = 5.5
    dim = 1
    dZn = scale_data(dZn, scale, dim)

    # setup  data loaders.
    batches = 20
    bs = 320
    test_split = 1/20
    train_loader, test_loader = get_dataloaders(dZn, M, Q, N, n, batches, bs, test_split)


    # Import known parameters of data generation process.
    m = data_setup['m']
    p = data_setup['p']
    lt = data_setup['lt']
    tmax = data_setup['tmax']
    mean_x0 = torch.tensor([data_setup['mean_r'], data_setup['mean_theta']])
    covar_x0 = data_setup['sigma'] * torch.eye(n)
    covar_noise = data_setup['var_noise'] * torch.eye(m*N, dtype=torch.double)

    # Import data.
    dT = torch.from_numpy( np.load(data_path + '/dT.npy') )
    mean_U = torch.from_numpy( np.load(data_path + '/mean_U.npy') ).unsqueeze(1)
    mean_dU = torch.from_numpy( np.load(data_path + '/mean_dU.npy') ).unsqueeze(1)

    # Instantiate model.
    # l = data_setup['l']
    A_train = None # torch.tensor([[-l, 0.0], [0.0, 0.0]])
    B_train = torch.tensor([[0.0], [1.0]])
    C_train = None # torch.tensor([[1, 0.0], [0.0, 1/scale]])
    D_train = torch.zeros(m,p)

    model = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A_train, B_train, C_train, D_train).to(device)

    model_settings = {'m':m, 'n':n, 'p':p, 'lt':lt, 'mean_x0':mean_x0, 'covar_x0':covar_x0, 'A_train':A_train,
                      'B_train':B_train, 'C_train':C_train, 'D_train':D_train}


    # Instantiate optimiser.
    lr = 1e-1
    wd = 1e-5 # weight decay
    decay = 0.1 # scalar to multiply lr by at decay_epochs
    decay_epochs = [] # epochs to perform lr cut
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)  # Includes GaussianLikelihood parameters

    # Instantiate loss - the marginal log likelihood.
    mll = MLL()
    min_epoch = 0
    max_epoch = 25

    train_settings = {'batches':batches, 'bs':bs, 'test_split':test_split, 'lr':lr, 'wd':wd,
                      'decay':decay, 'decay_epochs':decay_epochs, 'max_epoch':max_epoch}


    with open(model_dir + 'model_settings.pkl', 'wb') as f:
        pickle.dump(model_settings, f)
        f.close()
    with open(model_dir + 'train_settings.pkl', 'wb') as f:
        pickle.dump(train_settings, f)
        f.close()


    # Train model.
    print( 'parameter count =', sum(p.numel() for p in model.parameters() if p.requires_grad) )
    start_time = time.time()
    print( "--- %s seconds ---" % (start_time) )


    dT = dT.to(device)
    mean_U = mean_U.to(device)
    mean_dU = mean_dU.to(device)
    covar_noise = covar_noise.to(device)


    model, mean_train_losses, mean_test_losses = train(dT, tmax, mean_U, mean_dU, covar_noise, train_loader, test_loader, max_epoch, model, optimizer, mll, decay, decay_epochs, model_dir, device)


    exe_time = time.time() - start_time
    print( "--- %s seconds ---" % (exe_time) )


    train_settings = {'batches':batches, 'bs':bs, 'test_split':test_split, 'lr':lr, 'wd':wd,
                      'decay':decay, 'decay_epochs':decay_epochs, 'max_epoch':max_epoch,
                      'exe_time':exe_time}

    with open(model_dir + 'train_settings.pkl', 'wb') as f:
        pickle.dump(train_settings, f)
        f.close()


    return 0



if __name__ == '__main__':
    main()

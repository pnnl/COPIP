import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import sys
root = '/qfs/projects/atscale/atscale_dyn/'
sys.path.append(root + 'Code/atscale/GP/SEGP')
import GP_Tools as GPT

"""
Set of functions for training a SEGPVAE.
"""



def getData(data_path:str, test_split:int, bs:int, n_batch:int):
    """ Function for loading, preprocessing and splitting data. Returns dataloaders with
        shape (batch, tmax, py, px).
        args:
               data_path: String specifying directory where the dataset is saved.
              test_split: Ratio of trajectories to be reserved for testing.
                      bs: Batch size.
                 n_batch: Number of batches equal to train_batch + test_batch.
     returns: 
            train_loader: Iterable that pass samples of training data in minibatches.
             test_loader: Iterable that pass samples of test data in minibatches.
                       T: Tensor containing time array.
                  mean_U: Tensor containing the continuous mean function for the input.
               mean_U_dt: Tensor containing the sampled mean function of the input.
               exp_setup: Dictionary containing parameter values used for the data generation.
    """

    with open(data_path+'exp_setup.npy', 'rb') as f:
        exp_setup = np.load(f, allow_pickle=True)
    with open(data_path+'vid_batch_all.npy', 'rb') as f:
        train_data = torch.from_numpy( np.load(f) )
    with open(data_path+'dT.npy', 'rb') as f:
        T = torch.from_numpy( np.load(f) )
    with open(data_path+'mean_U.npy', 'rb') as f:
        mean_U = torch.from_numpy( np.load(f) )
    with open(data_path+'mean_U_dt.npy', 'rb') as f:
        mean_U_dt = torch.from_numpy( np.load(f) )

    
    exp_setup = exp_setup.tolist()
    seed = exp_setup['seed'][0]

    train_data = train_data.to(dtype=torch.double)
    exp, trials, tmax, py, px = train_data.shape
    N = exp*trials
    
    if not (bs*n_batch == N):
        print('N = ', N)
        raise Exception("bs*n_batch must be the same as N!")

    # stack n_exp, batch dimensions of train_data into one axis
    train_data = torch.flatten(train_data, start_dim=0, end_dim=1) # (n_exp*batch, tmax, py, px)

    # split into train and test datasets
    rng = np.random.default_rng(seed=seed+1)
    test_batch = int(N*test_split)
    test_batch = rng.choice( np.arange(0, N), size=test_batch, replace=False )
    train_batch = np.delete( np.arange(0, N), test_batch )    
    test_data = train_data[test_batch] # (test_batch, tmax, py, px)
    train_data = train_data[train_batch] # (train_batch, tmax, py, px)
    
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

    return train_loader, test_loader, T, mean_U, mean_U_dt, exp_setup



def lr_scheduler(epoch:int, optimizer, decay:float, decay_epochs:list):
    """
    Decay learning rate by a factor decay for every epoch in decay_epochs.
    args:
               epoch: Current epoch of training loop.
           optimizer: Optimizer with parameters from previous epoch.
               decay: Scalar to multiply lr by.
        decay_epochs: List containing the epochs which the lr should be cut at.
    returns:
           optimizer: Same optimizer as before with updated lr.
    """

    if epoch in decay_epochs:
      for param_group in optimizer.param_groups:
          param_group['lr'] = decay*param_group['lr']

      print( 'New learning rate is: %.4f' % ( param_group['lr']) )

    return optimizer



def train(train_loader, test_loader, MS, T:torch.tensor, enc, GP, dec, optimizer, 
          criterion, max_epoch:int, model_dir:str, decay:float, decay_epochs:list):
    """
    Training loop.
    args:
       train_loader: Iterable that pass samples of training data in minibatches.
        test_loader: Iterable that pass samples of test data in minibatches.
                 MS: Modify_Shape class for reshaping data to the required form for another function. 
                  T: Tensor containing time array.
                enc: Encoder component of model.
                 GP: Gaussian process component of model.
                dec: Decoder component of model.
          optimizer: Chosen optimizer.
          criterion: ELBO.
          max_epoch: Epoch which training will terminate at.
          model_dir: Path to where models and data are stored.
              decay: Scalar to multiply lr by.
       decay_epochs: List containing the epochs which the lr should be cut at.
    returns:
                enc: Final version of the encoder.
                 GP: Final version of the GP.
                dec: Final version of the decoder.
            stats: Dataframe containing training and test loss.
    """
    

    tmax = len(T)
    train_losses = []
    test_losses = []
    
    
    for epoch in range(max_epoch):
            
        enc.train()
        GP.train() 
        dec.train()
    
        batch_losses = []

        for batch_idx, vid_batch in enumerate(train_loader):
            
            optimizer.zero_grad()

            # model outputs
            mu_lhood, var_lhood = enc(vid_batch) # (bs*tmax, m)
            mu_prior, K_prior = GP(T) # (m*tmax) and (m*tmax, m*tmax)

            print('A = ', GP.A)
            print('B = ', GP.B)
            print('C = ', GP.C)
            print('D = ', GP.D)
            GPT.test_K(K_prior)

            mu_post, K_post = GP.posterior( MS.prior_2_lml(mu_prior, K_prior), 
                                           MS.enc_2_lml(mu_lhood, var_lhood) ) # (bs*m*tmax) & (bs*m*tmax, bs*m*tmax)

            print('Posterior computed!')
            
            samples = GP.sample_posterior( MS.post_2_dec(mu_post, K_post) ) # (bs, tmax, m) 
            p_theta_logits = dec(samples) # (bs, tmax, py*px)

            print('Posterior sampled!')
            
            # loss
            recon = criterion.recon(p_theta_logits, torch.flatten(vid_batch, start_dim=2, end_dim=3))
            GCE = criterion.GCE( mu_post, K_post, MS.enc_2_lml(mu_lhood, var_lhood) )
            lml = criterion.LML( MS.prior_2_lml(mu_prior, K_prior), 
                                MS.enc_2_lml(mu_lhood, var_lhood) )
            loss = -(recon + GCE + lml)

            print('recon = ', recon)
            print('GCE = ', GCE)
            print('lml = ', lml)
            print('loss = ', loss)

            print('loss computed!')
            
            loss.backward()
            optimizer.step()
            batch_losses.append( loss.item() )

        mean_train_loss = np.mean(batch_losses) # mean loss over epoch
        train_losses.append(mean_train_loss)

        optimizer = lr_scheduler(epoch, optimizer, decay, decay_epochs)

        print('starting test loop!')
        
        # testing loop
        enc.eval()
        GP.eval() 
        dec.eval()
        batch_losses = []
        
        with torch.no_grad():
            for batch_idx, vid_batch in enumerate(test_loader):
                
                # model outputs
                mu_lhood, var_lhood = enc(vid_batch) # (bs*tmax, m)        
                mu_post, K_post = GP.posterior( MS.prior_2_lml(mu_prior, K_prior), 
                                               MS.enc_2_lml(mu_lhood, var_lhood) ) # (bs*m*tmax) & (bs*m*tmax, bs*m*tmax)  
                samples = GP.sample_posterior( MS.post_2_dec(mu_post, K_post) ) # (bs, tmax, m) 
                p_theta_logits = dec(samples) # (bs, tmax, py*px)
            
                # loss
                recon = criterion.recon(p_theta_logits, torch.flatten(vid_batch, start_dim=2, end_dim=3))
                GCE = criterion.GCE( mu_post, K_post, MS.enc_2_lml(mu_lhood, var_lhood) )
                lml = criterion.LML( MS.prior_2_lml(mu_prior, K_prior), 
                                MS.enc_2_lml(mu_lhood, var_lhood) )
                loss = -(recon + GCE + lml)
            
                batch_losses.append( loss.item() )                

        mean_test_loss = np.mean(batch_losses) # mean loss over epoch
        test_losses.append(mean_test_loss)

        print( 'Epoch %d/%d - Train loss: %.3f - Test loss: %.3f' % (epoch + 1, max_epoch, mean_train_loss, mean_test_loss) )

        # save models
        enc_path = model_dir + 'Encoder/epoch{:03d}.pt'.format(epoch+1)
        GP_path = model_dir + 'GP/epoch{:03d}.pt'.format(epoch+1)
        dec_path = model_dir + 'Decoder/epoch{:03d}.pt'.format(epoch+1)
        torch.save(enc.state_dict(), enc_path)
        torch.save(GP.state_dict(), GP_path)
        torch.save(dec.state_dict(), dec_path)
            
        # save stats.
        stats_path = model_dir + 'stats.csv'
        stats = pd.DataFrame()
        stats["train loss"] = train_losses # mean loss over epoch
        stats["test loss"] = test_losses # mean loss over epoch
        stats.to_csv(stats_path, index=False)
        
    return model, stats



def plot_loss(loc:str, file:str, stats:pd.DataFrame, max_epoch:int, y_lim=None):
    """
    Plot the training and test loss of a single run.
    args:
        loc: location to save figure.
       file: to save figure as.
      stats: data to plot.
  max_epoch: for creating x axis data.
    """

    epoch = np.arange(1, max_epoch+1)

    fig, ax1 = plt.subplots(1, 1)
    fig.tight_layout()
    
    ax1.plot(epoch, stats['train loss'], label='train')
    ax1.plot(epoch, stats['test loss'], label='test')
    
    if y_lim != None:
        ax1.set_ylim(y_lim)
        
    ax1.set_xlabel('Epoch')
    ax1.set_title('ELBO loss')
    ax1.legend()

    fig.savefig(loc + file + '.png')
    plt.show()

    return 0



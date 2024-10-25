import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt



"""
Set of functions for training a GP with physics enhanced kernel.
"""


def get_Data(data_path:str, Y_data:str, T_data:str, bs:int, n_batch:int, test_split:float):
    """
    Load data, split into train and test sets and further split into batches. The batches are 
    ordered as follows Y_11, ... Y_1m, ..., Y_bs1, ..., Y_bsm where Y_ij denotes the time teries
    of component j in trajectory i.
    args:
          data_path: Location where data is stored.
             Y_data: Numpy file containing Y data to load.
             T_data: Numpy file containing T data to load.
                 bs: Batch size.
            n_batch: Number of batches equal to train_batch + test_batch.
         test_split: Ratio of trajectories to be reserved for testing.
    returns:
            Y_train: Training data set (train_batch, bs*m*tmax)
             Y_test: Testing data set (test_batch, bs*m*tmax) 
                  T: Input data.
          exp_setup: List containing information about parameters used to generate the data.
    """

    
    with open(data_path+'exp_setup.npy', 'rb') as f:
        exp_setup = np.load(f, allow_pickle=True)
    with open(data_path+Y_data, 'rb') as f:
        Y = torch.from_numpy( np.load(f) )
    with open(data_path+'mean_U.npy', 'rb') as f:
        mean_U = torch.from_numpy( np.load(f) )
    with open(data_path+'mean_U_dt.npy', 'rb') as f:
        mean_U_dt = torch.from_numpy( np.load(f) )
    with open(data_path+T_data, 'rb') as f:
        T = torch.from_numpy( np.load(f) )

    exp_setup = exp_setup.tolist()
    rseed = exp_setup['seed']+3
    
    exp_no, batch, tmax, m = Y.shape
    N = exp_no*batch
    
    if not (bs*n_batch == N):
        print('N = ', N)
        raise Exception("bs*n_batch must be the same as N!")
    
    Y = torch.transpose(Y,2,3) # (exp_no, batch, m, tmax)
    Y = torch.reshape(Y, (exp_no, batch, m*tmax))

    Y2 = torch.zeros((N, m*tmax))
    for l in range(exp_no):
        for k in range(batch):
            Y2[(l*batch)+k,:] = Y[l,k,:]

    # split into Y_train and Y_test
    rng = np.random.default_rng(seed=rseed+1)
    test_batch = int(N*test_split)
    test_batch = rng.choice( np.arange(0, N), size=test_batch, replace=False )
    train_batch = np.delete( np.arange(0, N), test_batch )    
    Y_test = Y2[test_batch]
    Y_train = Y2[train_batch]

    test_batch = int( len(test_batch) / bs )
    train_batch = int( len(train_batch) / bs )
    
    Y_test2 = torch.zeros((test_batch, bs, m*tmax))
    Y_train2 = torch.zeros((train_batch, bs, m*tmax))

    # reshaping data so it can be passed into the loss function as batches.
    for l in range(test_batch):
        for k in range(bs):
            Y_test2[l,k] = Y_test[(l*bs)+k]

    for l in range(train_batch):
        for k in range(bs):
            Y_train2[l,k] = Y_train[(l*bs)+k]
    
    Y_test = torch.zeros((test_batch, bs*m*tmax))
    tmax2 = 2*tmax
    for l in range(test_batch):
        for k in range(bs):
            Y_test[l, k*tmax2:(k+1)*tmax2] = Y_test2[l,k]

    Y_train = torch.zeros((train_batch, bs*m*tmax))
    tmax2 = 2*tmax
    for l in range(train_batch):
        for k in range(bs):
            Y_train[l, k*tmax2:(k+1)*tmax2] = Y_train2[l,k]

    return Y_test, Y_train, T, exp_setup, mean_U, mean_U_dt



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



def train(Y_train:torch.tensor, Y_test:torch.tensor, T:torch.tensor, model, optimizer, 
          criterion, max_epoch:int, model_dir:str, decay:float, decay_epochs:list):
    """
    Training loop.
    args:
          Y_train: Training data with shape (train_batch, bs*m*tmax).
           Y_test: Test data with shape (test_batch, bs*m*tmax).
                T: Time points which training and test data was sampled at.
            model: Instantiation of the chosen model.
        optimizer: Chosen optimizer.
        criterion: Loss function.
        max_epoch: Epoch which training will terminate at.
        model_dir: Path to where models and data are stored.
            decay: Scalar to multiply lr by.
     decay_epochs: List containing the epochs which the lr should be cut at.
    returns:
            model: Final version of the model.
            stats: Dataframe containing training and test loss.
    """
    
    train_batch = Y_train.shape[0]
    test_batch = Y_test.shape[0]
    tmax = len(T)

    train_losses = []
    test_losses = []
    
    for epoch in range(max_epoch):
            
        model.train()
        batch_losses = []

        for i in range(train_batch):
            optimizer.zero_grad()
            mu, K = model(T)
            loss = criterion(Y_train[i], model)
            loss.backward()
            optimizer.step()
            batch_losses.append( loss.item() )

        mean_train_loss = np.mean(batch_losses) # mean loss over epoch
        train_losses.append(mean_train_loss)

        optimizer = lr_scheduler(epoch, optimizer, decay, decay_epochs)
        
        # testing loop
        model.eval()
        batch_losses = []
        
        with torch.no_grad():
            for i in range(test_batch):
                mu, K = model(T)
                loss = criterion(Y_test[i], model)
                batch_losses.append( loss.item() )                

        mean_test_loss = np.mean(batch_losses) # mean loss over epoch
        test_losses.append(mean_test_loss)

        print( 'Epoch %d/%d - Train loss: %.3f - Test loss: %.3f' % (epoch + 1, max_epoch, mean_train_loss, mean_test_loss) )

        # save models
        model_path = model_dir + 'epoch{:03d}.pt'.format(epoch+1)
        torch.save(model.state_dict(), model_path)
            
        # save stats.
        stats_path = model_dir + 'stats.csv'
        stats = pd.DataFrame()
        stats["train loss"] = train_losses # mean loss over epoch
        stats["test loss"] = test_losses # mean loss over epoch
        stats.to_csv(stats_path, index=False)
        
    return model, stats



def plot_mse(loc:str, file:str, stats:pd.DataFrame, max_epoch:int, y_lim=None):
    """
    Plot the mse training and test loss of a single run.
    args:
        loc: location to save figure.
       file: to save figure as.
      stats: data to plot.
  max_epoch: for creating x axis data.
    """

    epoch = np.arange(1, max_epoch+1)

    fig, ax1 = plt.subplots(1, 1)
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



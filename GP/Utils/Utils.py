import torch
import matplotlib.pyplot as plt
import numpy as np



def scale_data(data:torch.tensor, scale:float, dim:int):
    """
    Function for scaling the specified  dimension of the data by a factor scale.
    args:
         data: Data matrix with shape (M, Q, N, m).
        scale: Value to divide dimension m=dim by.
          dim: Dimension m to scale.
    returns:
         data: Data matrix with m=dim scaled by scale.
    """

    M, Q, N, m = data.shape

    data = data.view(M*Q, N, m) # shape = (M*Q, N, m)
    data_dim = data[:,:,dim] # (M*Q, N)
    data[:,:,dim] = data_dim / scale

    return data.view(M,Q,N,m)



def scale_lhood(lhood:torch.tensor, scale:float, dim:int):
    """
    Function for scaling the specified dimension of the likelihood mean.
    args:
        lhood: Mean of likelihood distribution with shape (bs, N, m).
        scale: Value to divide dimension m=dim by.
          dim: Dimension m to scale.
returns:
        lhood: mean of likelihood with m=dim scaled by scale.
    """

    lhood_dim = lhood[:,:,dim] # (bs, N)
    lhood[:,:,dim] = lhood_dim / scale

    return lhood



def scale_mean(mean:torch.tensor, scale:float, dim:int):
    """
    Function for scaling the specified dimension of the GP mean.
    args:
         mean: Mean of GP with shape (N, m).
        scale: Value to divide dimension m=dim by.
          dim: Dimension m to scale.
returns:
        mean: mean of GP with m=dim scaled by scale.
    """

    mean_dim = mean[:,dim] # (N).
    mean[:,dim] = mean_dim / scale # (N,m).

    return mean



def scale_covar(covar:torch.tensor, m:int, scale:float, dim:int):
    """
    Function for scaling the specified dimension of the GP covariance matrix.
    args:
         covar: Covariance matrix of GP with shape (m*N, m*N).
             m: Dimension of multivariate GP.
         scale: Value to divide dimension m=dim by.
           dim: Dimension m to scale.
returns:
  scaled_covar: Covariance matrix of GP with m=dim scaled by scale.
    """
    
    mN = covar.shape[0]
    N = int(mN/m)

    scale_matrix = torch.eye(m*N) 
    scale_matrix_dim = scale_matrix[dim*N:(dim+1)*N, dim*N:(dim+1)*N] / scale
    scale_matrix[dim*N:(dim+1)*N, dim*N:(dim+1)*N] = scale_matrix_dim
    
    scaled_covar = torch.matmul(covar, scale_matrix)
    scaled_covar = torch.matmul(scale_matrix, scaled_covar)
    
    return scaled_covar



def plot_latents(loc:str, file_name:str, vid_batch:torch.tensor, dY:torch.tensor, dT:torch.tensor,
                 tmax:float, nplots:int, recon_batch=None, recon_traj=None, recon_covar=None):
    """
    Plots videos, latent states and reconstructions.
    args:
             loc: location to store file.
       file_name: filename.
       vid_batch: (_, N, d, d) tensor of videos.
              dY: (_, N, m) tensor of latent states.
              dT: (N) tensor of time.
            tmax: Maximum time point.
          nplots: number of cols of plot, col row is one video.
     recon_batch: (_, N, d, d) tensor of reconstructed videos.
      recon_traj: (_, N, m) tensor of reconstructed trajectory.
     recon_covar: (_, m*N, m*N) covariance matrix of lent state.
    returns:
              ax: figure object with all plots.
    """
    M, Q, N, d, _ = vid_batch.shape
    vid_batch = vid_batch.reshape(M*Q, N, d, d)

    _, _, K, m = dY.shape
    dY = dY.reshape(M*Q, K, m)

    if recon_batch is not None:
        fig, ax = plt.subplots(3, nplots, figsize=(6, 8))
    else:
        fig, ax = plt.subplots(2, nplots, figsize=(6, 4))

    fig.tight_layout()


    # get axis limits for the latent space
    xmin = torch.min(dY[:,:,0].min().cpu() - 0.1).numpy()
    xmax = torch.max(dY[:,:,0].max().cpu() + 0.1).numpy()
    ymin = torch.min(dY[:,:,1].min().cpu() - 0.1).numpy()
    ymax = torch.max(dY[:,:,1].max().cpu() + 0.1).numpy()


    def make_heatmap(vid):
        """
        args:
               vid: N, d, d
        returns:
          flat_vid: d, d
        """
        vid = torch.stack( [(t+4)*v for t,v in enumerate(vid)], dim=0 )
        flat_vid = (1/(4+tmax)) * torch.max(vid, 0).values
        return flat_vid


    def plot_set(i, j):
        # i: batch element
        # j: plot column

        # first row is original video
        vid = make_heatmap(vid_batch[i,:,:,:])
        ax[0][j].imshow(1-vid.cpu(), origin='lower', cmap='Greys')
        ax[0][j].axis('off')

        # second row is trajectories
        ax[1][j].plot(dY[i,:,0].cpu(), dY[i,:,1].cpu())
        ax[1][j].set_xlim([xmin, xmax])
        ax[1][j].set_ylim([ymin, ymax])
        ax[1][j].scatter(dY[i,0,0].cpu(), dY[i,0,1].cpu(), marker='o', c='C0')
        ax[1][j].scatter(dY[i,-1,0].cpu(), dY[i,-1,1].cpu(), marker='*', c='C0')

        if recon_traj is not None:
            ax[1][j].plot(recon_traj[i,:,0].cpu(), recon_traj[i,:,1].cpu(), c='C1')
            ax[1][j].scatter(recon_traj[i,0,0].cpu(), recon_traj[i,0,1].cpu(), marker='o', c='C1')
            ax[1][j].scatter(recon_traj[i,-1,0].cpu(), recon_traj[i,-1,1].cpu(), marker='*', c='C1')

        # Third row is reconstructed video
        if recon_batch is not None:
            recon = make_heatmap(recon_batch[i,:,:,:])
            ax[2][j].imshow(1-recon.cpu(), origin='lower', cmap='Greys')
            ax[2][j].axis('off')

    for i in range(nplots):
        rand_idx = np.random.randint(0, M*Q, nplots)
        plot_set(rand_idx[i], i)

    fig.savefig(loc + '/' + file_name + '.png')
    plt.show()


    if recon_traj is not None:
        recon_var = torch.diag( recon_covar[0] ) # shape = (m*N)
        recon_conf = 2 * torch.sqrt( recon_var ).cpu() # shape = (m*N)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].plot( dT.cpu(), dY[0,:,0].cpu(), 'r*' )
        ax[0].plot( dT.cpu(), recon_traj[0,:,0].cpu() )
        ax[0].fill_between(dT.cpu(), recon_traj[0,:,0].cpu() - recon_conf[:N],
                            recon_traj[0,:,0].cpu() + recon_conf[:N], alpha=0.5, label='95% CI')
        ax[1].plot( dT.cpu(), dY[0,:,1].cpu(), 'r*' )
        ax[1].plot( dT.cpu(), recon_traj[0,:,1].cpu() )
        ax[1].fill_between(dT.cpu(), recon_traj[0,:,1].cpu() - recon_conf[N:],
                            recon_traj[0,:,1].cpu() + recon_conf[N:], alpha=0.5, label='95% CI')

        fig.savefig(loc + '/' + file_name + '_posterior.png')
        plt.show()

    return 0



def plot_data(data_path, file_name, T, U, Z, dZn, var_noise):

    M, Q, K, n = Z.shape
    form_var = f"{var_noise:.3f}"[1:]

    # Randomly select q input signals and q-1 trajectories in response to each input.
    q = 3
    rand_M = np.random.randint(0, M, q)
    rand_Q = np.random.randint(0, Q, q-1)

    # Create a figure with p subplots side by side.
    fig, axes = plt.subplots(q, 3, figsize=(3*q, q*q))  # q rows, 3 columns.

    # First column - plot u(t).
    for i in range(q):
      randM = rand_M[i]
      axes[i,0].plot(T, U[randM], color="green")
      axes[i,0].set_title("Input Signal \n M = {}".format(randM))
      axes[i,0].set_xlabel("t")
      axes[i,0].set_ylabel("u")
      axes[i,0].grid(True)

    # Second column - plot clean and continuous trajectories in 2d.
    for i in range(q):
      for j in range(q-1):
        randM = rand_M[i]
        randQ = rand_Q[j]
        axes[i,1].plot(Z[randM, randQ, :, 0], Z[randM, randQ, :, 1], label="Q = {}".format(randQ))
        axes[i,1].scatter(Z[randM, randQ, 0, 0], Z[randM, randQ, 0, 1], color="red", marker="o")
        axes[i,1].scatter(0, 0, color="red", marker="*")
        axes[i,1].set_title("Trajectories in Response to \n Adjacent Input")
        axes[i,1].set_xlabel("x")
        axes[i,1].set_ylabel("y")
        axes[i,1].grid(True)
        axes[i,1].legend()

    # Third column - plot noisy and discretized trajectories in 2d.
    for i in range(q):
      for j in range(q-1):
        randM = rand_M[i]
        randQ = rand_Q[j]
        axes[i,2].scatter(dZn[randM, randQ, :, 0], dZn[randM, randQ, :, 1], s=3, label="Q = {}".format(randQ))
        axes[i,2].set_title("Sampled and Noisy (var = {}) \n Adjacent Trajectories".format(form_var) )
        axes[i,2].set_xlabel("x")
        axes[i,2].set_ylabel("y")
        axes[i,2].grid(True)

    # Adjust layout for better spacing.
    plt.tight_layout()

    # Save figure
    plt.savefig(data_path + "/" + file_name)

    # Show the plots
    plt.show()

    return 0



def plot_individual_dims(data_path, file_name, dT, dU, dZn, var_noise):
    
    M, Q, N, n = dZn.shape
    form_var = f"{var_noise:.3f}"[1:]

    # Randomly select q input signals and q-1 trajectories in response to each input.
    q = 3
    rand_M = np.random.randint(0, M, q)
    rand_Q = np.random.randint(0, Q, q-1)

    # Create a figure with p subplots side by side.
    fig, axes = plt.subplots(q, 3, figsize=(3*q, q*q))  # q rows, 3 columns.

    # First column - plot u(t).
    for i in range(q):
      randM = rand_M[i]
      axes[i,0].scatter(dT, dU[randM], color="green")
      axes[i,0].set_xlabel("t")
      axes[i,0].set_ylabel("u")
      axes[i,0].set_title("Input Signal \n M = {}".format(randM))
      axes[i,0].grid(True)

    # Second column - plot noisy first dimension.
    for i in range(q):
      for j in range(q-1):
        randM = rand_M[i]
        randQ = rand_Q[j]
        axes[i,1].scatter(dT, dZn[randM, randQ, :, 0], color="blue", marker="o")
        axes[i,1].set_xlabel("t")
        axes[i,1].set_ylabel("Radius")
        axes[i,1].set_title("Radius in Response to \n Adjacent Input")
        axes[i,1].grid(True)

    # Third column - plot noisy second dimension.
    for i in range(q):
      for j in range(q-1):
        randM = rand_M[i]
        randQ = rand_Q[j]
        axes[i,2].scatter(dT, dZn[randM, randQ, :, 1], color="blue", marker="o")
        axes[i,2].set_xlabel("t")
        axes[i,2].set_ylabel("Theta")
        axes[i,2].set_title("Theta in Response to \n Adjacent Input")
        axes[i,2].grid(True)

    # Adjust layout for better spacing.
    plt.tight_layout()

    # Save figure
    plt.savefig(data_path + "/" + file_name)

    # Show the plots
    plt.show()

    return 0



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



def test_covar(covar, tol_sym=1e-3, tol_pos=1e-3):
    """
    Tests if covariance matrix is symmetric and positive definite.
    args:
          covar: Covariance matrix.
        tol_sym: tolerance for  symmetry check.
        tol_pos: tolerance for pos. def. check.
    returns:
        True if both conditions are met.
    """
    dim = covar.shape[0]
    eigs = torch.linalg.eigvals(covar)
    real_eigs = torch.real(eigs)

    sym_test = torch.allclose( covar, covar.t(), atol=tol_sym )
    eig_test = torch.all( torch.ge( real_eigs, -tol_pos*torch.ones(dim) ) )

    if not sym_test:
        raise Exception("Sorry, matrix is not symmetric!")
    if not eig_test.item():
        raise Exception("Sorry, matrix is not positive definite!")

    return True

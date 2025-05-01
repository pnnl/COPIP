import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp



def scale_mean(scaling_vec:torch.tensor, mean:torch.tensor):
    """
    Function for scaling the mean vector. This function assumes m=2.
    args: 
        scaling_vec: Tensor containing the scaling terms (m).
               mean: Mean vector. Could be (bs, N, m) or (N, m).
    returns:
        mean_scaled: Scaled mean vector with shape corresponding to mean.
    """
    
    # Get shape of mean.
    try:
        bs, N, m = mean.shape # will work for lhood, posterior
    except:
        N, m = mean.shape # will work for prior
        bs = None
    
    # Scale m=0 dim, then m=1 dim.
    if bs == None:
        mean_scaled = torch.zeros(N, m)
        mean_scaled[:,0] = scaling_vec[0] * mean[:,0]
        mean_scaled[:,1] = scaling_vec[1] * mean[:,1]
    else:
        mean_scaled = torch.zeros(bs, N, m)
        mean_scaled[:,:,0] = scaling_vec[0] * mean[:,:,0]
        mean_scaled[:,:,1] = scaling_vec[1] * mean[:,:,1]

    return mean_scaled



def scale_covar(scaling_vec:torch.tensor, covar:torch.tensor):
    """
    Function for scaling the covariance matrix. This function assumes m=2.
    args: 
        scaling_vec: Tensor containing the scaling terms (m).
              covar: Covariance matrix. Could be (bs, m*N, m*N) or (m*N, m*N).
    returns:
        covar_scaled: Scaled covariance matrix  with shape corresponding to covar.
    """

    # Get shape of covar.
    try:
        bs, mN, _ = covar.shape # will work for lhood, posterior
    except:
        mN, _ = covar.shape # will work for prior
        bs = None
    
    m = len(scaling_vec)

    # Augment scaling_vec into scaling matrix of appropriate shape.
    N = int(mN/m)
    scaling_mat = torch.eye(mN)
    scaling_mat[:N, :N] = scaling_vec[0] * torch.eye(N)
    scaling_mat[N:2*N, N:2*N] = scaling_vec[1] * torch.eye(N)
    
    # Scale covar.
    if  bs == None:
        covar_scaled = torch.matmul(covar, scaling_mat.t() )
        covar_scaled = torch.matmul(scaling_mat, covar_scaled)
    else:
        covar_scaled = torch.bmm(covar, scaling_mat.t().repeat(bs,1,1))
        covar_scaled = torch.bmm(scaling_mat.repeat(bs,1,1), covar_scaled)

    return covar_scaled



def MSE_projection(X, Y, VX=None):
    """
    Given X, project it onto Y.
    args:
          X: np array (bs, N, m).
          Y: np array (bs, N, m).
         VX: Covariance of X values as np array (bs, m*N, m*N).

    returns:
            Y_pred: affine transformation of X (bs, N, m).
                 W: nparray (m+1, m).
               MSE: ||Y - Y_pred||^2.
           VY_pred: cov matrix of Y_pred (bs, m*N, m*N).
    """

    bs, N, m = X.shape

    # Reshape to stack all samples: (bs * N, m)
    X_flat = X.reshape(-1, m)
    Y_flat = Y.reshape(-1, m)

    # Stack ones to include biases in W
    X_flat = np.hstack([X_flat, np.ones((bs*N, 1))]) # (bs*N, m+1)
    
    # Get least squares solution, W = (m+1, m), MSE = (2)
    W, MSE, rank, s = sp.linalg.lstsq(X_flat, Y_flat) 

    try:
        MSE = MSE[0] + MSE[1]
    except:
        MSE = np.nan

    # Get predicted Y.
    Y_pred = np.einsum('bnm,km->bnm', X, W)
    Y_pred = torch.from_numpy(Y_pred) # (bs, N, m) 

    # Get covariance matrix of Y_pred.
    if VX is not None:
      
      # Build Kronecker product: W ⊗ I_N -> shape: (q*N, m*N)
      I_N = np.eye(N)
      kron = np.kron(W[:m], I_N)  # shape = (q*N, m*N)

      # Transform covariance: K_new = kron.T @ K @ kron for each batch
      VY_pred = np.empty((bs, m*N, m*N))
      for b in range(bs):
        VXb = VX[b]
        VY_pred[b] = np.transpose(kron) @ VXb @ kron

      VY_pred = torch.from_numpy(VY_pred)
    else:
      VY_pred = None

    return Y_pred, torch.from_numpy(W), MSE, VY_pred



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
     recon_covar: (_, m*N, m*N) covariance matrix of reconstructed trajectory.
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


    def plot_set(j):
        # j: plot on column j.
        
        idx = np.random.randint(0, M*Q)

        # first row is original video
        vid = make_heatmap(vid_batch[idx,:,:,:])
        ax[0][j].imshow(1-vid.cpu(), origin='lower', cmap='Greys')
        ax[0][j].axis('off')

        # second row is trajectories
        ax[1][j].plot(dY[idx,:,0].cpu(), dY[idx,:,1].cpu(), label='True' )
        ax[1][j].set_xlim([xmin, xmax])
        ax[1][j].set_ylim([ymin, ymax])
        ax[1][j].scatter(dY[idx,0,0].cpu(), dY[idx,0,1].cpu(), marker='o', c='C0')
        ax[1][j].scatter(dY[idx,-1,0].cpu(), dY[idx,-1,1].cpu(), marker='*', c='C0')

        if recon_traj is not None:
            ax[1][j].plot(recon_traj[idx,:,0].cpu(), recon_traj[idx,:,1].cpu(), c='C1', label='Prediction')
            ax[1][j].scatter(recon_traj[idx,0,0].cpu(), recon_traj[idx,0,1].cpu(), marker='o', c='C1')
            ax[1][j].scatter(recon_traj[idx,-1,0].cpu(), recon_traj[idx,-1,1].cpu(), marker='*', c='C1')
        
        ax[1][j].legend()

        # Third row is reconstructed video
        if recon_batch is not None:
            recon = make_heatmap(recon_batch[idx,:,:,:])
            ax[2][j].imshow(1-recon.cpu(), origin='lower', cmap='Greys')
            ax[2][j].axis('off')

    for i in range(nplots):
        plot_set(i)

    fig.savefig(loc + '/' + file_name + '.png')
    plt.close()

    if recon_covar is not None:
        idx = np.random.randint(0, M*Q)
        recon_var = torch.diag( recon_covar[idx] ) # shape = (m*N)
        recon_conf = 2 * torch.sqrt( recon_var ).cpu() # shape = (m*N)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].plot( dT.cpu(), dY[idx,:,0].cpu(), 'r*' )
        ax[0].plot( dT.cpu(), recon_traj[idx,:,0].cpu() )
        ax[0].fill_between(dT.cpu(), recon_traj[idx,:,0].cpu() - recon_conf[:N],
                            recon_traj[idx,:,0].cpu() + recon_conf[:N], alpha=0.5, label='95% CI')
        ax[1].plot( dT.cpu(), dY[idx,:,1].cpu(), 'r*' )
        ax[1].plot( dT.cpu(), recon_traj[idx,:,1].cpu() )
        ax[1].fill_between(dT.cpu(), recon_traj[idx,:,1].cpu() - recon_conf[N:],
                            recon_traj[idx,:,1].cpu() + recon_conf[N:], alpha=0.5, label='95% CI')

        fig.savefig(loc + '/' + file_name + '_posterior.png')
        plt.close()

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
    plt.close()

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
    plt.close()

    return 0



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

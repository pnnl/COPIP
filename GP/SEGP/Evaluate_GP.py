import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pandas as pd
import seaborn as sns
import sys
import os

root = '/bask/projects/v/vjgo8416-lurienet/SEGP/'

# Directory SEGP model is stored.
GP_path = root + 'Code/SEGP'

# Utility functions directory.
Utils_path = root + 'Code/Utils'

# Add directories to path.
if GP_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(GP_path)

# Add directories to path.
if Utils_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(Utils_path)

# Import custom files.
import SEGP
from Train_GP import MLL
from Utils import scale_data, scale_mean, scale_covar



def find_bestmodel(stats):
    """
    find model with highest test accuracy.
    """

    best_model_idx = np.argmin(stats['test loss']) + 1
    best_loss = np.min(stats['test loss'])

    print('Best model was epoch{:03d}.pt '.format(best_model_idx) + 'with test loss {}'.format(best_loss) )

    return best_model_idx



def plot_loss(model_path, stats):

    train_loss = stats['train loss']
    test_loss = stats['test loss']
    max_epoch = len(train_loss)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(np.arange(1,max_epoch+1), train_loss, label='Train')
    axs[0].plot(np.arange(1,max_epoch+1), test_loss, label='Test')
    # axs[0].set_ylim([0, 10000])
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(np.arange(1,max_epoch+1), np.log(train_loss), label='Train')
    axs[1].plot(np.arange(1,max_epoch+1), np.log(test_loss), label='Test')
    # axs[1].set_ylim([6.5, 9])
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Log Loss')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(model_path + "1_Loss.png")
    plt.show()

    return 0



def plot_prior(model_path, dT, dZ, mean_lt, covar_lt, mean_gt, covar_gt):

    M, Q, N, m = dZ.shape

    # 95% Confidence intervals.
    conf_gt = 2 * torch.sqrt( torch.diag(covar_gt) ).cpu().detach().numpy()
    conf_lt = 2 * torch.sqrt( torch.diag(covar_lt) ).cpu().detach().numpy()

    fy, axs = plt.subplots(2, 2, figsize=(12, 8))
    ylt_ax = axs[:,0]
    ygt_ax = axs[:,1]

    rand_idx = np.random.randint([M-1, Q-1], size=(3,2))

    # Plot dimension 1 of sampled trajectories.
    ygt_ax[0].plot(dT.cpu(), dZ[rand_idx[0,0],rand_idx[0,1],:,0].cpu(), 'g*')
    ygt_ax[0].plot(dT.cpu(), dZ[rand_idx[1,0],rand_idx[1,1],:,0].cpu(), 'k*')
    ygt_ax[0].plot(dT.cpu(), dZ[rand_idx[2,0],rand_idx[2,1],:,0].cpu(), 'r*')

    ylt_ax[0].plot(dT.cpu(), dZ[rand_idx[0,0],rand_idx[0,1],:,0].cpu(), 'g*')
    ylt_ax[0].plot(dT.cpu(), dZ[rand_idx[1,0],rand_idx[1,1],:,0].cpu(), 'k*')
    ylt_ax[0].plot(dT.cpu(), dZ[rand_idx[2,0],rand_idx[2,1],:,0].cpu(), 'r*')

    # Plot dimension 2 of sampled trajectories.
    ygt_ax[1].plot(dT.cpu(), dZ[rand_idx[0,0],rand_idx[0,1],:,1].cpu(), 'g*')
    ygt_ax[1].plot(dT.cpu(), dZ[rand_idx[1,0],rand_idx[1,1],:,1].cpu(), 'k*')
    ygt_ax[1].plot(dT.cpu(), dZ[rand_idx[2,0],rand_idx[2,1],:,1].cpu(), 'r*')

    ylt_ax[1].plot(dT.cpu(), dZ[rand_idx[0,0],rand_idx[0,1],:,1].cpu(), 'g*')
    ylt_ax[1].plot(dT.cpu(), dZ[rand_idx[1,0],rand_idx[1,1],:,1].cpu(), 'k*')
    ylt_ax[1].plot(dT.cpu(), dZ[rand_idx[2,0],rand_idx[2,1],:,1].cpu(), 'r*')

    # Plot means.
    ygt_ax[0].plot(dT.cpu(), mean_gt[:,0].cpu().detach().numpy(), 'blue', label='Mean')
    ylt_ax[0].plot(dT.cpu(), mean_lt[:,0].cpu().detach().numpy(), 'blue', label='Mean')
    ygt_ax[1].plot(dT.cpu(), mean_gt[:,1].cpu().detach().numpy(), 'blue', label='Mean')
    ylt_ax[1].plot(dT.cpu(), mean_lt[:,1].cpu().detach().numpy(), 'blue', label='Mean')

    # Plot confidence intervals.
    ygt_ax[0].fill_between(dT.cpu(), mean_gt[:,0].cpu().detach().numpy() - conf_gt[:N],
                      mean_gt[:,0].cpu().detach().numpy() + conf_gt[:N], alpha=0.5, label='95% CI')

    ylt_ax[0].fill_between(dT.cpu(), mean_lt[:,0].cpu().detach().numpy() - conf_lt[:N],
                      mean_lt[:,0].cpu().detach().numpy() + conf_lt[:N], alpha=0.5, label='95% CI')

    ygt_ax[1].fill_between(dT.cpu(), mean_gt[:,1].cpu().detach().numpy() - conf_gt[N:],
                      mean_gt[:,1].cpu().detach().numpy() + conf_gt[N:], alpha=0.5, label='95% CI')

    ylt_ax[1].fill_between(dT.cpu(), mean_lt[:,1].cpu().detach().numpy() - conf_lt[N:],
                      mean_lt[:,1].cpu().detach().numpy() + conf_lt[N:], alpha=0.5, label='95% CI')


    # fy.suptitle('Comparison Between Ground Truth SEGP and Learnt SEGP Prior')
    ygt_ax[0].set_title('Ground Truth Prior')
    ylt_ax[0].set_title('Learnt Prior')
    ylt_ax[0].set_ylabel('Radius')
    ylt_ax[1].set_ylabel('Theta')
    ygt_ax[1].set_xlabel('Time (s)')
    ylt_ax[1].set_xlabel('Time (s)')
    ygt_ax[0].grid(True)
    ylt_ax[0].grid(True)
    ygt_ax[1].grid(True)
    ylt_ax[1].grid(True)
    ygt_ax[0].legend()
    ylt_ax[0].legend()
    ygt_ax[1].legend()
    ylt_ax[1].legend()

    plt.tight_layout()
    plt.savefig(model_path + "2_GT_versus_Learnt_Prior.png")
    plt.show()

    return 0



def plot_covar(model_path, covar_gt, covar_lt):

    N = int( covar_gt.shape[0] / 2 )

    fy, (covgt_ax, covlt_ax) = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = sns.heatmap(covar_gt.cpu().detach().numpy(), cmap='Blues', ax=covgt_ax)
    ax1 = sns.heatmap(covar_lt.cpu().detach().numpy(), cmap='Blues', ax=covlt_ax)

    # Specify the tick locations on the x-axis and y-axis.
    xtick_loc = np.arange(0, covar_gt.shape[0], 1)
    xtick_loc = xtick_loc[np.remainder(xtick_loc, 3) == 0]
    ytick_loc = np.arange(0, covar_gt.shape[0], 1)
    ytick_loc = ytick_loc[np.remainder(ytick_loc, 4) == 0]

    covgt_ax.xaxis.set_major_locator(FixedLocator(xtick_loc))
    covlt_ax.xaxis.set_major_locator(FixedLocator(xtick_loc))
    covgt_ax.yaxis.set_major_locator(FixedLocator(ytick_loc))
    covlt_ax.yaxis.set_major_locator(FixedLocator(ytick_loc))

    # Specify x-tick and y-tick labels to repeat count from 0 to N.
    xlabels = np.arange(0, N, 1)
    xlabels = xlabels[np.remainder(xlabels, 3) == 0]
    xlabels = np.concatenate((xlabels, xlabels[1:]))
    ylabels = np.arange(0, N, 1)
    ylabels = ylabels[np.remainder(ylabels, 4) == 0]
    ylabels = np.concatenate((ylabels, ylabels[1:]))

    # Specify ticks on x-axis and y-axis.
    covgt_ax.set_xticklabels(xlabels)
    covlt_ax.set_xticklabels(xlabels)
    covgt_ax.set_yticklabels(ylabels)
    covlt_ax.set_yticklabels(ylabels)

    # fy.suptitle('Comparison Between Ground Truth SEGP and Learnt SEGP Prior Covariance Matrix')
    covgt_ax.set_title('Ground Truth Prior Covariance Matrix')
    covlt_ax.set_title('Learnt Prior Covariance Matrix')
    covgt_ax.set_xlabel("$\Leftarrow$ $y_{1}(t\')$ | $y_{2}(t\')$ $\Rightarrow$")
    covlt_ax.set_xlabel("$\Leftarrow$ $y_{1}(t\')$ | $y_{2}(t\')$ $\Rightarrow$")
    covgt_ax.set_ylabel("$\Leftarrow$ $y_{2}(t)$ | $y_{1}(t)$ $\Rightarrow$")
    covlt_ax.set_ylabel("$\Leftarrow$ $y_{2}(t)$ | $y_{1}(t)$ $\Rightarrow$")

    plt.tight_layout()
    plt.savefig(model_path + "3_GT_versus_Learnt_Prior_Cov.png")
    plt.show()

    return 0



def get_posterior(T, Ts2, mean_U, model, dT, tmax, mean_dU, covar_noise, dZn):

    M, Q, N, m = dZn.shape

    # Compute new discretised arrays.
    dT2 = T[torch.remainder(torch.arange(T.shape[0]), Ts2) == 0] # shape = (N2)
    mean_dU2 = mean_U[torch.remainder(torch.arange(mean_U.shape[0]), Ts2) == 0, :] # shape = (N2, p)

    # Get random index.
    rand_idx = np.random.randint([M-1, Q-1], size=(1,2))
    rand_M, rand_Q = rand_idx[0]

    # Compute posterior mean and covariance.
    mean_post, covar_post = model.posterior(dT, dT2, tmax, tmax, mean_U, mean_U, mean_dU, mean_dU2,
                                                    covar_noise.repeat(1, 1, 1), dZn[rand_M,rand_Q].repeat(1,1,1) ) # shapes = (1, N2, m) and (1, m*N2, m*N2)

    mean_post = mean_post.squeeze(0) # shape = (N2, m)
    covar_post = covar_post.squeeze(0) # shape = (m*N2, m*N2)

    return dT2, mean_post, covar_post, rand_M, rand_Q


def plot_posterior_versus_prior(model_path, dT, dT2, dZ, mean_lt, covar_lt, mean_post, covar_post, rand_M, rand_Q):

    M, Q, N, m = dZ.shape
    N2 = len(dT2)

    # 95% Confidence intervals.
    conf_lt = 2 * torch.sqrt( torch.diag(covar_lt) ).cpu().detach().numpy()
    conf_post = 2 * torch.sqrt( torch.diag(covar_post) ).cpu().detach().numpy()

    fy, axs = plt.subplots(2, 2, figsize=(12, 8))
    ylt_ax = axs[:,0]
    ypost_ax = axs[:,1]

    # Plot trajectory.
    string = 'Traj. M={0}, Q={1}'.format(rand_M, rand_Q)
    ylt_ax[0].plot(dT.cpu(), dZ[rand_M,rand_Q,:,0].cpu(), 'r*', label=string)
    ylt_ax[1].plot(dT.cpu(), dZ[rand_M,rand_Q,:,1].cpu(), 'r*', label=string)
    ypost_ax[0].plot(dT.cpu(), dZ[rand_M,rand_Q,:,0].cpu(), 'r*', label=string)
    ypost_ax[1].plot(dT.cpu(), dZ[rand_M,rand_Q,:,1].cpu(), 'r*', label=string)

    # Plot means.
    ylt_ax[0].plot(dT.cpu(), mean_lt[:,0].cpu().detach().numpy(), 'blue', label='Mean')
    ylt_ax[1].plot(dT.cpu(), mean_lt[:,1].cpu().detach().numpy(), 'blue', label='Mean')
    ypost_ax[0].plot(dT2.cpu(), mean_post[:,0].cpu().detach().numpy(), 'blue', label='Posterior')
    ypost_ax[1].plot(dT2.cpu(), mean_post[:,1].cpu().detach().numpy(), 'blue', label='Posterior')

    # Plot confidence intervals.
    ylt_ax[0].fill_between(dT.cpu(), mean_lt[:,0].cpu().detach().numpy() - conf_lt[:N],
                      mean_lt[:,0].cpu().detach().numpy() + conf_lt[:N], alpha=0.5, label='95% CI')

    ylt_ax[1].fill_between(dT.cpu(), mean_lt[:,1].cpu().detach().numpy() - conf_lt[N:],
                      mean_lt[:,1].cpu().detach().numpy() + conf_lt[N:], alpha=0.5, label='95% CI')

    ypost_ax[0].fill_between(dT2.cpu(), mean_post[:,0].cpu().detach().numpy() - conf_post[:N2],
                      mean_post[:,0].cpu().detach().numpy() + conf_post[:N2], alpha=0.5, label='95% CI')

    ypost_ax[1].fill_between(dT2.cpu(), mean_post[:,1].cpu().detach().numpy() - conf_post[N2:],
                      mean_post[:,1].cpu().detach().numpy() + conf_post[N2:], alpha=0.5, label='95% CI')


    string = 'Comparison of Learnt Prior and Posterior Conditioned on {0} Noisy Observations of Traj. M={1}, Q={2}'.format(N2, rand_M, rand_Q)
    # fy.suptitle(string)
    ylt_ax[0].set_xlabel('Time (s)')
    ypost_ax[1].set_xlabel('Time (s)')
    ylt_ax[0].set_ylabel('Radius')
    ylt_ax[1].set_ylabel('Theta')
    ylt_ax[0].set_title('Learnt Prior')
    ypost_ax[0].set_title('Learnt Posterior')
    ylt_ax[0].grid(True)
    ylt_ax[1].grid(True)
    ypost_ax[0].grid(True)
    ypost_ax[1].grid(True)
    ylt_ax[0].legend()
    ylt_ax[1].legend()
    ypost_ax[0].legend()
    ypost_ax[1].legend()

    plt.tight_layout()
    plt.savefig(model_path + "4_Learnt_Posterior_versus_prior.png")
    plt.show()

    return 0



def sample_posterior(model_path, n_samples, dT2, mean_post, covar_post, rand_M, rand_Q):

    N2, m = mean_post.shape

    # Sample standard normal dist.
    sample_std = torch.randn( size=(N2, m, n_samples) ) # shape = (N2, m, n_samples)

    # Transform to samples from posterior.
    var = torch.diag(covar_post).unflatten( dim=0, sizes=(m, N2) ).t().unsqueeze(2).repeat(1, 1, n_samples) # shape = (N2, m, n_samples)
    mean = mean_post.unsqueeze(2).repeat(1, 1, n_samples) # shape = (N2, m, n_samples)
    samples = mean + (sample_std * torch.sqrt(var) ) # shape = (N2, m, n_samples)

    # 95% Confidence interval.
    conf_post = 2 * torch.sqrt( torch.diag(covar_post) ).cpu().detach().numpy()

    rand_idx = np.random.randint(n_samples, size=(3))
    a, b, c = rand_idx

    fy, ypost_ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot dimension 1 of sample trajectories.
    ypost_ax[0].plot(dT2.cpu(), samples[:,0,a].cpu().detach().numpy(), 'g*')
    ypost_ax[0].plot(dT2.cpu(), samples[:,0,b].cpu().detach().numpy(), 'k*')
    ypost_ax[0].plot(dT2.cpu(), samples[:,0,c].cpu().detach().numpy(), 'r*')

    # Plot dimension 2 of sample trajectories.
    ypost_ax[1].plot(dT2.cpu(), samples[:,1,a].cpu().detach().numpy(), 'g*')
    ypost_ax[1].plot(dT2.cpu(), samples[:,1,b].cpu().detach().numpy(), 'k*')
    ypost_ax[1].plot(dT2.cpu(), samples[:,1,c].cpu().detach().numpy(), 'r*')

    # Plot means.
    ypost_ax[0].plot(dT2.cpu(), mean_post[:,0].cpu().detach().numpy(), 'blue', label='Mean')
    ypost_ax[1].plot(dT2.cpu(), mean_post[:,1].cpu().detach().numpy(), 'blue', label='Mean')

    # Plot confidence intervals.
    ypost_ax[0].fill_between(dT2.cpu(), mean_post[:,0].cpu().detach().numpy() - conf_post[:N2],
                      mean_post[:,0].cpu().detach().numpy() + conf_post[:N2], alpha=0.5, label='95% CI')

    ypost_ax[1].fill_between(dT2.cpu(), mean_post[:,1].cpu().detach().numpy() - conf_post[N2:],
                      mean_post[:,1].cpu().detach().numpy() + conf_post[N2:], alpha=0.5, label='95% CI')


    string = 'Samples from Learnt Posterior Conditioned on {0} Noisy Observations of Traj. M={1}, Q={2}'.format(N2, rand_M, rand_Q)
    fy.suptitle(string)
    ypost_ax[0].set_xlabel('Time (s)')
    ypost_ax[1].set_xlabel('Time (s)')
    ypost_ax[0].set_ylabel('Radius')
    ypost_ax[1].set_ylabel('Theta')
    ypost_ax[0].grid(True)
    ypost_ax[1].grid(True)
    ypost_ax[0].legend()
    ypost_ax[1].legend()

    plt.tight_layout()
    plt.savefig(model_path + "5_Learnt_Posterior_and_Samples.png")
    plt.show()

    return 0



def main():
    
    # Hardware settings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print("Device in use is: ", device)


    # Paths.
    model_name = 'SEGP'
    exp_no = 2 # Experiment number.
    dataset_no = 1
    model_path = root + 'Models/' + model_name + '/Exp_{:03d}/'.format(exp_no)
    data_path = root + 'Data/Dataset{0}'.format(dataset_no)
    plots_path = model_path + 'Plots/'
    
    if os.path.isdir(plots_path):
        print("Plots path exists!")
    else:
        os.mkdir(plots_path)

    # Import training data.
    data_setup = np.load(data_path + '/data_setup.pkl', allow_pickle=True)
    dZ = torch.from_numpy( np.load(data_path + '/dZ.npy') ).float().to(device)
    dZn = torch.from_numpy( np.load(data_path + '/dZn.npy') ).float().to(device)
    T = torch.from_numpy( np.load(data_path + '/T.npy') ).to(device)
    dT = torch.from_numpy( np.load(data_path + '/dT.npy') ).to(device)
    mean_U = torch.from_numpy( np.load(data_path + '/mean_U.npy') ).unsqueeze(1).to(device)
    mean_dU = torch.from_numpy( np.load(data_path + '/mean_dU.npy') ).unsqueeze(1).to(device)
    
    M, Q, N, _ = dZn.shape
    
    # Scale data for computing posterior.
    scale = 5.5
    dim = 1
    dZn_theta = scale_data(dZn, scale, dim)

    # Import training stats.
    stats = pd.read_csv(model_path + '/stats.csv')

    # Find best model index / define model to evaluate.
    model_idx = find_bestmodel(stats)

    # Instantiate and load best model.
    m = data_setup['m']
    n = data_setup['n']
    p = data_setup['p']
    lt = data_setup['lt']
    l = data_setup['l']
    tmax = data_setup['tmax']
    mean_x0 = torch.tensor([data_setup['mean_r'], data_setup['mean_theta']]).to(device)
    covar_x0 = data_setup['sigma'] * torch.eye(n, device=device)
    covar_noise = data_setup['var_noise'] * torch.eye(m*N, device=device)

    A = None
    B = torch.tensor([[0.0], [1.0]])
    C = None
    D = torch.zeros(m,p)
    
    model_lt = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A, B, C, D).to(device)
    model_lt.load_state_dict( torch.load(model_path + 'epoch{:03d}.pt'.format(model_idx),
                                      map_location=device, weights_only=True) )
    model_lt.eval()
    mean_lt, covar_lt = model_lt(dT, tmax, mean_U, mean_dU)
    
    print('Learnt Model Parameters were:')
    print('A = ', model_lt.A)
    print('B = ', model_lt.B)
    print('C = ', model_lt.C.weight)
    print('D = ', model_lt.D)


    # Instantiate ground truth model.
    A = torch.tensor([[-l, 0.0], [0.0, 0.0]])
    B = torch.tensor([[0.0], [1.0]])
    C = torch.eye(m)
    D = torch.zeros(m,p)

    model_gt = SEGP.SEGP(m, n, p, lt, mean_x0, covar_x0, A, B, C, D).to(device)
    model_gt.eval()
    mean_gt, covar_gt = model_gt(dT, tmax, mean_U, mean_dU)
    
    # Re-scale learnt mean and covariance.
    rs_mean_lt = scale_mean(mean_lt, 1/scale, dim)
    rs_covar_lt = scale_covar(covar_lt, m, 1/scale, dim)

    # print loss for ground truth.
    mll = MLL()
    loss_gt = -mll(dZn.flatten(start_dim=0, end_dim=1), mean_gt, covar_gt, covar_noise)
    loss_lt = -mll(dZn.flatten(start_dim=0, end_dim=1), rs_mean_lt, rs_covar_lt, covar_noise)
    print('ground truth loss is ', loss_gt)
    print('learnt loss is ', loss_lt)

    # Plot training curves.
    plot_loss(plots_path, stats)

    # Plot prior mean and uncertainty of trained SEGP against 3 random trajectories from batch.
    plot_prior(plots_path, dT, dZ, rs_mean_lt, rs_covar_lt, mean_gt, covar_gt)

    # Plot prior covariance matrices.
    plot_covar(plots_path, covar_gt, rs_covar_lt)

    # Pick random trajectory and return posterior.
    Ts2 = data_setup['Ts']
    dT2, mean_post, covar_post, rand_M, rand_Q = get_posterior(T, Ts2, mean_U, model_lt, dT, tmax, mean_dU, covar_noise, dZn_theta)

    # Re-scale posterior mean and covariance.
    rs_mean_post = scale_mean(mean_post, 1/scale, dim)
    rs_covar_post = scale_covar(covar_post, m, 1/scale, dim)
    
    # Plot learnt posterior versus learnt prior against the relevant ground truth data.
    plot_posterior_versus_prior(plots_path, dT, dT2, dZ, rs_mean_lt, rs_covar_lt, rs_mean_post, rs_covar_post, rand_M, rand_Q) 

    # Plot posterior mean and uncertainty of SEGP against samples from posterior.
    n_samples = 10
    sample_posterior(plots_path, n_samples, dT2, rs_mean_post, rs_covar_post, rand_M, rand_Q)

    return 0



if __name__ == '__main__':
    main()





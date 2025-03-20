import torch
import numpy as np
import math
import pickle
import os
import sys
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import gpytorch

root = ''
utils_path = root + 'Code/Utils'

if utils_path in sys.path:
  print('directory already in path!')
else:
  sys.path.append(utils_path)

from Utils import plot_data, plot_latents, plot_individual_dims



def sample_SE_input(seed, M, Q, tmax, mean_U, lt, T):

    K = len(T)

    # Define GP.
    mean = gpytorch.means.ZeroMean()
    covar = gpytorch.kernels.RBFKernel()
    covar.lengthscale = torch.tensor(lt)
    mean.eval()
    covar.eval()
    U_GP = gpytorch.distributions.MultivariateNormal( mean(T), covar(T) )

    # Sample GP.
    U = torch.zeros(M, K)
    for i in range(M):
      U[i] = U_GP.sample() # shape = (K)
      seed = seed + 1
      torch.manual_seed(seed) # Update seed for new sample.

    U = U + mean_U  # add non-zero mean, shape = (M, K).

    return U



def sample_init(seed, Q, mean_r, mean_theta, sigma):

    # Generate Q random samples from normal distributions.
    rng = np.random.default_rng(seed)
    r = mean_r + sigma * rng.standard_normal(Q) # shape = (Q)
    theta = mean_theta + sigma * rng.standard_normal(Q) # shape = (Q)

    z0 = np.array([r, theta]).T # shape = (Q, n)

    return z0



def solve_particle(T, U, Z0, l):

    M, K = U.shape
    Q, n = Z0.shape

    z = np.zeros( (M, Q, K, n) )

    def particle(z, t, l):
      r, theta = z
      u_t = u_func(t)  # Get input value at time t.
      dzdt = [-l * r, u_t]
      return dzdt

    # Solve ODE.
    for i in range(M):
        u_func = interp1d(T, U[i], kind='linear', fill_value="extrapolate") # Interpolation function for input.
        for j in range(Q):
            z[i, j] = odeint(particle, Z0[j], T, args=(l,) ) # shape = (K, n).

    return z



def transform_coords(z):

    # Radius and angle.
    r = z[:,:,:,0] # shape = (M, Q, K).
    theta = z[:,:,:,1] # shape = (M, Q, K).

    # Convert to Cartesian coordinates.
    x = r * np.cos(theta) # shape = (M, Q, K).
    y = r * np.sin(theta) # shape = (M, Q, K).

    z_transformed = np.array([x, y]) # shape = (n, M, Q, K).
    z_transformed = z_transformed.transpose( (1,2,3,0) ) # shape = (M, Q, K, n).

    return z_transformed



def discretize_signals(Ts, T, mean_U, U, Z):

    dT = T[torch.remainder(torch.arange(T.shape[0]), Ts) == 0] # shape = (N).
    dU = U[:, torch.remainder(torch.arange(U.shape[1]), Ts) == 0] # shape = (M, N).
    dZ = Z[:, :, torch.remainder(torch.arange(Z.shape[2]), Ts) == 0, :] # shape = (M, Q, N, n).
    mean_dU = mean_U[torch.remainder(torch.arange(mean_U.shape[0]), Ts) == 0] # shape = (N).

    return dT, mean_dU, dU, dZ



def add_noise(seed, var_noise, dZ):
    return dZ + np.sqrt(var_noise) * torch.randn( dZ.size() ) # shape = (M, Q, N, n).



def traj_2_vid(r:int, d:int, dZ:torch.tensor):
    """
    Convert discretised trajectories to video.
    args:
                r: radius of ball in pixels.
                d: number of horizontal/vertical pixels in a frame.
              dZ: (M, Q, N, n) tensor of discretised latent states (noisy or noise-free).
    returns:
        vid_batch: (M, Q, N, d, d) tensor of videos.
    """

    M, Q, N, n = dZ.shape

    Z_pix = (d/5) * dZ + 0.5 * d # shape = (M, Q, N, n)

    rr = r*r # radius squared.


    def pixelate_frame(xy):
        """
        takes a single x,y pixel point and converts to binary image
        with ball centered at x,y.
        """

        x = xy[0]
        y = xy[1]

        sq_x = (torch.arange(d) - x)**2
        sq_y = (torch.arange(d) - y)**2

        sq = sq_x.reshape(1,-1) + sq_y.reshape(-1,1)

        image = 1 * (sq < rr)

        return image


    def pixelate_series(XY):
        """
        takes a single 2d trajectory and converts to a video of binary images
        with ball centered at x,y in each frame.
        """

        vid = map(pixelate_frame, XY)
        vid = [v for v in vid]

        return torch.stack(vid, dim=0) # shape = (N, d, d)


    vid_batch = []
    for i in range(M):
        vid_batch.append( torch.stack(
            [pixelate_series(traj_i) for traj_i in Z_pix[i]], dim=0
                        ) ) # shape = (Q, N, d, d)

    vid_batch = torch.stack(vid_batch, dim=0) # (M, Q, N, d, d)

    return vid_batch



def main():

    # dataset directory.
    dataset_number = 1
    data_path = root + 'Data/Dataset{0}'.format(dataset_number)

    if os.path.isdir(data_path):
        raise Exception('Directory exists!')
    else:
        os.makedirs(data_path)

    # Hardware settings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('Default tensor type is now cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Device in use is: ", device)


    # Set random seeds for reproducibility.
    seed = 40
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Time.
    tmax = 4.
    K = 400
    T = torch.linspace(0, tmax, K) # shape = (K).
    step = T[1] - T[0]

    # Parameters of the dynamics.
    l = 0.6 # Decay rate (higher = faster convergence).
    om = 0.4 * np.pi # Angular velocity (rad/s).
    m = 2; n = 2; p = 1 # dimensions of state space.

    # GP input parameters.
    lt = 1.0 # length scale of squared exponential kernel.
    mean_U = om * T # Shape = (K).

    # Initial condition distribution parameters.
    mean_r = 1.5 # Mean radius.
    mean_theta = 0.0 # Mean angle.
    sigma = 0.5 # Standard deviation.

    # Sample M input signals and Q initial conditions.
    M = 80
    Q = 80
    U = sample_SE_input(seed, M, Q, tmax, mean_U, lt, T) # shape = (M, K).
    Z0 = sample_init(seed, Q, mean_r, mean_theta, sigma) # shape = (Q, n).

    # Convert to nparrays - this will run on cpu.
    T = T.cpu().numpy()
    U = U.cpu().numpy()
    mean_U = mean_U.cpu().numpy()

    # Solve ODE.
    Z = solve_particle(T, U, Z0, l) # shape = (M, Q, K, n).

    #### new lines: Rescaling Z here (to approximately [0, 1] range) corresponds to 
    #### dividing diagonal terms of C matrix. However, this completely changes images.
    # Z[:,:,:,0] = Z[:,:,:,0] / np.max(Z[:,:,:,0])
    # Z[:,:,:,1] = Z[:,:,:,1] / np.max(Z[:,:,:,1])
  
    # Convert to tensors.
    T = torch.from_numpy(T).to(device)
    mean_U = torch.from_numpy(mean_U).to(device)
    U = torch.from_numpy(U).to(device)
    Z = torch.from_numpy(Z).to(device) 

    # Discretize data.
    Ts = 12 # sampling period is Ts * step.
    dT, mean_dU, dU, dZ = discretize_signals(Ts, T, mean_U, U, Z)
    N = len(dT) # number of discrete time points.

    # Add noise to discretised trajectory.
    var_noise = 1e-3
    dZn = add_noise(seed, var_noise, dZ)

    # Transform data to Cartesian co-ords for plotting and generating videos.
    dZn_transformed = transform_coords(dZn.cpu().numpy())
    Z_transformed = transform_coords(Z.cpu().numpy())

    #### new lines: Rescaling here (to approximately [0, 1] range) keeps patterns the same, but
    #### scales all trajectories to be squashed towards the origin (not ideal).
    # Z_transformed[:,:,:,0] = Z_transformed[:,:,:,0] / np.max(Z_transformed[:,:,:,0])
    # Z_transformed[:,:,:,1] = Z_transformed[:,:,:,1] / np.max(Z_transformed[:,:,:,1])
    # dZn_transformed[:,:,:,0] = dZn_transformed[:,:,:,0] / np.max(dZn_transformed[:,:,:,0])
    # dZn_transformed[:,:,:,1] = dZn_transformed[:,:,:,1] / np.max(dZn_transformed[:,:,:,1])
  
    # Convert to tensors.
    dZn_transformed = torch.from_numpy(dZn_transformed).to(device)
    Z_transformed = torch.from_numpy(Z_transformed).to(device)

    # Map trajectories to videos.
    r = 1.5 # radius of ball in pixels.
    d = 40 # number of horizontal/vertical pixels in a frame.
    vid_batch = traj_2_vid(r, d, dZn_transformed) # shape = (M, Q, N, d, d).
    if device.type == 'cuda':
      torch.cuda.empty_cache()

    # Plot video as a single frame and associated trajectory.
    file_name = "Videos"
    nplots = 3
    plot_latents(data_path, file_name, vid_batch, Z_transformed, dT, tmax, nplots)

    # Convert to numpy.
    T = T.cpu().numpy()
    dT = dT.cpu().numpy()
    U = U.cpu().numpy()
    dU = dU.cpu().numpy()
    dZ = dZ.cpu().numpy()
    dZn = dZn.cpu().numpy()
    Z_transformed = Z_transformed.cpu().numpy()
    dZn_transformed = dZn_transformed.cpu().numpy()

    # Plot transformed data (x,y).
    file_name = "particle_examples.png"
    plot_data(data_path, file_name, T, U, Z_transformed, dZn_transformed, var_noise)

    # Plot data in original co-ordinates (r, theta).
    file_name = "radius_theta_evolution.png"
    plot_individual_dims(data_path, file_name, dT, dU, dZ, var_noise)

    # Save data.
    data_setup = {'seed':seed, 'tmax':tmax, 'K':K, 'l':l, 'om':om, 'm':m, 'n':n, 'p':p, 'lt':lt,
                  'mean_r':mean_r, 'mean_theta':mean_theta, 'sigma':sigma, 'M':M, 'Q':Q, 'Ts':Ts,
                  'var_noise':var_noise, 'r':r, 'd':d}

    with open(data_path + '/data_setup.pkl', 'wb') as f:
          pickle.dump(data_setup, f)
          f.close()

    with open(data_path + '/mean_U.npy', 'wb') as f:
          np.save(f, mean_U.cpu().numpy())

    with open(data_path + '/mean_dU.npy', 'wb') as f:
          np.save(f, mean_dU.cpu().numpy())

    with open(data_path + '/T.npy', 'wb') as f:
          np.save(f, T)

    with open(data_path + '/dT.npy', 'wb') as f:
          np.save(f, dT)

    with open(data_path + '/dZ.npy', 'wb') as f:
          np.save(f, dZ)

    with open(data_path + '/dZn.npy', 'wb') as f:
          np.save(f, dZn)

    with open(data_path + '/vid_batch.npy', 'wb') as f:
          np.save(f, vid_batch.cpu().numpy())

    return 0



if __name__ == '__main__':
    main()

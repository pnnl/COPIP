import numpy as np
from numpy.linalg import multi_dot as md
from scipy.linalg import expm
from scipy.linalg import block_diag
import cvxpy as cp


"""
Set of functions used for generating the synthetic experimental data.
"""



def SE_kernel(T:np.array, p:int, lt:float):
    """
    Squared exponential kernel.
    args:
               T: array of time inputs for RVs to be sampled at.
               p: dimension of input.
              lt: GP length scale.
    returns:
           Sigma: Covariance between each element of T with shape (p*len(T), p*len(T)). 
    """
    T = np.reshape(T, (1,len(T)) ) # row vector
    Sigma = np.exp( -0.5/(lt*lt) * (T - T.T)**2)

    Sigma_tiled = np.tile( Sigma, (p,1,1) ) # (p, len(T), len(T) )
    Sigma_block = block_diag(*Sigma_tiled) # (p*len(T), p*len(T))
    
    return Sigma_block



def get_ABCD(name:str):
    """
    A set of example LTI system parameters (A,B,C,D) which satisfy a 
    property specified by name.
    args:
         name: name of example.
    returns:
            A: nparray (n, n) LTI parameter
            B: nparray (n, p) LTI parameter
            C: nparray (m, n) LTI parameter  
            D: nparray (m, p) LTI parameter
    """

    if name == "PE-GPVAE":
        freq = 3.0; damp = 0.3; c = freq**2; d = 2*damp*freq;
        
        # A = np.array([[-1e-2, 0, 1, 0], 
        #               [0, -1e-2, 0, 1],
        #               [-c, 0, -d, 0],
        #               [0, -c, 0, -d]
        #              ])

        A = -np.eye(4) # using for testing.
        
        B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]
                     ])
        
        C = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0]])

        D = np.zeros((2,2))


    # check system is contracting
    n = A.shape[0]
    eps = 1e-3
    P = cp.Variable((n,n), diag=True)
    constraints = [P - eps*np.eye(n) >> 0]
    constraints += [np.transpose(A) @ P + P @ A + eps*np.eye(n) << 0]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve()
    print("The optimal value is", prob.value)
    print("A solution P is")
    print(P.value)

    return A, B, C, D



def LTI(batch:int, A:np.array, B:np.array, C:np.array, D:np.array, time:np.array, lt:float, 
        mean_eta:np.array, cov_eta:np.array, mean_x0:np.array, cov_x0:np.array, 
        seed:int, mean_U:np.array):
    """ 
    Simulates continuous time response of the specified LTI system (A,B,C,D).
    args:
        batch: number of videos.
            A: (n, n) LTI parameter.
            B: (n, p) LTI parameter.
            C: (m, n) LTI parameter.
            D: (m, p) LTI parameter.
         time: array containing tmin, tmax and resolution for integration.
           lt: SE length scale.
     mean_eta: mean of noise measurements.
      cov_eta: covariance of noise measurements.
      mean_x0: mean of initial condition.
       cov_x0: covariance matrix for initial condition.
         seed: rng seed.
       mean_U: (p, len(T)) mean function of GP for input.
    returns:
            U: nparray (batch, len(T), p) input to LTI system.
            X: nparray (batch, len(T), n) states of LTI system.
            Y: nparray (batch, len(T), m) output of LTI system.
      Y_tilde: nparray (batch, len(T), m) noisy output of LTI system.
    """

    tmin = time[0]; tmax = time[1]; step = time[2]
    p = B.shape[1]; n = B.shape[0]; m = C.shape[0]

    T = np.arange(tmin, tmax+step, step) # time steps for numerical integration.
    
    # Sample U(t) = [u1(t), ..., up(t)]' from GP.
    Sigma_U = SE_kernel(T, p, lt)
    rng = np.random.default_rng(seed=seed+2)
    sample_U = rng.multivariate_normal(mean_U, Sigma_U) # (p*len(T),)

    U = np.zeros( (p,len(T)) ) # (p, len(T))
    for i in range(p):
        U[i] = sample_U[i*len(T):(i+1)*len(T)]

    U = np.transpose(U)
    U = np.tile(U, (batch,1,1) ) # (batch, len(T), p)   
    
    X = np.zeros((batch, len(T), n))
    Y = np.zeros((batch, len(T), m))
    Y_tilde = np.zeros((batch, len(T), m))

    # randomly sample initial conditions from N(mean_x0, cov_x0)
    rng = np.random.default_rng(seed=seed+1)
    X[:,0,:] = rng.multivariate_normal(mean_x0, cov_x0, size=batch)

    # simulate
    for i in range(len(T)-1):
        X[:,i+1,:] = X[:,i,:] + step * ( np.matmul( X[:,i,:], A.T) + np.matmul(U[:,i,:], B.T) )
        Y[:,i,:] = np.matmul(X[:,i,:], C.T ) + np.matmul(U[:,i,:], D.T)
        rng = np.random.default_rng(seed=seed+i)
        Y_tilde[:,i,:] = Y[:,i,:] + rng.multivariate_normal(mean_eta, cov_eta, size=batch)

    Y[:,len(T)-1,:] = np.matmul(X[:,len(T)-1,:], C.T) + np.matmul(U[:,len(T)-1,:], D.T)
    rng = np.random.default_rng(seed=seed+len(T)-1)
    Y_tilde[:,len(T)-1,:] = Y[:,len(T)-1,:] + rng.multivariate_normal(mean_eta, cov_eta, size=batch)
    
    return U, X, Y, Y_tilde 



def Make_path_batch(batch:int, time:np.array, Ts:int, cond:str, lt:float, seed:int, 
                    mean_eta:np.array, cov_eta:np.array, mean_x0:np.array, cov_x0:np.array, 
                    p:int, n:int, m:int, mean_U:np.array):
    """
    Specifies and simulates the LTI system and returns the continuous and sampled trajectories.
    args:
        batch: number of videos.
         time: array containing tmin, tmax and resolution for integration.
           Ts: Sampling period is Ts*time[2].
         cond: Example system.
           lt: SE length scale.
         seed: random seed.
     mean_eta: mean of noise measurements.
      cov_eta: covariance of noise measurements.
      mean_x0: mean of initial condition.
       cov_x0: covariance matrix for initial condition.
            p: dimension of input.
            n: dimension of state.
            m: dimension of output.
       mean_U: (p, len(T)) mean function of input.
    returns:
            A: nparray (n,n) LTI system parameter.
            B: nparray (n,p) LTI system parameter.
            C: nparray (m,n) LTI system parameter.
            D: nparray (m,p) LTI system parameter.
            U: nparray (batch, len(T), p) continuous input.
            X: nparray (batch, len(T), n) continuous state.
            Y: nparray (batch, len(T), m) continuous output.
      Y_tilde: nparray (batch, len(T), m) continuous noisy output.
         U_dt: nparray (batch, len(T)/Ts, p) sampled input.
         X_dt: nparray (batch, len(T)/Ts, n) sampled state.
         Y_dt: nparray (batch, len(T)/Ts, m) sampled output.
   Y_tilde_dt: nparray (batch, len(T)/Ts, m) sampled noisy output.
    """
    
    A, B, C, D = get_ABCD(cond)
    U, X, Y, Y_tilde = LTI(batch, A, B, C, D, time, lt, mean_eta, cov_eta, mean_x0, cov_x0, 
                             seed, mean_U)
    # sample
    U_dt = U[:, np.mod(np.arange(U.shape[1]), Ts) == 0, :]
    X_dt = X[:, np.mod(np.arange(X.shape[1]), Ts) == 0, :]
    Y_dt = Y[:, np.mod(np.arange(Y.shape[1]), Ts) == 0, :]
    Y_tilde_dt = Y_tilde[:, np.mod(np.arange(Y_tilde.shape[1]), Ts) == 0, :]

    return A, B, C, D, U, X, Y, Y_tilde, U_dt, X_dt, Y_dt, Y_tilde_dt



def Make_Video_batch(batch:int, time:np.array, Ts:int, p:int, n:int, m:int, cond:str, 
                     px:int, py:int, mean_eta:np.array, cov_eta:np.array, mean_x0:np.array, cov_x0:np.array, 
                     seed:int, lt:float, r:int, mean_U:np.array):
    """
    Constructs videos of trajectories based on the sampled path trajectories.
    params:
            batch: number of videos.
             time: array containing tmin, tmax and resolution for integration.
               Ts: Sampling period is Ts*time[2].
                p: dimension of input.
                n: dimension of state.
                m: dimension of output.
             cond: Example LTI system.
               px: horizontal pixel resolution.
               py: vertical pixel resolution.
         mean_eta: mean of noise measurements.
          cov_eta: covariance of noise measurements.
          mean_x0: mean of initial condition.
           cov_x0: covariance matrix for initial condition.
             seed: random seed.
               lt: SE length scale.
                r: radius of ball in pixels.
           mean_U: (p, len(T)) mean function of input.
    returns:
                A: nparray (n,n) LTI system parameter,.
                B: nparray (n,p) LTI system parameter.
                C: nparray (m,n) LTI system parameter.
                D: nparray (m,p) LTI system parameter.
                U: nparray (batch, len(T), p) sampled input.
                X: nparray (batch, len(T), n) sampled state.
                Y: nparray (batch, len(T), m) sampled output.
          Y_tilde: nparray (batch, len(T), m) sampled noisy output.
             U_dt: nparray (batch, len(T)/Ts, p) sampled input.
             X_dt: nparray (batch, len(T)/Ts, n) sampled state.
             Y_dt: nparray (batch, len(T)/Ts, m) sampled output.
       Y_tilde_dt: nparray (batch, len(T)/Ts, m) sampled noisy output.
        vid_batch: nparray (batch, len(T), py, px) video arrays.
    """
    
    A, B, C, D, U, X, Y, Y_tilde, \
    U_dt, X_dt, Y_dt, Y_tilde_dt  = Make_path_batch(batch, time, Ts, cond, lt, seed, mean_eta,
                                                   cov_eta, mean_x0, cov_x0, p, n, m, mean_U)
    
    # convert trajectories to pixel
    Y_pix = Y_dt.copy()
    Y_pix[:,:,0] = Y_pix[:,:,0] * (px/5) + (0.5*px)
    Y_pix[:,:,1] = Y_pix[:,:,1] * (py/5) + (0.5*py)
    
    rr = r*r

    def pixelate_frame(xy):
        """
        takes a single x,y pixel point and converts to binary image
        with ball centered at x,y.
        """
        x = xy[0]
        y = xy[1]

        sq_x = (np.arange(px) - x)**2
        sq_y = (np.arange(py) - y)**2

        sq = sq_x.reshape(1,-1) + sq_y.reshape(-1,1)

        image = 1*(sq < rr)

        return image

    
    def pixelate_series(XY):
        vid = map(pixelate_frame, XY)
        vid = [v for v in vid]
        return np.asarray(vid)


    vid_batch = [pixelate_series(traj_i) for traj_i in Y_pix]
    vid_batch = np.asarray(vid_batch)

    return A, B, C, D, U, X, Y, Y_tilde, U_dt, X_dt, Y_dt, Y_tilde_dt, vid_batch # (batch, tmax, dim)




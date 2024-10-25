import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as MN



"""
Set of functions for constructing GP with physics enhanced kernel.
"""



def SE_kernel(Ti:torch.tensor, Tl:torch.tensor, p:int, lt:float):
    """
    Squared exponential kernel.
    args:
              Ti: array of time inputs for RVs to be sampled at.
              Tl: array of time inputs for RVs to be sampled at.
               p: dimension of input.
              lt: SE length scale.
    returns:
           Sigma: Covariance matrix between Ti and Tl (len(Ti), len(Tl)).
    """
    Ti = torch.reshape(Ti, (1,len(Ti)) ) # row vector
    Tl = torch.reshape(Tl, (len(Tl),1) ) # column vector
    Sigma = torch.exp( -0.5/(lt*lt) * (Ti - Tl)**2)
    
    return Sigma.t()



def Lambda(t1:torch.tensor, T:torch.tensor, A, B, C):
    """
    Computes Lambda term used in the physics enhanced kernel.
    args:
        t1: time step.
         T: array of time inputs for RVs to be sampled at.
         A: LTI parameter of PE kernel.
         B: LTI parameter of PE kernel.
         C: LTI parameter of PE kernel.
    returns:
         L: Lambda matrix for each element of (t1-T) (len(T), m, p).
    """
    
    diff = torch.reshape(t1-T, (len(T),1,1) )
    L = A*diff # (len(T), n, n)
    L = torch.matrix_exp(L) # (len(T), n, n)
    L = torch.matmul(L, B) # (len(T), n, p)
    L = torch.matmul(C, L) # (len(T), m, p)
    
    return L.double()



def PE_kernel(j:int, q:int, ti:torch.tensor, tl:torch.tensor, step:float, lt:float, 
              cov_x0:torch.tensor, A, B, C, D):
    """
    Computes covariance between outputs y_j(ti) and y_q(tl) using the physics enhanced kernel.
    args:
        j: dimension of y.
        q: dimension of y.
       ti: time step.
       tl: time step.
     step: resolution of integration.
       lt: SE length scale.
   cov_x0: Covariance matrix of initial condition.
        A: LTI parameter of PE kernel.
        B: LTI parameter of PE kernel.
        C: LTI parameter of PE kernel.
        D: LTI parameter of PE kernel.
    returns:
        k_jqil: Covariance between outputs y_j(ti) and y_q(tl).
    """

    p = B.shape[1]
    
    # time steps approximating domain of integration for each integral.
    Ti = torch.arange(start=0, end=ti+step, step=step)
    Tl = torch.arange(start=0, end=tl+step, step=step)

    # term 1
    term1 = torch.matmul( torch.matrix_exp(A*tl).t(), C[q].t() ) # (n, 1)
    term1 = torch.matmul( cov_x0, term1 ) # (n, 1)
    term1 = torch.matmul( torch.matrix_exp(A*ti), term1 ) # (n, 1)
    term1 = torch.matmul( C[j], term1 ) # (1, 1)

    # term 2
    Li = Lambda(ti, Ti, A, B, C) # (len(Ti), m, p)
    Ll = Lambda(tl, Tl, A, B, C) # (len(Tl), m, p)
    Ku = SE_kernel(Ti, Tl, p, lt=lt) # (len(Ti), len(Tl))

    term2 = 0.0
    for i in range(len(Ti)):
        # Tl integral
        temp = torch.reshape(Ku[i], (len(Tl),1))
        temp = temp*Ll[:,q,:] # (len(Tl), p))
        temp = step*torch.sum(temp, dim=0) # (1, p) Tl integral
        temp = torch.reshape(temp, (p,1))
        term2 = term2 + step*torch.matmul(Li[i,j,:], temp) # Ti integral

    if torch.count_nonzero(D) != torch.tensor(0):
        
        # term 3 - Ti integral
        term3 = torch.reshape( Ku[:,-1], (len(Ti),1) ) # (len(Ti),1)
        tiled_Dq = torch.tile( D[q], (len(Ti),1) ) # (len(Ti), p)  
        term3 = term3*tiled_Dq # (len(Ti),p)
        term3 = torch.reshape( term3, (len(Ti), p, 1) ) # (len(Ti), p, 1)
        term3 = torch.matmul( torch.reshape(Li[:,j,:], (len(Ti),1,p) ), term3) # (len(Ti), 1, 1)
        term3 = step*torch.sum(term3, dim=0)[0,0]

        # term 4 - Tl integral
        tiled_Dj = torch.tile( D[j], (len(Tl),1) ) # (len(Tl), p)  
        tiled_Dj = torch.reshape(tiled_Dj, (len(Tl),1,p) ) # (len(Tl), 1, p)
        term4 = torch.reshape(Ku[-1], (len(Tl),1))
        term4 = term4*Ll[:,q,:] # (len(Tl), p))
        term4 = torch.reshape(term4, (len(Tl),p,1) ) # (len(Tl), p, 1)
        term4 = torch.matmul(tiled_Dj, term4) # (len(Tl), 1, 1)
        term4 = step*torch.sum(term4, dim=0)[0,0]

        # term 5
        term5 = torch.matmul( D[j], Ku[-1,-1]*D[q].t() ) # (1, 1)

    
    if torch.count_nonzero(D) != torch.tensor(0):
        k_jqil = term1 + term2 + term3 + term4 + term5
    else:
        k_jqil = term1 + term2

    return k_jqil



def K_jq(j:int, q:int, T1:torch.tensor, T2:torch.tensor, step:float, lt:float, cov_x0:torch.tensor, 
         A, B, C, D):
    """
    Construct K_jq block of K from k_jqil.
    args:
            j: component of y.
            q: component of y.
            T1: array of time inputs.
            T2: array of time inputs.
         step: resolution of integration.
       cov_x0: Covariance matrix of initial condition.
            A: LTI parameter of PE kernel.
            B: LTI parameter of PE kernel.
            C: LTI parameter of PE kernel.
            D: LTI parameter of PE kernel.
     returns:
          Kjq: Covariance matrix between y_j and y_q.
    """

    tmax1 = len(T1)
    tmax2 = len(T2)
    Kjq = torch.zeros((tmax1,tmax2))       

    for i in range(tmax1):
        for l in range(tmax2):
            if tmax1==tmax2 and j==q and i>l : # just compute diagonal and upper triangular elements
                continue
            Kjq[i,l] = PE_kernel(j, q, T1[i], T2[l], step, lt, cov_x0, A, B, C, D)

    if tmax1==tmax2 and j==q: # leverage symmetry of diagonal blocks to reduce comp.
        mask = torch.ones((tmax1,tmax2)) - torch.eye(tmax1)
        Kjq = mask*Kjq + Kjq.t()
    
    return Kjq.double()




def K(m:int, T1:torch.tensor, T2:torch.tensor, step:float, lt:float, cov_x0:torch.tensor, 
      A, B, C, D):
    """
    Construct K_mat from K_jq blocks.
        args:
            m: dimension of y.
           T1: array of time inputs.
           T2: array of time inputs.
         step: resolution of integration.
           lt: SE length scale.
       cov_x0: Covariance matrix of initial condition.
            A: LTI parameter of PE kernel.
            B: LTI parameter of PE kernel.
            C: LTI parameter of PE kernel.
            D: LTI parameter of PE kernel.
     returns:
            K_mat: Covariance matrix of y.
    """
    
    tmax1 = len(T1)
    tmax2 = len(T2)
    K_mat = torch.zeros((m*tmax1,m*tmax2))

    if tmax1==tmax2: # create mask for leveraging symmetry of off-diagonal blocks to reduce comp.
        mask = torch.ones((m*tmax1,m*tmax2))
        one_block = torch.ones((tmax1,tmax2))
        mask2 = one_block

    for j in range(m):
        for q in range(m):
            if tmax1==tmax2 and j>q: # just compute diagonal and upper triangular blocks
                mask2 = torch.block_diag(mask2, one_block)
                continue
            K_mat[j*tmax1:(j+1)*tmax1,q*tmax2:(q+1)*tmax2] = K_jq(j, q, T1, T2, step, lt, cov_x0, A, B, C, D)
            text = 'K_{0}{1} complete!'.format(j,q)
            print(text)

    if tmax1 == tmax2: # leverage symmetry of off-diagonal blocks to reduce comp.
        mask = mask - mask2
        K_mat = mask*K_mat + K_mat.t()

    return K_mat.double()



def K_joint(m:int, T1:torch.tensor, T2:torch.tensor, step:float, lt:float, 
            cov_x0:torch.tensor, A:torch.tensor, B:torch.tensor, C:torch.tensor, D:torch.tensor):
    """
    Construct the joint covariance matrix between Y_star and Y_tilde.
        args:
            m: dimension of y.
           T1: array of test time inputs.
           T2: array of training time inputs.
         step: resolution of integration.
           lt: SE length scale.
       cov_x0: Covariance matrix of initial condition.
            A: LTI parameter of PE kernel.
            B: LTI parameter of PE kernel.
            C: LTI parameter of PE kernel.
            D: LTI parameter of PE kernel.
     returns:
            K_joint_mat: Covariance matrix of Y_star and Y_tilde.
    """
    
    tmax1 = len(T1)
    tmax2 = len(T2)

    K_joint_mat = torch.zeros( ( m*(tmax1+tmax2),m*(tmax1+tmax2) ) )
    for T_star in range(2):
        for T in range(2):
            if T_star==0 and T==0:
                K_joint_mat[0:m*tmax1,0:m*tmax1] = K(m, T1, T1, step, lt, cov_x0, A, B, C, D)
            elif T_star==0 and T==1:
                K_joint_mat[0:m*tmax1,m*tmax1:m*(tmax1+tmax2)] = K(m, T1, T2, step, lt, cov_x0, A, B, C, D)
            elif T_star==1 and T==0:
                K_joint_mat[m*tmax1:m*(tmax1+tmax2),0:m*tmax1] =  K_joint_mat[0:m*tmax1,m*tmax1:m*(tmax1+tmax2)].t()
            else:
                K_joint_mat[m*tmax1:m*(tmax1+tmax2),m*tmax1:m*(tmax1+tmax2)] = K(m, T2, T2, step, lt, cov_x0, A, B, C, D)
            
            text = 'K_joint_{0}{1} complete!'.format(T_star,T)
            print(text)

    return K_joint_mat.double()




def mean_j(j:int, T:torch.tensor, mean_x0:torch.tensor, mean_U:torch.tensor,  mean_U_dt:np.array, 
           time:np.array, A, B, C, D):
    """
    Construct mean_j vector.
        args:
            j: dimension of y.
            T: array of time inputs for measurements to be taken at.
      mean_x0: mean of the initial condition.
       mean_U: mean function for the continuous input signal.
    mean_U_dt: mean function for the sampled input signal.
         time: array containing integration limits and step size (start, stop, step).
            A: LTI parameter of PE kernel.
            B: LTI parameter of PE kernel.
            C: LTI parameter of PE kernel.
            D: LTI parameter of PE kernel.
     returns:
            meanfunc_j: mean function of element y_{j}.
    """

    p = B.shape[1]
    tmax = len(T)
    meanfunc_j = torch.zeros((tmax,))
    
    for i in range(tmax):

        # term 1
        term1 = torch.matmul( torch.matrix_exp(A*T[i]), mean_x0) # (n, 1)
        term1 = torch.matmul( C[j], term1 ) # (1, 1)

        # term 2
        Ti = torch.arange(start=time[0], end=T[i]+time[2], step=time[2])
        L = Lambda(T[i], Ti, A, B, C) # (len(Ti), m, p)
        term2 = torch.reshape(mean_U[:len(Ti)], (len(Ti),p,1) ) # (len(Ti),p,1)
        term2 = torch.matmul( torch.reshape(L[:,j,:], (len(Ti),1,p) ), term2) # (len(Ti), 1, 1)
        term2 = time[2]*torch.sum(term2, dim=0)[0,0]

        # term 3
        term3 = torch.matmul( D[j], mean_U_dt[i]) # mean_U (tmax, p)
        
        meanfunc_j[i] = term1 + term2 + term3

    return meanfunc_j.double()



def mean(m:int, T:torch.tensor, mean_x0:torch.tensor, mean_U:torch.tensor, mean_U_dt:torch.tensor, 
         time:np.array, A, B, C, D):
    """
    Construct mean_func from mean_j vectors.
        args:
            m: dimension of y.
            T: array of time inputs.
      mean_x0: mean of the initial condition.
       mean_U: mean function for the continuous input signal.
    mean_U_dt: mean function for the sampled input signal.
         time: array containing integration limits and step size (start, stop, step).
            A: LTI parameter of PE kernel.
            B: LTI parameter of PE kernel.
            C: LTI parameter of PE kernel.
            D: LTI parameter of PE kernel.
     returns:
            mean_func: Covariance matrix of y.
    """

    tmax = len(T)
    mean_func = torch.zeros((m*tmax,))

    for j in range(m):
        mean_func[j*tmax:(j+1)*tmax] = mean_j(j, T, mean_x0, mean_U, mean_U_dt, time, A, B, C, D)
        text = 'mean_{0} complete!'.format(j)
        print(text)

    return mean_func.double()



def Sigma_comp(cov_eta, m, tmax):
    """
    Constructs Sigma from noise covariance matrix.
    args:
          cov_eta: Covariance of measurement noise at a single time step (m, m).
                m: Dimension of y.
             tmax: Number of time steps.
    returns:
            Sigma: Covariance of measurement noise for all time steps (m*tmax, m*tmax).
    """
    Sigma = torch.zeros((m*tmax,m*tmax))
    for i in range(m):
        Sigma[i*tmax:(i+1)*tmax, i*tmax:(i+1)*tmax] = cov_eta[i,i]*torch.eye(tmax)
        
    return Sigma



def inv(K:torch.tensor, cov_eta:torch.tensor, tmax:int):
    """ Computes the inverse of (K + Sigma) via the Cholesky decomposition.
        i.e. (K+Sigma)^-1 = (L*L^T)^-1 = L^-T * L^-1
        args:
            K: Covariance matrix to be inverted.
      cov_eta: Covariance of measurement noise.
         tmax: Number of sampled time steps.
     returns:
        K_inv: Inverse of K + Sigma.
    """
    
    m = cov_eta.shape[0]
    
    Sigma = Sigma_comp(cov_eta, m, tmax)
    
    # Cholesky will only work for symmetric pos. def. matrices.
    K_inv = torch.linalg.cholesky( K + Sigma ) # L   
    K_inv = torch.inverse(K_inv) # L_inv
    K_inv = torch.matmul( K_inv.t(), K_inv ) # K_inv
    
    return K_inv.double()



def post(K_prior:torch.tensor, mu_prior:torch.tensor, cov_eta:torch.tensor, 
         Y_tilde_train:torch.tensor):    
    """ 
    Computes posterior mean and covariance in a more efficient manner than doing so via the predictive function.
           args:
                K_prior: Prior covariance matrix between Y_train and Y_tilde_train.
               mu_prior: Prior mean of Y_train.
                cov_eta: covariance of measurement noise.
          Y_tilde_train: noisy output training data vector (tmax,m).
        returns:
                mu_post: posterior mean.
                 K_post: posterior covariance matrix.
    """
    
    tmax, m = Y_tilde_train.shape
    
    Y_tilde_stacked = stack_traj(Y_tilde_train)
    
    K_inv = inv(K_prior, cov_eta, tmax)

    mu_post = torch.matmul(K_inv, (Y_tilde_stacked - mu_prior) )
    mu_post = torch.matmul(K_prior, mu_post)
    mu_post = mu_prior + mu_post

    K_post = torch.matmul(K_inv, K_prior)
    K_post = K_prior - torch.matmul(K_prior, K_post)

    return mu_post, K_post



def pred(K_joint_mat:torch.tensor, mu_ob:torch.tensor, mu_unob:torch.tensor,
         cov_eta:torch.tensor, Y_tilde_ob:torch.tensor):    
    """ 
    Compute predictive mean and covariance. 
           args:
          K_joint_mat: Joint covariance matrix between Y_unob and Y_tilde_ob.
                mu_ob: Prior mean of observed Y.
              mu_unob: Prior mean of unobserved Y.
              cov_eta: covariance of measurement noise.
           Y_tilde_ob: noisy unobserved Y (tmax,m).
        returns:
              mu_pred: predictive mean.
               K_pred: predictive covariance matrix.
    """
    
    tmax2, m = Y_tilde_ob.shape
    dim0 = len(mu_unob)
    
    K00 = K_joint_mat[:dim0, :dim0]
    K01 = K_joint_mat[:dim0, dim0:]
    K10 = K_joint_mat[dim0:, :dim0]
    K11 = K_joint_mat[dim0:, dim0:]
    
    Y_tilde_stacked = stack_traj(Y_tilde_ob)
    
    K_inv = inv(K11, cov_eta, tmax2)

    mu_pred = torch.matmul(K_inv, (Y_tilde_stacked - mu_ob) )
    mu_pred = torch.matmul(K01, mu_pred)
    mu_pred = mu_unob + mu_pred

    K_pred = torch.matmul(K_inv, K10)
    K_pred = K00 - torch.matmul(K01, K_pred)

    return mu_pred, K_pred



def stack_traj(Y:torch.tensor):
    """
    Stacks trajectory Y into a vector starting with time series of element 0, 
    followed by time series of element 1, and so on ...
    args:
        Y: output trajectory with shape (tmax, m).
    returns:
        Y2: stacked and ordered array with shape (m*tmax).
    """
    tmax = Y.shape[0]; m = Y.shape[1]
    Y2 = torch.zeros(m*tmax)

    for j in range(m):
        Y2[j*tmax:(j+1)*tmax] = Y[:,j]
    
    return Y2.double()



def test_K(K_mat, tol_sym=1e-6, tol_pos=1e-4):
    """
    Tests if K_mat is symmetric and positive definite.
    args:
          K_mat: Covariance matrix.
        tol_sym: tolerance for  symmetry check.
        tol_pos: tolerance for pos. def. check.
    returns:
        True if both conditions are met.
    """
    dimK = K_mat.shape[0]
    eigs_K = torch.linalg.eigvals(K_mat)
    
    sym_test = torch.allclose( K_mat, K_mat.t(), atol=tol_sym )
    eig_test = torch.all( torch.ge( torch.real(eigs_K), -tol_pos*torch.ones(dimK) ) )
    
    if not sym_test:
        raise Exception("Sorry, matrix is not symmetric!")
    if not eig_test.item():
        raise Exception("Sorry, matrix is not positive definite!")
        
    return True



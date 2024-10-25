import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

import GP_Tools as GPT



"""
Classes for constructing the parametrised stability enhanced GP model.
"""



class diag_plus(nn.Module):
    """
    For mapping an unconstrained square matrix to a diagonal matrix with positive elements.
    """
    def __init__(self):
        super().__init__()
        self.e = 1e-2
    
    def forward(self, Z):
        return torch.diag( torch.abs( Z.diag() ) ) + self.e*torch.eye(Z.shape[0])


        
class L_plus(nn.Module):
    """
    For mapping an unconstrained square matrix to a lower triangular matrix with positive elements
    along the main diagonal.
    """
    def __init__(self):
        super().__init__()
        self.e = 1e-2
    
    def forward(self, Z):
        return Z.tril(-1) + torch.diag( torch.abs( Z.diag() ) ) + self.e*torch.eye(Z.shape[0])



class Skew(nn.Module):
    """
    For mapping an unconstrained square matrix to a skew-symmetric matrix.
    """
    def forward(self, Z):
        return Z.triu(1) - Z.triu(1).transpose(-1, -2)



def Cholesky_inv(K):
    """
    Compute the inverse of K via the Cholesky decomposition. Needed for ELBO.LML, ELBO.Gauss_Cross_Entropy, 
    GP.posterior, ...
    """
    K_inv = torch.linalg.cholesky(K) # L   
    K_inv = torch.inverse(K_inv) # L_inv
    K_inv = torch.matmul( K_inv.t(), K_inv )
    return K_inv



class SEGP(nn.Module):
    """
    Stability enhanced GP model class.
    args:
           m: dimension of output.
           n: dimension of state.
           p: dimension of input.
          lt: SE length scale.
        time: array containing the initial time, end time and integration step for kernel.
     mean_x0: mean of the initial condition.
      mean_U: mean function for the GP input (m*tmax,).
   mean_U_dt: sampled mean function of the GP input.
      cov_x0: covariance matrix for the initial condition.
     cov_eta: covariance matrix for the measurement noise.
        tmax: number of sampled time points.
           B: Parameter of mean and kernel. Learnt if set to None.
           C: Parameter of mean and kernel. Learnt if set to None.
           D: Parameter of mean and kernel. Learnt if set to None.

    functions:
            forward: computes the prior mean (mu) and covariance matrix (K) for the time points (T).
    """

    def __init__(self, m:int, n:int, p:int, lt:float, time:np.array, mean_x0:torch.tensor, mean_U:torch.tensor, 
                 mean_U_dt:torch.tensor, cov_x0:torch.tensor, cov_eta:torch.tensor, tmax:int, B, C, D):
        super().__init__()
        self.m = m
        self.n = n
        self.p = p
        self.lt = lt
        self.time = time
        self.mean_x0 = mean_x0
        self.mean_U = mean_U
        self.mean_U_dt = mean_U_dt
        self.cov_x0 = cov_x0
        self.Sigma = GPT.Sigma_comp(cov_eta, m, tmax)
        self.e = 1e-1

        self.V1 = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.V2 = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.V3 = nn.Linear(n, n, bias=False, dtype=torch.double)
        nn.init.normal_(self.V1.weight)
        nn.init.normal_(self.V2.weight)
        nn.init.normal_(self.V3.weight)

        if B==None:
            self.B_ind = True
            self.B_ = nn.Linear(p, n, bias=False, dtype=torch.double)
            nn.init.normal_(self.B_.weight)
        else:
            self.B_ind = False
            self.B = B

        if C==None:
            self.C_ind = True
            self.C_ = nn.Linear(n, m, bias=False, dtype=torch.double)
            nn.init.normal_(self.C_.weight)
        else:
            self.C_ind = False
            self.C = C
        
        if D==None:
            self.D_ind = True
            self.D_ = nn.Linear(p, m, bias=False, dtype=torch.double)
            nn.init.normal_(self.D_.weight)
        else:
            self.D_ind = False
            self.D = D       
        
        # restricting V1 to being diagonal due to numerical issues
        parametrize.register_parametrization(self.V1, "weight", diag_plus())
        parametrize.register_parametrization(self.V2, "weight", L_plus())
        parametrize.register_parametrization(self.V3, "weight", Skew())


    def forward(self, T:torch.tensor):
        """
        Computes the prior mean and covariance matrix.
        args:
                T: Time points to compute the prior mean and covariance matrix at (tmax,1).
        returns:
                mu: Prior mean. (m*tmax)
                 K: Prior covariance matrix (m*tmax, m*tmax).
             Sigma: Covariance matrix of the measurement noise across the time series (m*tmax, m*tmax).
        """
        
        V1_inv = torch.inverse(self.V1.weight)
        self.P = torch.matmul( self.V1.weight, self.V1.weight.t() )
        self.P_inv = torch.matmul( V1_inv.t(), V1_inv )
        
        self.A = torch.matmul( self.V2.weight, self.V2.weight.t() )
        self.A = -0.5*torch.matmul(self.P_inv,  self.A) - self.e*torch.eye(self.n, dtype=torch.double)
        self.A = self.A + torch.matmul(self.P_inv, self.V3.weight)

        # set to this for debugging problem with K matrix
        # self.A = -torch.eye(self.n, dtype=torch.double)
            
        if self.B_ind == True:
            self.B = self.B_.weight

        if self.C_ind == True:
            self.C = self.C_.weight

        if self.D_ind == True:
            self.D = self.D_.weight

        self.mu = GPT.mean(self.m, T, self.mean_x0, self.mean_U, self.mean_U_dt, self.time, 
                                self.A, self.B, self.C, self.D)
        
        self.K = GPT.K(self.m, T, T, self.time[2], self.lt, self.cov_x0, 
                     self.A, self.B, self.C, self.D)

        return self.mu, self.K



    def posterior(self, dist1:list, dist2:list):
        """
        Returns the variational posterior distribution of the VAE for a batch of data.
        args:
                  dist1: contains the mean (bs*m*tmax) and covariance matrix (bs*m*tmax, bs*m*tmax) of the prior.
                  dist2: contains the variational mean (bs*m*tmax) and covariance matrix (bs*m*tmax, bs*m*tmax) of the likelihood.
        returns:
                mu_post: variational posterior mean (bs*m*tmax).
                 K_post: variational posterior covariance matrix (bs*m*tmax, bs*m*tmax).
        """

        mu, K = dist1
        mu_lhood, var_lhood = dist2
        
        dimK, _ = K.shape

        # remove after debugging code
        GPT.test_K(K)
        GPT.test_K(var_lhood)
        
        inv = Cholesky_inv( K  + var_lhood + self.e*torch.eye(dimK) ) # (bs*m*tmax, bs*m*tmax)

        mu_post = torch.matmul(inv, mu_lhood - mu) # (bs*m*tmax)
        mu_post = mu + torch.matmul( K, mu_post ) # (bs*m*tmax)

        K_post = torch.matmul( inv, K ) # (bs*m*tmax, bs*m*tmax)
        K_post = K - torch.matmul( K, K_post ) # (bs*m*tmax, bs*m*tmax)
        
        return mu_post, K_post



    def sample_posterior(self, dist):
        """
        Sample the Normal dist. with mean mu and diagonal covariance matrix.
        args:
                dist: contains the mean (bs,tmax,m) and variance (bs,tmax,m) across a batch of the multivariate GP. 
        returns:
                samples from dist (bs,tmax,m).
        """

        mu, var = dist
        bs, tmax, _ = mu.shape
        E = torch.randn( size=(bs, tmax, self.m) ) # sample standard normal dist.

        return mu + ( E * var ) # sample dist


    
    def set_ABCD(self):
        """
        Calculates the LTI system parameter values from the initialised variables. Used when loading in trained model to 
        GP notebook.
        """
        
        V1_inv = torch.inverse(self.V1.weight)
        self.P = torch.matmul( self.V1.weight, self.V1.weight.t() )
        self.P_inv = torch.matmul( V1_inv.t(), V1_inv )
        
        self.A = torch.matmul( self.V2.weight, self.V2.weight.t() )
        self.A = -0.5*torch.matmul(self.P_inv,  self.A) - self.e*torch.eye(self.n, dtype=torch.double)
        self.A = self.A + torch.matmul(self.P_inv, self.V3.weight)

        if self.B_ind == True:
            self.B = torch.ones((self.n,self.p), dtype=torch.double) * self.B_.weight

        if self.C_ind == True:
            self.C = torch.ones((self.m,self.n), dtype=torch.double) * self.C_.weight

        if self.D_ind == True:
            self.D = torch.ones((self.m,self.p), dtype=torch.double) * self.D_.weight   

        return 0



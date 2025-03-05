import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize



"""
Classes for SEGP model with squared exponential kernel for the input signal.
"""




class L_plus(nn.Module):
    """
    For mapping an unconstrained square matrix to a lower triangular matrix with positive elements
    along the main diagonal.
    """
    def __init__(self,  n:int):
        super().__init__()
        self.register_buffer('pos', 1e-3 * torch.eye(n))

    def forward(self, Z):
        return Z.tril(-1) + torch.diag( torch.abs( Z.diag() ) ) + self.pos



class L(nn.Module):
    """
    For mapping an unconstrained square matrix to a lower triangular matrix.
    """
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return Z.tril()
    


class Skew(nn.Module):
    """
    For mapping an unconstrained square matrix to a skew-symmetric matrix.
    """
    def forward(self, Z):
        return Z.triu(1) - Z.triu(1).transpose(-1, -2)



class SEGP(nn.Module):
    """
    Stability enhanced GP model class.
    args:
           m: dimension of output.
           n: dimension of state.
           p: dimension of input.
          lt: SE length scale.
     mean_x0: mean of the initial condition.
    covar_x0: covariance matrix for the initial condition.
           A: Parameter of GP. Learnt if set to None.
           B: Parameter of GP. Learnt if set to None.
           C: Parameter of GP. Learnt if set to None.
           D: Parameter of GP. Learnt if set to None.

    functions:
            forward: computes the prior mean and covariance matrix.
    """

    def __init__(self, m:int, n:int, p:int, lt:float, mean_x0:torch.tensor, covar_x0:torch.tensor,
                 A:torch.tensor, B:torch.tensor, C:torch.tensor, D:torch.tensor):
        super().__init__()
        self.m = m
        self.n = n
        self.p = p
        self.lt = lt
        self.register_buffer('mean_x0', mean_x0.unsqueeze(0).t() ) # shape = (n,1).
        self.register_buffer('covar_x0', covar_x0) # shape = (n,n).

        if A==None:
            self.A_ind = True
            self.V1 = nn.Linear(n, n, bias=False)
            self.V2 = nn.Linear(n, n, bias=False)
            self.V3 = nn.Linear(n, n, bias=False)
            nn.init.eye_(self.V1.weight)
            nn.init.normal_(self.V2.weight, std=1e-3)
            nn.init.zeros_(self.V3.weight)

            parametrize.register_parametrization(self.V1, "weight", L_plus(n))
            # parametrize.register_parametrization(self.V2, "weight", L_plus(n)) # contraction.
            parametrize.register_parametrization(self.V2, "weight", L() ) # semi-contraction.
            parametrize.register_parametrization(self.V3, "weight", Skew())

        else:
            self.A_ind = False
            self.A = A

        if B==None:
            self.B_ind = True
            self.B = nn.Linear(p, n, bias=False)
            nn.init.normal_(self.B.weight, std=1e-3)
        else:
            self.B_ind = False
            self.B = B

        if C==None:
            self.C_ind = True
            self.C = nn.Linear(n, m, bias=False)
            nn.init.normal_(self.C.weight, std=1e-3)
        else:
            self.C_ind = False
            self.C = C

        if D==None:
            self.D_ind = True
            self.D = nn.Linear(p, m, bias=False)
            nn.init.normal_(self.D.weight, std=1e-3)
        else:
            self.D_ind = False
            self.D = D



    def get_AP(self):
        """
        Function for computing GP parameters (A,P) from the variables initialised in init().
        """

        self.register_buffer( 'eps', 1e-3 * torch.eye(self.n), persistent=False )
        
        V1_inv = torch.inverse( self.V1.weight + self.eps )

        P_inv = torch.matmul( V1_inv.t(), V1_inv )
        self.P = torch.matmul( self.V1.weight, self.V1.weight.t() )

        self.A = torch.matmul( self.V2.weight, self.V2.weight.t() )
        self.A = -0.5 * torch.matmul(P_inv,  self.A) + torch.matmul(P_inv, self.V3.weight)



    def Greens1(self, t:float, T_int:torch.tensor, A:torch.tensor, B:torch.tensor, C:torch.tensor):
        """
        Computes G1 term for each integration step (T_int) up to the specified discrete time step, t.
        args:
            t: discrete time step.
        T_int: time array sliced with last element equal to t.
            A: Parameter of SEGP.
            B: Parameter of SEGP.
            C: Parameter of SEGP.
        returns:
            G1: G1 matrix for each element of (t-T_int) (N_int, m, p).
        """

        N_int = T_int.shape[0]

        G1 = torch.reshape( t-T_int, (N_int,1,1) ) # shape = (N_int,1,1).
        G1 = torch.matrix_exp( A.repeat(N_int, 1, 1) * G1 ) # shape = (N_int, n, n).
        G1 = torch.matmul(G1, B.repeat(N_int, 1, 1) ) # shape = (N_int, n, p).
        G1 = torch.matmul(C.repeat(N_int, 1, 1) , G1) # shape = (N_int, m, p).

        return G1



    def SE_kernel(self, T1:torch.tensor, T2:torch.tensor):
        """
        Squared exponential (SE) kernel chosen for input signal.
        args:
                 T1: array of time inputs with shape = (N1).
                 T2: array of time inputs with shape = (N2).
        returns:
            covar_U: SE Covariance matrix with shape = (N1, N2).
        """

        N1 = T1.shape[0]
        N2 = T2.shape[0]

        T1 = torch.reshape(T1, (N1,1) )
        T2 = torch.reshape(T2, (N2,1) )

        covar_U = torch.exp(( -0.5 * ( T1 - T2.t() )**2 ) / (self.lt*self.lt) )

        return covar_U



    def mean_module(self, dT:torch.tensor, tmax:float, mean_U:torch.tensor, mean_dU:torch.tensor,
                    A:torch.tensor, B:torch.tensor, C:torch.tensor, D:torch.tensor):
        """
        Computes the mean function of the SEGP.
        args:
           dT: Sampled time points to compute the prior mean and covariance matrix.
         tmax: Maximum time point.
       mean_U: Continuous mean function of U used in integral.
      mean_dU: Discretised mean function of U.
            A: Parameter of SEGP.
            B: Parameter of SEGP.
            C: Parameter of SEGP.
            D: Parameter of SEGP.
        returns:
       mean_y: Mean of the SEGP.
        """


        N = dT.shape[0]
        dT = dT.view(N, 1, 1) # shape = (N, 1, 1)

        T = torch.linspace(0, tmax, mean_U.shape[0] )
        step = T[1] - T[0]

        mean_x0 = self.mean_x0.repeat(N, 1, 1) # shape = (N, n, 1)
        mean_U = mean_U.unsqueeze(2) # shape = (K, p, 1)
        mean_dU = mean_dU.unsqueeze(2) # shape = (N, p, 1)

        term1 = torch.linalg.matrix_exp(A.repeat(N, 1, 1) * dT) # shape = (N, n, n)
        term1 = torch.matmul(term1, mean_x0) # shape = (N, n, 1)
        term1 = torch.matmul(C.repeat(N, 1, 1), term1).squeeze(2) # shape = (N, m)

        term2 = []
        for i in range(N):
          t = dT[i].item() # upper limit of integral
          k = torch.where(T == t)[0].item() # index of contin. time tensor matching t
          term2a = self.Greens1(t, T[:k+1], A, B, C) # shape = ( len(T[:k+1]), m, p)
          term2a = torch.matmul(term2a, mean_U[:k+1]) # shape = (len(T[:k+1]), m, 1)
          term2a = torch.trapezoid(term2a, dx=step, dim=0) # shape = (m, 1)
          term2.append(term2a)
        term2 = torch.stack(term2, dim=0).squeeze(2) # shape = (N, m)

        term3 = torch.matmul(D.repeat(N, 1, 1), mean_dU).squeeze(2) # shape = (N, m)

        mean_y = term1 + term2 + term3

        return mean_y



    def covar_module(self, dT1:torch.tensor, dT2:torch.tensor, tmax1:float, tmax2:float, mean_U1:torch.tensor,
                     mean_U2:torch.tensor, mean_dU1:torch.tensor, mean_dU2:torch.tensor,
                     A:torch.tensor, B:torch.tensor, C:torch.tensor, D:torch.tensor):
      """
        Computes the Covariance matrix of the SEGP.
        args:
              dT1: Set 1 of sampled time points.
              dT2: Set 2 of sampled time points.
            tmax1: Maximum time point for set 1.
            tmax2: Maximum time point for set 2.
          mean_U1: Continuous mean function of U used in integrals corresponding to T1.
          mean_U2: Continuous mean function of U used in integrals corresponding to T2.
         mean_dU1: Discretised mean function of U corresponding to dT1.
         mean_dU2: Discretised mean function of U corresponding to dT2.
                A: Parameter of SEGP.
                B: Parameter of SEGP.
                C: Parameter of SEGP.
                D: Parameter of SEGP.
        returns:
          covar_y: Mean of the SEGP.
      """

      N1 = dT1.shape[0]
      N2 = dT2.shape[0]
      dT1 = dT1.view(N1, 1, 1) # shape = (N1, 1, 1)
      dT2 = dT2.view(N2, 1, 1) # shape = (N2, 1, 1)
      T1 = torch.linspace(0, tmax1, mean_U1.shape[0] )
      T2 = torch.linspace(0, tmax2, mean_U2.shape[0] )
      step1 = T1[1] - T1[0]
      step2 = T2[1] - T2[0]

      covar_U = self.SE_kernel(T1, T2) # shape = ( K1, K2 )

      ################################################ term 1 ################################################

      # matrix exp RHS of inital condition covariance
      term1a = torch.linalg.matrix_exp( A.repeat(N2, 1, 1) * dT2) # shape = (N2, n, n)

      # matrix exp LHS of inital condition covariance
      term1b = torch.linalg.matrix_exp( A.repeat(N1, 1, 1) * dT1).unsqueeze(1) # shape = (N1, 1, n, n)

      term1c = torch.matmul(torch.transpose(term1a, 1, 2), C.t().repeat(N2, 1, 1) ) # shape = (N2, n, m)
      term1c = torch.matmul( self.covar_x0.repeat(N2, 1, 1), term1c) # shape = (N2, n, m)
      term1c = torch.matmul(term1b, term1c) # shape = (N1, N2, m, m)
      term1c = torch.matmul( C.repeat(N1, N2, 1, 1), term1c) # shape = (N1, N2, m, n)

      ############################################## term 1 end ##############################################

      term2 = []
      term3 = []
      term4 = []
      for i in range(N1):
           term2a = []
           term3a = []
           term4a = []
           t1 = dT1[i].item() # upper limit of external integral.
           k1 = torch.where(T1 == t1)[0].item() # index of T matching t1.
           for j in range(N2):
              t2 = dT2[j].item() # upper limit of internal integral.
              k2 = torch.where(T2 == t2)[0].item() # index of T matching t2.

              ############################################# term 2 ############################################

              # RHS Green's matrix for all integration steps up to t'.
              term2b = self.Greens1(t2, T2[:k2+1], A, B, C).unsqueeze(0) # shape = (1, len(T2[:k2+1]), m, p)

              # Ku term for all integration steps up to t and t'.
              term2c = covar_U[:k1+1, :k2+1] # shape = ( len(T1[:k1+1]), len(T2[:k2+1]) )
              term2c = term2c.unsqueeze(2).unsqueeze(3) # shape = ( len(T1[:k1+1]), len(T2[:k2+1]), 1, 1)
              self.register_buffer('rep_I', torch.eye(self.p).repeat(term2c.shape[0], term2c.shape[1], 1, 1), persistent=False ) # shape = ( len(T1[:k1+1]), len(T2[:k2+1]), p, p )
              term2c = term2c * self.rep_I # shape = ( len(T1[:k1+1]), len(T2[:k2+1]), p, p )

              # LHS Green's matrix for all integration steps up to t.
              term2d = self.Greens1(t1, T1[:k1+1], A, B, C).unsqueeze(1) # shape = (len(T1[:k1+1]), 1, m, p)

              # Matrix multiply and sum together all terms above.
              term2e = torch.matmul(term2c, torch.transpose(term2b, 2,3) ) # shape = ( len(T1[:k1+1]), len(T2[:k2+1]), p, m )
              term2e = torch.matmul(term2d, term2e) # shape = ( len(T1[:k1+1]), len(T2[:k2+1]), m, m )
              term2e = torch.trapezoid(term2e, dx=step2, dim=1) # shape = (len(T1[:k1+1]), m, m)
              term2e = torch.trapezoid(term2e, dx=step1, dim=0) # shape = (m, m)
              term2a.append(term2e)

              ############################################# term 3 ############################################

              # Ku term for all integration steps up to t.
              term3b = covar_U[:k1+1, k2] # shape = ( len(T1[:k1+1]) )
              term3b = term3b.unsqueeze(1).unsqueeze(2) # shape = ( len(T1[:k1+1]), 1, 1)
              self.register_buffer('rep_I', torch.eye(self.p).repeat(term3b.shape[0], 1, 1), persistent=False ) # shape = ( len(T1[:k1+1]), p, p )
              term3b = term3b * self.rep_I # shape = ( len(T1[:k1+1]), p, p )

              # Green's matrix for all integration steps up to t.
              term3c = self.Greens1(t1, T1[:k1+1], A, B, C) # shape = (len(T1[:k1+1]), m, p)

              # Matrix multiply and sum together all terms above.
              term3d = torch.matmul(term3b, D.t().repeat(term3b.shape[0], 1, 1)) # shape = (len(T1[:k1+1]), p, m)
              term3d = torch.matmul(term3c, term3d) # shape = (len(T1[:k1+1]), m, m)
              term3d = torch.trapezoid(term3d, dx=step1, dim=0) # shape = (m, m)
              term3a.append(term3d)

              ############################################# term 4 ############################################

              # Ku term for all integration steps up to t'.
              term4b = covar_U[k1, :k2+1] # shape = ( len(T2[:k2+1]) )
              term4b = term4b.unsqueeze(1).unsqueeze(2) # shape = ( len(T2[:k2+1]), 1, 1)
              self.register_buffer('rep_I', torch.eye(self.p).repeat(term4b.shape[0], 1, 1), persistent=False ) # shape = ( len(T2[:k2+1]), p, p )
              term4b = term4b * self.rep_I # shape = ( len(T2[:k2+1]), p, p )

              # Green's matrix for all integration steps up to t'.
              term4c = self.Greens1(t2, T2[:k2+1], A, B, C) # shape = (len(T2[:k2+1]), m, p)

              # Matrix multiply and sum together all terms above.
              term4d = torch.matmul(term4b, torch.transpose(term4c, 1, 2) ) # shape = (len(T2[:k2+1]), p, m)
              term4d = torch.matmul(D.repeat(term4d.shape[0], 1, 1), term4d) # shape = (len(T2[:k2+1]), m, m)
              term4d = torch.trapezoid(term4d, dx=step2, dim=0) # shape = (m, m)
              term4a.append(term4d)


           term2a = torch.stack(term2a, dim=0) # (N2, m, m)
           term2.append(term2a)
           term3a = torch.stack(term3a, dim=0) # (N2, m, m)
           term3.append(term3a)
           term4a = torch.stack(term4a, dim=0) # (N2, m, m)
           term4.append(term4a)

      term2 = torch.stack(term2, dim=0) # (N1, N2, m, m)
      term3 = torch.stack(term3, dim=0) # (N1, N2, m, m)
      term4 = torch.stack(term4, dim=0) # (N1, N2, m, m)

      ############################################### term 5 ################################################

      # Ku term at t, t'.
      term5a = self.SE_kernel(dT1, dT2) # shape = ( N1, N2 )
      term5a = term5a.unsqueeze(2).unsqueeze(3) # shape = ( N1, N2, 1, 1)
      self.register_buffer('rep_I', torch.eye(self.p).repeat(term5a.shape[0], term5a.shape[1], 1, 1), persistent=False ) # shape = ( N1, N2, p, p )
      term5a = term5a * self.rep_I # shape = ( N1, N2, p, p )

      # matrix multiply
      term5b = torch.matmul(term5a, D.t().repeat(N2, 1, 1) ) # shape = (N1, N2, p, m)
      term5b = torch.matmul( D.repeat(N2, 1, 1), term5b) # shape = (N1, N2, m, m)

      ############################################## term 5 end ##############################################
      
      # Add all terms together and reshape into (m*N1, m*N2).
      covar_y = term1c # + term2 + term3 + term4 + term5b # shape = (N1, N2, m, m)
      covar_y = torch.transpose(covar_y, 1, 2) # shape = (N1, m, N2, m)
      covar_y = torch.transpose(covar_y, 2, 3) # shape = (N1, m, m, N2)
      covar_y = torch.transpose(covar_y, 0, 1) # shape = (m, N1, m, N2)
      covar_y = covar_y.reshape(self.m*N1, self.m*N2) # shape = (m*N1, m*N2)
      self.register_buffer('eps', 1e-3 * torch.eye(self.m*N1, self.m*N2), persistent=False)
      covar_y = covar_y + self.eps # shape = (m*N1, m*N2)

      return covar_y



    def posterior(self, dT1:torch.tensor, dT2:torch.tensor, tmax1:float, tmax2:float, mean_U1:torch.tensor, mean_U2:torch.tensor,
                  mean_dU1:torch.tensor, mean_dU2:torch.tensor, covar_noise:torch.tensor, obs:torch.tensor):
        """
        Returns the mean and covariance of the posterior SEGP for each trajectory in the batch.
        args:
                    dT1: Set 1 of sampled time points.
                    dT2: Set 2 of sampled time points.
                  tmax1: Maximum time point for set 1.
                  tmax2: Maximum time point for set 2.
                mean_U1: Continuous mean function of U used in integrals corresponding to T1.
                mean_U2: Continuous mean function of U used in integrals corresponding to T2.
               mean_dU1: Discretised mean function of U corresponding to dT1.
               mean_dU2: Discretised mean function of U corresponding to dT2.
            covar_noise: Covariance matrix for the measurement noise (bs, m*N1, m*N1).
                    obs: Observed data (bs, N1, m).

        returns:
              mean_post: Posterior mean (bs, N2, m).
             covar_post: Posterior covariance matrix (bs, m*N2, m*N2).
        """

        bs, N1, _ = obs.shape
        N2 = len(dT2)

        if self.A_ind == True:
            self.get_AP()
            A = self.A
        else:
            A = self.A

        if self.B_ind == True:
            B = self.B.weight
        else:
            B = self.B

        if self.C_ind == True:
            C = self.C.weight
        else:
            C = self.C

        if self.D_ind == True:
            D = self.D.weight
        else:
            D = self.D


        mean_dT1 = self.mean_module(dT1, tmax1, mean_U1, mean_dU1, A, B, C, D) # shape = (N1, m)
        covar_dT1 = self.covar_module(dT1, dT1, tmax1, tmax1, mean_U1, mean_U1, mean_dU1, mean_dU1, A, B, C, D) # shape = (m*N1, m*N1)

        mean_dT2 = self.mean_module(dT2, tmax2, mean_U2, mean_dU2, A, B, C, D) # shape = (N2, m)
        covar_dT2 = self.covar_module(dT2, dT2, tmax2, tmax2, mean_U2, mean_U2, mean_dU2, mean_dU2, A, B, C, D) # shape = (m*N2, m*N2)
        covar_dT1_dT2 = self.covar_module(dT1, dT2, tmax1, tmax2, mean_U1, mean_U2, mean_dU1, mean_dU2, A, B, C, D) # shape = (m*N1, m*N2)
        covar_dT2_dT1 = covar_dT1_dT2.t() # shape = (m*N2, m*N1)

        L = torch.linalg.cholesky(covar_dT1.repeat(bs,1,1) + covar_noise, upper=False) # shape = (bs, m*N1, m*N1)
    
        self.mean_post = (obs - mean_dT1.repeat(bs, 1, 1)).transpose(1, 2).flatten(start_dim=1, end_dim=2).unsqueeze(2) # shape = (bs, m*N1, 1)
        self.mean_post = torch.cholesky_solve(self.mean_post, L ) # shape = (bs , m*N1, 1)    
        self.mean_post = torch.bmm(covar_dT2_dT1.repeat(bs,1,1), self.mean_post ).squeeze(2).unflatten(dim=1, sizes=(self.m, N2)).transpose(1,2) # shape = (bs, N2, m)
        self.mean_post = mean_dT2.repeat(bs,1,1) + self.mean_post # shape = (bs, N2, m)

        self.covar_post = torch.cholesky_solve( covar_dT1_dT2.repeat(bs,1,1) , L) # shape = (bs, m*N1, m*N2)
        self.covar_post = covar_dT2.repeat(bs,1,1) - torch.bmm( covar_dT2_dT1.repeat(bs,1,1), self.covar_post ) # (bs, m*N2, m*N2)

        return self.mean_post, self.covar_post



    def sample_posterior(self, n_samples=3):
        """
        Draw n_samples of the SEGP posterior distribution (for each item in batch) with mean self.mean_post (bs, N2, m) and covariance matrix self.covar_post (bs, m*N2, m*N2).
        args:
                n_samples: Number of samples to draw.
        returns:
                  samples: Samples from posterior distribution (bs, N2, m, n_samples).
        """

        bs, N2, _ = self.mean_post.shape

        # sample standard normal dist.
        sample_std = torch.randn( size=(bs, N2, self.m, n_samples) ) # shape = (bs, N2, m, n_samples)

        # Transform to samples from SEGP posterior.
        var = torch.diagonal(self.covar_post, dim1=1, dim2=2).unflatten( dim=1, sizes=(self.m, N2) ).transpose(1,2).unsqueeze(3).repeat(1, 1, 1, n_samples) # shape = (bs, N2, m, n_samples)
        mean = self.mean_post.unsqueeze(3).repeat(1, 1, 1, n_samples) # shape = (bs, N2, m, n_samples)
        sample_post = mean + (sample_std * torch.sqrt(var) ) # shape = (bs, N2, m, n_samples)

        return sample_post



    def forward(self, dT:torch.tensor, tmax:float, mean_U:torch.tensor, mean_dU:torch.tensor):
        """
        Computes the SEGP prior mean and covariance matrix.
        args:
                       dT: Set of sampled time points.
                     tmax: Maximum time point.
                   mean_U: Continuous mean function of U used in integrals corresponding to T.
                  mean_dU: Discretised mean function of U corresponding to dT.
        returns:
               mean_prior: Prior SEGP mean. (N, m)
              covar_prior: Prior SEGP covariance matrix (m*N, m*N).
        """
        
        N = len(dT)

        if self.A_ind == True:
            self.get_AP()
            A = self.A
        else:
            A = self.A

        if self.B_ind == True:
            B = self.B.weight
        else:
            B = self.B

        if self.C_ind == True:
            C = self.C.weight
        else:
            C = self.C

        if self.D_ind == True:
            D = self.D.weight
        else:
            D = self.D
        
        self.mean_prior = self.mean_module(dT, tmax, mean_U, mean_dU, A, B, C, D) # shape = (N, m)
        self.covar_prior = self.covar_module(dT, dT, tmax, tmax, mean_U, mean_U, mean_dU, mean_dU, A, B, C, D) # shape = (m*N, m*N)

        return self.mean_prior, self.covar_prior

    

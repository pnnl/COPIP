import torch
import torch.nn as nn
import math



"""
Objective function for training the GP, not the VAE.
"""



class LogMarginalLikelihood(nn.Module):
    """
    Class for computing the negative log marginal likelihood loss function used for training the 
    hyperparameters of a vector valued GP on batches of time series data. The batches are 
    ordered as follows Y_11, ... Y_1m, ..., Y_bs1, ..., Y_bsm where Y_ij denotes the time teries
    of component j in trajectory i.
    args:
        bs: batch size.
      tmax: length of time series.
    """
    
    def __init__(self, bs:int, tmax:int):
        super().__init__()  
        self.bs = bs
        self.tmax = tmax

    
    def Chol_inv(self, K_mat, Sigma):
        """
        Computes inverse of K + Sigma via Cholesky decomposition.
        args: 
              K_mat: Covariance matrix of the vector valued function Y with shape (m*tmax, m*tmax).
              Sigma: Block diagonal covariance matrix of the measurement noise (m*tmax, m*tmax).
        returns:
                inv: Inverse of K_mat + Sigma.
        """
        inv = torch.linalg.cholesky( K_mat + Sigma ) # L   
        inv = torch.inverse(inv) # L_inv
        inv = torch.matmul( inv.t(), inv ) # (K+Sigma)^-1
        return inv.double()

    def forward(self, Y:torch.tensor, model):
        
        tiled_mean = torch.tile(model.mu, (self.bs,))
        Chol_inv_tile = torch.tile( self.Chol_inv(model.K, model.Sigma), (self.bs,1,1) )
        Chol_inv_block = torch.block_diag(*Chol_inv_tile)

        lml = - ( 1/(2*self.bs) ) * torch.matmul( (tiled_mean - Y).t(), torch.matmul(Chol_inv_block, tiled_mean - Y) ) - 0.5*torch.log( torch.linalg.det(model.K + model.Sigma) ) - 0.5*self.tmax*torch.log( torch.tensor(2*math.pi) )     
        
        return -lml 




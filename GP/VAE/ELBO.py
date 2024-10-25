import torch
import torch.nn as nn
import math
import TrainVAE_Tools as TVAE
import SEGP as SEGP


"""
Contains class implementing the Evidence Lower Bound (ELBO) objective function.
"""



class ELBO(nn.Module):
    """
    Class for computing the terms of the ELBO objective function.
    args:
                 bs: batch size.
                  m: dimension of latent state.

    functions:
                LML: log marginal likelihood summed over a batch of data.
                GCE: Gaussian cross entropy summed over a batch of data. 
              Recon: reconstruction error summed over a batch of data.
    
    
    """
    
    def __init__(self, bs:int, m:int):
        super(ELBO, self).__init__()
        self.bs = bs
        self.m = m
        self.log_2pi = torch.log( torch.tensor(2*math.pi) )
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')

        
    def LML(self, dist1:list, dist2:list):
        """
        Log marginal likelihood summed over a batch of data.
        args:
          dist1: list with prior mean (bs*m*tmax) and block diag. cov. matrix (bs*m*tmax, bs*m*tmax) of batch. 
          dist2: list with mean and cov. of variational likelihood in compatible shapes.
        returns:
                lml: log marginal likelihood (scalar)
        """
        mu1, K1 = dist1
        mu2, K2 = dist2

        tmax = int( len(mu1)/(self.bs*self.m) )
        
        lml = torch.matmul( SEGP.Cholesky_inv(K1 + K2) , mu1 - mu2 )
        lml = -( 1/(2*self.bs) )*torch.matmul( (mu1 - mu2).t(), lml ) - 0.5*torch.log( torch.linalg.det(K1 + K2) ) - 0.5*tmax*self.log_2pi   
        return lml
        

    def GCE(self, mu1:torch.tensor, K1:torch.tensor, dist2:list):
        """
        Gaussian Cross Entropy summed over a batch of data.
        args:
               mu1: Mean of variational posterior dist. (bs*m*tmax).
                K1: Covariance matrix of variational posterior dist. (bs*m*tmax, bs*m*tmax).
             dist2: List containing mu2, K2.
                       mu2: Mean of variational likelihood. (bs*m*tmax).
                        K2: Covariance matrix of variational likelihood (bs*m*tmax, bs*m*tmax).
    
        returns:
            cross_entropy (scalar)
        """

        mu2, K2 = dist2
        inv = SEGP.Cholesky_inv(K2) # (bs*m*tmax, bs*m*tmax)
        
        tmax = int( len(mu1)/(self.bs*self.m) )
        dim_mtmax = int( len(mu1)/self.bs )

        term1 = 0.5 * len(mu2) * self.log_2pi

        term2 = 0.0
        for i in range(self.bs-1):
            term2 = term2 + torch.log( torch.det( K2[i*dim_mtmax:(i+1)*dim_mtmax, i*dim_mtmax:(i+1)*dim_mtmax] ) )
        term2 = 0.5 * term2
        
        term3a = torch.matmul(inv, torch.diag(K1)) # (bs*m*tmax)
        term3a = 0.5 * torch.matmul( torch.diag(K1).t(), term3a )

        term3b = torch.matmul(inv, mu2) # (bs*m*tmax)
        term3b = -0.5 * torch.matmul( mu1.t(), term3b)

        term3c = torch.matmul(inv, mu2) # (bs*m*tmax)
        term3c = 0.5 * torch.matmul( mu2.t(), term3c)
    
        return term1 + term2 + term3a + term3b + term3c


    def recon(self, p_theta_logits, vid_batch):
        """
        Computes the reconstruction term summed over a batch of data.
        args:
            p_theta_logits: output of decoder - logits for pixel probability (bs, tmax, py*px).
                 vid_batch: video batch (bs, tmax, py*px).
        returns: binary cross entropy averaged over batch size.
            
        """ 
        return self.BCE(p_theta_logits, vid_batch) / self.bs



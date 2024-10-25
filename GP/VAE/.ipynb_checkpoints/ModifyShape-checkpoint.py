import torch



"""
Class which reshapes data for the different steps in the VAE training script.
"""



class Modify_Shape():
    """
    Class for reshaping data between different steps in the VAE training script.
    args:
        bs: batch size.
         m: dimension of latent state.
    """

    def __init__(self, bs:int, m:int):
        super(Modify_Shape, self).__init__()
        self.bs = bs
        self.m = m
        
    def enc_2_lml(self, mu, var):
        """
        Function for mapping encoder outputs to log marginal likelihood inputs.
        args:
                 mu: mean of variational likelihood (bs*tmax, m).
                var: variance of variational likelihood (bs*tmax, m).
        returns:
                 mu: mean of variational likelihood (bs*m*tmax).
                var: covariance of variational likelihood (bs*m*tmax, bs*m*tmax).
        """  

        bs_tmax, m = mu.shape
        tmax = int(bs_tmax/self.bs)
        
        mu = torch.unflatten( mu, 0, (self.bs, tmax) ) # (bs , tmax, m)
        mu = torch.transpose(mu, 1, 2) # (bs ,m, tmax)
        mu = torch.flatten(mu, start_dim=1, end_dim=2) # (bs, m*tmax)
        mu = torch.flatten(mu, start_dim=0, end_dim=1) # (bs*m*tmax)
        
        var = torch.unflatten( var, 0, (self.bs, tmax) ) # (bs , tmax, m)
        var = torch.transpose(var, 1, 2) # (bs ,m, tmax)
        var = torch.flatten(var, start_dim=1, end_dim=2) # (bs, m*tmax)
        var = torch.flatten(var, start_dim=0, end_dim=1) # (bs*m*tmax)
        var = torch.diag(var) # (bs*m*tmax, bs*m*tmax)
        
        return [mu, var]



    def prior_2_lml(self, mu, K):
        """
        Function for mapping prior outputs to log marginal likelihood inputs or inputs for computing posterior.
        args:
                  mu: prior mean of latent state (m*tmax).
                   K: prior covariance matrix of latent state (m*tmax, m*tmax).
        returns:
            mu_tiled: prior mean of latent state repeated over batch (bs*m*tmax).
             K_block: prior cov. matrix of latent state in block diag. form (bs*m*tmax, bs*m*tmax).
        """ 
        mu_tiled = torch.tile(mu, (self.bs,)) # (bs*m*tmax)
        K_tiled = torch.tile( K, (self.bs,1,1) ) # (bs, m*tmax, m*tmax)
        K_block = torch.block_diag(*K_tiled) # (bs*m*tmax, bs*m*tmax)

        return [mu_tiled, K_block]


    def post_2_dec(self, mu, K):
        """
         Function for mapping posterior outputs to form required for computing samples of posterior.
         args:
             mu: posterior mean of latent state repeated over batch (bs*m*tmax).
              K: posterior cov. matrix of latent state in block diag. form (bs*m*tmax, bs*m*tmax).
        returns:
             mu: posterior mean of latent state repeated over batch (bs, tmax, m).
            var: variance of latent state repeated over batch (bs, tmax, m).
        """

        tmax = int( len(mu) / (self.bs*self.m) ) # (bs*m*tmax)
        
        var = torch.sqrt( torch.diag(K) ) # (bs*m*tmax)
        var = torch.unflatten( var, 0, (self.bs, self.m, tmax) ) # (bs, m, tmax)
        var = torch.transpose(var, 1, 2) # (bs, tmax, m)
        
        mu = torch.unflatten( mu, 0, (self.bs, self.m, tmax) ) # (bs, m, tmax)
        mu = torch.transpose(mu, 1, 2) # (bs, tmax, m)

        return [mu, var]



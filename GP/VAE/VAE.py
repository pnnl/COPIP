import torch.nn as nn
import torch



"""
Contains encoder and decoder classes for the VAE.
"""



class VAEEncoder(nn.Module):
    '''
    Encoder. Maps frames (bs, tmax, py, px) to mean (bs*tmax, m) and variance (bs*tmax, m) of latent state.
    args:
          input_dim: flattened dimension of image (py * px).
         hidden_dim: dimension of hidden layer.
         output_dim: dimension of latent state (m).
    '''
    
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim, dtype=torch.double),     
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, 2*output_dim, dtype=torch.double)
                                    )

        for i in self.encoder.modules():
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, mean=0, std=0.1)
                nn.init.constant_(i.bias, val=0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=2, end_dim=3) # (bs, tmax, py*px)
        x = torch.flatten(x, start_dim=0, end_dim=1) # (bs*tmax, py*px)
        out = self.encoder(x) # (bs*tmax, 2*output_dim)
        mu = out[:, :self.output_dim] # (bs*tmax, output_dim)
        var = torch.exp( out[:, self.output_dim:] ) # encoder returns log(var) to avoid numerical stability issues.
        return mu, var



class VAEDecoder(nn.Module):
    '''
    Decoder. Returns probabilities logits of Bernoulli dist. for each pixel in frame.
    Logits rather than probabilities due to numerical stability.
    args:
          input_dim: dimension of latent state (m).
         hidden_dim: diemension of hidden layer.
         output_dim: flattened dimension of image (py * px).

    forward: maps tensor of latent states (bs, tmax, m) to tensor of logits (bs, tmax, py*px).
    '''
    
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim, dtype=torch.double),     
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim, dtype=torch.double)
                                    )

        for i in self.decoder.modules():
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, mean=0, std=0.1)
                nn.init.constant_(i.bias, val=0)


    def forward(self, x):
        return self.decoder(x)




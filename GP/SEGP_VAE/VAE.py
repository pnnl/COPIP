import torch
import torch.nn as nn



class VAEEncoder(nn.Module):
    '''
    Encoder. Maps frames (bs, N, d, d) to mean (bs, N, m) and variance (bs, N, m) of likelihood.
    self.encoder returns log(var) to avoid numerical stability issues.
    args:
          input_dim: flattened dimension of image (d * d).
         hidden_dim: dimension of hidden layer.
         output_dim: dimension of latent state (m).
    '''

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, 2*output_dim)
                                    )

        for i in self.encoder.modules():
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, mean=0, std=1e-2)
                nn.init.constant_(i.bias, val=0)


    def forward(self, x):
        x = torch.flatten(x, start_dim=2, end_dim=3) # shape = (bs, N, d*d)
        out = self.encoder(x) # shape = (bs, N, 2*m)

        # constant term sets min(var) to ensure covariance matrix is positive def.
        mean = out[:, :, :self.output_dim] # shape = (bs, N, m)
        var = torch.exp( out[:, :, self.output_dim:] ) + 1e-3 # shape = (bs, N, m)

        return mean, var



class VAEDecoder(nn.Module):
    '''
    Decoder. Maps latent states (bs, N, m) to logits of Bernoulli dist. (bs, N, d*d).
    Returns logits rather than probabilities due to format required for nn.BCEWithLogitsLoss().
    args:
          input_dim: dimension of latent state (m).
         hidden_dim: diemension of hidden layer.
         output_dim: flattened dimension of image (d * d).
    '''

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim)
                                    )

        for i in self.decoder.modules():
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, mean=0, std=1e-2)
                nn.init.constant_(i.weight, val=0) # initialise so all pixels are black.
                nn.init.constant_(i.bias, val=0)

    def forward(self, x):
        return self.decoder(x) - 6.0 # translate so all pixels are initialised black.

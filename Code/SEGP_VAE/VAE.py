


"""
Script for defining the VAE encoder and decoder.
"""


import torch
import torch.nn as nn
import torch.nn.init as init


class TransposeLayer(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)  # Transposing dimensions 1 and 2



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

        self.layer1 = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    TransposeLayer(),
                                    nn.BatchNorm1d(hidden_dim),
                                    TransposeLayer(),
                                    nn.ReLU()
                                    )
        
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.var = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()



    def _initialize_weights(self):

        # He initialization for Linear layers; biases as zero.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)



    def forward(self, x):

        x = torch.flatten(x, start_dim=2, end_dim=3) # shape = (bs, N, d*d)
        out = self.layer1(x) # shape = (bs, N, hidden_dim)
        mean = self.mean(out) # shape = (bs, N, m)
        var = torch.exp(self.var(out)/2) # shape = (bs, N, m)
        
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
                                    TransposeLayer(),
                                    nn.BatchNorm1d(hidden_dim),
                                    TransposeLayer(),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim)
                                    )

        self._initialize_weights()



    def _initialize_weights(self):

        # He initialization for Linear layers; biases as zero.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)



    def forward(self, x):
        return self.decoder(x)




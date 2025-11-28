

"""
Script for defining the VAE encoder and decoder, where the encoder is a CNN.
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
    self.var returns log(var) to avoid numerical stability issues.
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

         # Define the CNN component of encoder. Shapes below assume d=40.
        self.CNN = nn.Sequential(
                                 nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2, padding='valid'), # shape = (bs, 5, 18, 18)
                                 nn.ReLU()              
                                )

        self.layer1 = nn.Sequential(
                                    nn.Linear( int(5*18*18), hidden_dim),
                                    TransposeLayer(),
                                    nn.BatchNorm1d(hidden_dim),
                                    TransposeLayer(),
                                    nn.ReLU()
                                    )
        
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.var = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()


    def _initialize_weights(self):
        
        # He initialization for Linear and Conv layers; biases as zero.
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)



    def forward(self, x):
        
        bs, N, d, _ = x.shape

        # Shape input for CNN.
        x = torch.flatten(x, start_dim=0, end_dim=1) # shape = (bs*N, d, d)
        x = x.unsqueeze(dim=1) # shape = (bs * N, 1, d, d)

        # Pass through convolutional layers. Shapes below assume d=40.
        x = self.CNN(x) # shape = (bs * N, 5, 18, 18)
        
        # Shape input for MLP.
        x = torch.flatten(x, start_dim=1, end_dim=3) # shape = (bs * N, 5*18*18)
        x = torch.unflatten(x, dim=0, sizes=(bs,N) ) # shape = (bs, N, 5*18*18)
        
        x = self.layer1(x) # shape = (bs, N, hidden_dim)

        mean = self.mean(x) # shape = (bs, N, m)
        var = torch.exp(self.var(x)/2) # shape = (bs, N, m)

        
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

        for i in self.decoder.modules():
            if isinstance(i, nn.Linear):
                init.kaiming_normal_(i.weight, mode='fan_in', nonlinearity='relu')
                init.zeros_(i.bias)

    def forward(self, x):
        out = self.decoder(x)

        return out

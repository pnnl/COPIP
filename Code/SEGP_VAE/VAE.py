


"""
Script for defining the VAE encoder and decoder.
"""


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


def unwrap_angle(theta:torch.Tensor, dim:int=1) -> torch.Tensor:
    """
    Unwrap angle sequence to avoid discontinuities at ±π.
    
    When the angle jumps from π to -π (or vice versa), this function adds/subtracts 2π
    to make the sequence continuous. This is essential for GP training since GPs assume
    smooth underlying functions.
    
    Args:
        theta: Angle tensor with shape (..., N, ...) where N is the time dimension.
               Angles should be in radians, typically in [-π, π].
        dim: The dimension along which to unwrap (time dimension). Default is 1.
    
    Returns:
        theta_unwrapped: Continuous angle sequence with same shape as input.
    
    Example:
        Input:  [0.0, 1.0, 2.0, 3.0, -3.0, -2.0]  (jump at index 4)
        Output: [0.0, 1.0, 2.0, 3.0,  3.28, 4.28]  (continuous)
    """
    # Compute differences along the time dimension
    diff = torch.diff(theta, dim=dim)
    
    # Find where jumps occur (difference > π or < -π)
    # If diff > π, we jumped from positive to negative, need to subtract 2π from correction
    # If diff < -π, we jumped from negative to positive, need to add 2π to correction
    correction = torch.zeros_like(diff)
    correction = torch.where(diff > math.pi, -2 * math.pi, correction)
    correction = torch.where(diff < -math.pi, 2 * math.pi, correction)
    
    # Cumulative sum of corrections to apply to each time step
    cumulative_correction = torch.cumsum(correction, dim=dim)
    
    # Pad with zeros at the beginning (first time step has no correction)
    pad_shape = list(theta.shape)
    pad_shape[dim] = 1
    zero_pad = torch.zeros(pad_shape, device=theta.device, dtype=theta.dtype)
    cumulative_correction = torch.cat([zero_pad, cumulative_correction], dim=dim)
    
    # Apply correction
    theta_unwrapped = theta + cumulative_correction
    
    return theta_unwrapped


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer that extracts expected (x, y) coordinates from feature maps.
    Given a feature map of shape (bs, C, H, W), outputs expected coordinates (bs, C, 2).
    The coordinates are normalized to [-1, 1] with (0, 0) at the image center.
    """
    
    def __init__(self, height:int, width:int, temperature:float=1.0):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        
        # Create normalized coordinate grids in [-1, 1] with (0,0) at center
        # pos_x: left=-1, right=1
        # pos_y: top=-1, bottom=1
        pos_x = torch.linspace(-1.0, 1.0, width)
        pos_y = torch.linspace(-1.0, 1.0, height)
        
        # Register as buffers (not parameters, but move with model to device)
        self.register_buffer('pos_x', pos_x.view(1, 1, 1, width))  # (1, 1, 1, W)
        self.register_buffer('pos_y', pos_y.view(1, 1, height, 1))  # (1, 1, H, 1)
    
    def forward(self, feature_map:torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (bs, C, H, W) tensor
        Returns:
            coords: (bs, C, 2) tensor of (x, y) coordinates in [-1, 1]
        """
        bs, C, H, W = feature_map.shape
        
        # Flatten spatial dimensions and apply softmax
        feature_flat = feature_map.view(bs, C, -1)  # (bs, C, H*W)
        attention = F.softmax(feature_flat / self.temperature, dim=-1)  # (bs, C, H*W)
        attention = attention.view(bs, C, H, W)  # (bs, C, H, W)
        
        # Compute expected x and y coordinates
        expected_x = (attention * self.pos_x).sum(dim=[2, 3])  # (bs, C)
        expected_y = (attention * self.pos_y).sum(dim=[2, 3])  # (bs, C)
        
        # Stack to get (bs, C, 2) with order (x, y)
        coords = torch.stack([expected_x, expected_y], dim=-1)  # (bs, C, 2)
        
        return coords


class CartesianToPolar(nn.Module):
    """
    Differentiable Cartesian-to-Polar coordinate conversion layer.
    Converts (x, y) in Cartesian coordinates to (r, theta) in polar coordinates.
    
    r = sqrt(x^2 + y^2)
    theta = atan2(y, x)
    
    Note: atan2 is differentiable everywhere except at (0, 0).
    A small epsilon is added for numerical stability.
    """
    
    def __init__(self, eps:float=1e-8):
        super(CartesianToPolar, self).__init__()
        self.eps = eps
    
    def forward(self, cartesian:torch.Tensor) -> torch.Tensor:
        """
        Args:
            cartesian: (..., 2) tensor where last dim is (x, y)
        Returns:
            polar: (..., 2) tensor where last dim is (r, theta)
        """
        x = cartesian[..., 0]
        y = cartesian[..., 1]
        
        # Compute radius with numerical stability
        r = torch.sqrt(x**2 + y**2 + self.eps)
        
        # Compute angle using atan2 (handles all quadrants correctly)
        theta = torch.atan2(y, x)
        
        # Stack to get (..., 2) with order (r, theta)
        polar = torch.stack([r, theta], dim=-1)
        
        return polar



class VAEEncoderCNNGN(nn.Module):
    """
    CNN-based VAE Encoder with Spatial Softmax and Cartesian-to-Polar conversion.
    Same as VAEEncoderCNN but uses GroupNorm instead of BatchNorm2d.

    Notes:
      - GroupNorm does not depend on batch statistics (more stable than BatchNorm for VAEs / variable N).
      - Returns variance (not std). var = exp(logvar) with clamp for numeric stability.
    """

    def __init__(
        self,
        d: int = 40,
        num_keypoints: int = 1,
        temperature: float = 1.0,
        data_scale: float = 2.5,
        gn_groups: int = 8,          # default groups; will be clamped to a valid divisor
        var_max: float = 1e2,        # clamp for variance
    ):
        super().__init__()
        self.d = d
        self.num_keypoints = num_keypoints
        self.data_scale = data_scale
        self.var_max = var_max

        # --- helper to pick valid group count (must divide channels) ---
        def _valid_groups(channels: int, desired: int) -> int:
            g = min(desired, channels)
            while channels % g != 0 and g > 1:
                g -= 1
            return g

        g1 = _valid_groups(32, gn_groups)
        g2 = _valid_groups(64, gn_groups)
        # last conv outputs num_keypoints channels; GN there is optional, but can help
        g3 = _valid_groups(num_keypoints, min(gn_groups, num_keypoints))

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=g1, num_channels=32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=g2, num_channels=64),
            nn.ReLU(),

            nn.Conv2d(64, num_keypoints, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=g3, num_channels=num_keypoints) if num_keypoints > 1 else nn.Identity(),
        )

        self.feature_height = d // 4
        self.feature_width = d // 4

        self.spatial_softmax = SpatialSoftmax(
            height=self.feature_height,
            width=self.feature_width,
            temperature=temperature
        )

        self.cart_to_polar = CartesianToPolar()

        # Separate scale/bias for r and theta (no mixing)
        self.r_scale = nn.Parameter(torch.ones(1))
        self.r_bias = nn.Parameter(torch.zeros(1))
        self.theta_scale = nn.Parameter(torch.ones(1))
        self.theta_bias = nn.Parameter(torch.zeros(1))

        # log-variance params (per-dimension, global)
        self.logvar_r = nn.Parameter(torch.zeros(1))
        self.logvar_theta = nn.Parameter(torch.zeros(1))

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

            if isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    init.ones_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: (bs, N, d, d)
        Returns:
            mean: (bs, N, 2) where last dim is (r, theta)
            var:  (bs, N, 2) variance (positive, clamped)
        """
        bs, N, d, _ = x.shape
        x = x.view(bs * N, 1, d, d)

        features = self.cnn(x)  # (bs*N, num_keypoints, H', W')

        coords_normalized = self.spatial_softmax(features)  # (bs*N, num_keypoints, 2) in [-1, 1]
        coords_scaled = coords_normalized * self.data_scale  # (bs*N, num_keypoints, 2)
        coords_scaled = coords_scaled.mean(dim=1)            # (bs*N, 2)

        polar_coords = self.cart_to_polar(coords_scaled)     # (bs*N, 2)
        r = polar_coords[..., 0]
        theta = polar_coords[..., 1]

        mean_r = self.r_scale * r + self.r_bias
        mean_theta = self.theta_scale * theta + self.theta_bias
        mean = torch.stack([mean_r, mean_theta], dim=-1)     # (bs*N, 2)

        # return variance (not std)
        var_r = torch.exp(self.logvar_r).clamp(min=1e-6, max=self.var_max).expand(bs * N)
        var_theta = torch.exp(self.logvar_theta).clamp(min=1e-6, max=self.var_max).expand(bs * N)
        var = torch.stack([var_r, var_theta], dim=-1)        # (bs*N, 2)

        mean = mean.view(bs, N, 2)
        var = var.view(bs, N, 2)
        return mean, var



class VAEEncoderCNN(nn.Module):
    """
    CNN-based VAE Encoder with Spatial Softmax and Cartesian-to-Polar conversion.
    
    Pipeline:
    1. CNN backbone extracts feature maps from input images (using stride conv instead of pooling)
    2. Spatial Softmax extracts expected (x, y) coordinates in [-1, 1]
    3. Coordinates are scaled to match the data generation scale (based on d/5 scaling in DataGeneration_Particle.py)
    4. Cartesian-to-Polar layer converts (x, y) to (r, theta)
    5. Outputs mean and variance for the polar coordinates (preserving geometric structure)
    
    Args:
        d: Image dimension (height = width = d), default 40
        num_keypoints: Number of spatial keypoints to extract (default 1 for single object tracking)
        temperature: Spatial softmax temperature (lower = sharper attention)
        data_scale: Scale factor to convert normalized coords to physical coords (default 2.5 based on d/5 * 0.5)
    
    Note:
        - Uses stride=2 convolutions instead of MaxPooling to preserve translation equivariance
        - Does not mix r and theta through linear layers to preserve polar coordinate structure
    """
    
    def __init__(self, d:int=40, num_keypoints:int=1, temperature:float=1.0, data_scale:float=2.5):
        super(VAEEncoderCNN, self).__init__()
        self.d = d
        self.num_keypoints = num_keypoints
        self.data_scale = data_scale  # From DataGeneration: Z_pix = (d/5) * dZ + 0.5 * d
        # When normalized to [-1,1]: Z_normalized = (Z_pix - 0.5*d) / (0.5*d) = 2*Z_pix/d - 1
        # So dZ = Z_normalized * (0.5*d) / (d/5) = Z_normalized * 2.5
        
        # CNN backbone using stride convolutions instead of MaxPooling
        # This preserves translation equivariance for accurate coordinate extraction
        # Input: (bs*N, 1, d, d)
        self.cnn = nn.Sequential(
            # Layer 1: d x d -> d/2 x d/2 (stride=2 for downsampling)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2: d/2 x d/2 -> d/4 x d/4 (stride=2 for downsampling)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: d/4 x d/4 -> d/4 x d/4 (stride=1, keep resolution)
            nn.Conv2d(64, num_keypoints, kernel_size=3, stride=1, padding=1),
            # Output: (bs*N, num_keypoints, d/4, d/4)
        )
        
        # Spatial dimensions after CNN
        self.feature_height = d // 4
        self.feature_width = d // 4
        
        # Spatial Softmax to extract (x, y) coordinates
        self.spatial_softmax = SpatialSoftmax(
            height=self.feature_height,
            width=self.feature_width,
            temperature=temperature
        )
        
        # Cartesian to Polar conversion
        self.cart_to_polar = CartesianToPolar()
        
        # Learnable scale and bias for r and theta SEPARATELY (no mixing!)
        # This preserves the geometric meaning of polar coordinates
        # r: radius, should be non-negative
        # theta: angle in radians
        self.r_scale = nn.Parameter(torch.ones(1))
        self.r_bias = nn.Parameter(torch.zeros(1))
        self.theta_scale = nn.Parameter(torch.ones(1))
        self.theta_bias = nn.Parameter(torch.zeros(1))
        
        # Learnable log-variance parameters (separate for r and theta)
        self.logvar_r = nn.Parameter(torch.zeros(1))
        self.logvar_theta = nn.Parameter(torch.zeros(1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # He initialization for Conv2d layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                init.ones_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x:torch.Tensor):
        """
        Args:
            x: Input images with shape (bs, N, d, d)
        Returns:
            mean: Mean of polar coordinates (bs, N, 2) where dim 2 is (r, theta)
            var: Variance of polar coordinates (bs, N, 2)
        """
        bs, N, d, _ = x.shape
        
        # Reshape for CNN: (bs, N, d, d) -> (bs*N, 1, d, d)
        x = x.view(bs * N, 1, d, d)
        
        # CNN forward pass
        features = self.cnn(x)  # (bs*N, num_keypoints, H', W')
        
        # Spatial Softmax: extract (x, y) coordinates in [-1, 1]
        coords_normalized = self.spatial_softmax(features)  # (bs*N, num_keypoints, 2)
        
        # Scale coordinates to physical space
        # From DataGeneration_Particle.py: Z_pix = (d/5) * dZ + 0.5 * d
        # Our normalized coords are in [-1, 1], mapping to physical coords:
        # dZ = coords_normalized * 2.5
        coords_scaled = coords_normalized * self.data_scale  # (bs*N, num_keypoints, 2)
        
        # Average over keypoints if multiple (for single keypoint, this is identity)
        coords_scaled = coords_scaled.mean(dim=1)  # (bs*N, 2)
        
        # Convert to polar coordinates
        polar_coords = self.cart_to_polar(coords_scaled)  # (bs*N, 2) -> (r, theta)
        
        r = polar_coords[..., 0]      # (bs*N,)
        theta = polar_coords[..., 1]  # (bs*N,)
        
        # Apply separate scale and bias to r and theta (preserving geometric structure)
        mean_r = self.r_scale * r + self.r_bias          # (bs*N,)
        mean_theta = self.theta_scale * theta + self.theta_bias  # (bs*N,)
        
        # Stack to form mean: (bs*N, 2)
        mean = torch.stack([mean_r, mean_theta], dim=-1)
        
        # Variance: learnable but separate for r and theta
        var_r = torch.exp(self.logvar_r / 2).expand(bs * N)      # (bs*N,)
        var_theta = torch.exp(self.logvar_theta / 2).expand(bs * N)  # (bs*N,)
        var = torch.stack([var_r, var_theta], dim=-1)  # (bs*N, 2)
        
        # Reshape back: (bs*N, 2) -> (bs, N, 2)
        mean = mean.view(bs, N, 2)
        var = var.view(bs, N, 2)
        
        return mean, var


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




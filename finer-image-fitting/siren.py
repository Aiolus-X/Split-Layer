import torch
from torch import nn
from math import sqrt
import numpy as np

class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out
    

class Finer(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers-1):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output  

class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0,
                                use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)

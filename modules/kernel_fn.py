import pywt
import numpy as np
import torch
import torch.nn as nn
from modules.basic_blocks import MLP
from modules.utils import normalize, Swish

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.numpy as jnp
from jax.config import config
import neural_tangents as nt
from neural_tangents import stax

class NeuralKernel(nn.Module):
    def __init__(self, in_channels, enc_channels, out_channels=None, n_layers=2, normalize=False, kernel_type='ntk', activation='erf'):
        super(NeuralKernel, self).__init__()

        assert kernel_type in ['ntk', 'nngp']
        act_dict = {'erf': stax.Erf(), 
                    'relu': stax.Relu(), 
                    'sin': stax.Sin(),
                    'gelu': stax.Gelu(),
                    'sigmoid': stax.Sigmoid_like()}
        if out_channels is None:
            out_channels = in_channels
        layers = []
        for i in range(n_layers):
            layers.append(stax.Dense(enc_channels))
            layers.append(act_dict[activation])
        layers.append(stax.Dense(out_channels))
        self.kernel_fn = stax.serial(*layers)[-1]
        self.n_layers = n_layers
        self.normalize = normalize
        self.feature_depth = in_channels
        self.kernel_type = kernel_type
    
    def forward(self, x, y=None):
        if y is None:
            y = x

        x = self.get_feature(x)
        y = self.get_feature(y)

        K = self._get_kernel(x, y)
        return K
    
    # A wacky solution to compute closed-form NTK using neural-tangents 
    # Not differentiable for the moment (translate to pure pytorch in the future)
    def _get_kernel(self, x, y):
        # convert into jax array
        x_jnp = jnp.asarray(x.detach().cpu().numpy().astype(np.float32))
        y_jnp = jnp.asarray(y.detach().cpu().numpy().astype(np.float32))

        K = self.kernel_fn(x_jnp, y_jnp, self.kernel_type)
        # convert back to pytorch
        K = torch.tensor(np.copy(K), device=x.device)

        return K
        
    def get_feature(self, x):
        if self.normalize:
            return normalize(x)
        return x
    
    def diag(self, x, batch_size=5000):
        n = x.size(0)
        batch_size = min(n, batch_size)
        diag = []
        for i in range(n // batch_size + np.sign(n % batch_size)):
            x_batch = x[i * batch_size: (i + 1) * batch_size]
            diag.append(self._get_kernel(x_batch, x_batch).diag())
        return torch.cat(diag, dim=0)
    
class MixRBFKernel(nn.Module):
    def __init__(self, in_channels, sigma=[1, np.sqrt(2), 2, 2 * np.sqrt(2), 4], normalize=False):
        super(MixRBFKernel, self).__init__()
        self.feature_depth = in_channels
        self.sigma = sigma
        self.normalize = normalize

    def forward(self, x, y=None):
        if y is None:
            y = x
        bx = x.size(0)
        by = y.size(0)

        x = self.get_feature(x)
        y = self.get_feature(y)

        KXY = mix_rbf_kernel(x.view(bx, -1), y.view(by, -1), self.sigma)
        return KXY
    
    def diag(self, x):
        b = x.size(0)
        return torch.ones(b,).to(x.device)

    def get_feature(self, x):
        if self.normalize:
            return normalize(x)
        return x

class ExpKernel(nn.Module):
    def __init__(self, in_channels, tau=1., normalize=False):
        super(ExpKernel, self).__init__()
        self.feature_depth = in_channels
        self.tau = 1.
        self.normalize = normalize

    def forward(self, x, y=None):
        if y is None:
            y = x
        bx = x.size(0)
        by = y.size(0)

        x = self.get_feature(x)
        y = self.get_feature(y)

        KXY = exp_kernel(x.view(bx, -1), y.view(by, -1), self.tau)
        return KXY
    
    def diag(self, x):
        b = x.size(0)
        return torch.exp(x.view(b, -1).pow(2).sum(-1) / self.tau)

    def get_feature(self, x):
        if self.normalize:
            return normalize(x)
        return x
    
class PolynomialKernel(nn.Module):
    def __init__(self, in_channels, deg=2, c=1, normalize=False):
        super(PolynomialKernel, self).__init__()
        self.deg = deg
        self.c = c
        self.feature_depth = in_channels
        self.normalize = normalize

    def forward(self, x, y=None):
        if y is None:
            y = x
        bx = x.size(0)
        by = y.size(0)

        x = self.get_feature(x)
        y = self.get_feature(y)

        KXY = polynomial_kernel(x.view(bx, -1), y.view(by, -1), deg=self.deg, c=self.c)
        return KXY

    def get_feature(self, x):
        if self.normalize:
            return normalize(x)
        return x

class LinearKernel(nn.Module):
    def __init__(self, in_channels, normalize=False):
        super(LinearKernel, self).__init__()
        self.feature_depth = in_channels
        self.normalize = normalize

    def forward(self, x, y=None):
        if y is None:
            y = x
        bx = x.size(0)
        by = y.size(0)

        x = self.get_feature(x)
        y = self.get_feature(y)

        KXY = polynomial_kernel(x.view(bx, -1), y.view(by, -1), deg=1, c=0)
        return KXY

    def get_feature(self, x):
        if self.normalize:
            return normalize(x)
        return x

class ArccosKernel(nn.Module):
    def __init__(self, in_channels, layers=1, deg=0, normalize=False):
        super(ArccosKernel, self).__init__()
        self.layers = layers
        self.deg = deg
        self.feature_depth = in_channels
        self.normalize = normalize

    def forward(self, x, y=None):
        if y is None:
            y = x

        bx = x.size(0)
        by = y.size(0)

        x = self.get_feature(x)
        y = self.get_feature(y)
        
        KXY = arccos_kernel(x.view(bx, -1), y.view(by, -1), self.layers, self.deg, self.normalize)
        return KXY
    
    def diag(self, x):
        b = x.size(0)
        x = self.get_feature(x)
        if self.deg == 0:
            return torch.ones(b,).to(x.device)
        elif self.deg == 1:
            return x.pow(2).view(b, -1).sum(-1)
    
    def get_feature(self, x):
        if self.normalize:
            return normalize(x)
        return x

def mix_rbf_kernel(X, Y, sigma_list=[1, np.sqrt(2), 2, 2 * np.sqrt(2), 4]):
    m = X.size(0)

    XYT = torch.mm(X, Y.t())
    X_norm_sqr = (X * X).sum(-1).unsqueeze(1)
    Y_norm_sqr = (Y * Y).sum(-1).unsqueeze(1)
    exponent = X_norm_sqr - 2 * XYT + Y_norm_sqr.t()

    K = 0.
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)
    K /= len(sigma_list)

    return K

def mix_rbf_kernel_full(X, Y, sigma_list=[1, np.sqrt(2), 2, 2 * np.sqrt(2), 4]):
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)
    K /= len(sigma_list)
    
    return K[:m, :m], K[:m, m:], K[m:, m:]

def dot_prod_kernel(X, Y):
    return polynomial_kernel(X, Y, deg=1, c=0)

def polynomial_kernel(X, Y, deg=1, c=1):
    XYT = torch.mm(X, Y.t())
    K = torch.pow(XYT + c, deg)
    return K

def polynomial_kernel_full(X, Y, scale=1., deg=1, c=1):
    m = X.size(0)
    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    K = torch.pow(scale * ZZT + c, deg)
    return K[:m, :m], K[:m, m:], K[m:, m:]

def cosine_sim_kernel(X, Y):
    Z = torch.cat((X, Y), dim=0)
    ZZT = torch.mm(Z, Z.t())
    Z_norm_sqrt = torch.diag(ZZT).unsqueeze(1).sqrt().expand_as(ZZT)
    K = ZZT / (Z_norm_sqrt * Z_norm_sqrt.t())
    return K

def exp_kernel(X, Y, tau=1.):
    XYT = torch.mm(X, Y.t())
    K = torch.exp(XYT / tau)
    return K

def arccos_kernel(X, Y, layers=1, deg=1, normalized=False):
    assert deg >= 0 and deg <= 2

    if deg == 0:
        J = lambda theta: 1. - theta / np.pi
    elif deg == 1:
        J = lambda theta: (torch.sin(theta) + (np.pi - theta) * torch.cos(theta)) / np.pi
    else:
        J = lambda theta: (3. * torch.sin(theta) * torch.cos(theta)
              + (np.pi - theta) * (1 + 2. * (torch.cos(theta).pow(2)))) / np.pi

    return _arccos_kernel_recursive(X, Y, layers, deg, J, normalized=normalized)

def _arccos_kernel_recursive(X, Y, l, deg, J, normalized=False):
    if l == 1:
        if normalized:
            nxny = 1.
        else:
            nxny = torch.norm(X, p=2, dim=-1).unsqueeze(1) * torch.norm(Y, p=2, dim=-1).unsqueeze(0)
        xty = torch.mm(X, Y.t())
    elif l > 1:
        if normalized:
            nxny = 1.
        else:
            nxny = torch.sqrt(_arccos_kernel_recursive(X, X, l - 1, deg, J).diag().unsqueeze(1)
                            * _arccos_kernel_recursive(Y, Y, l - 1, deg, J).diag().unsqueeze(0))
        xty = _arccos_kernel_recursive(X, Y, l - 1, deg, J)

    a = torch.acos(torch.clamp(xty / nxny, min=-1. + 1e-7, max=1. - 1e-7))
    K = (nxny ** deg) * J(a)
    return K

def get_kernel(kernel_config, input_dim):
    k_type = kernel_config['type']
    if k_type == 'ntk':
        return NeuralKernel(input_dim, 10000, n_layers=kernel_config['n_layers'], activation=kernel_config['activation'])
    elif k_type == 'rbf':
        return MixRBFKernel(input_dim, sigma=kernel_config['sigmas'])
    else:
        raise RuntimeError("No such kernel: {:s}".format(k_type))
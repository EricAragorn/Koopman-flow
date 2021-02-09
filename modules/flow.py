import torch
import torch.nn as nn
import numpy as np
from .basic_blocks import InvertibleLinear, InvertibleResNet1d, GlowRevNet1d, GlowRevNet2d, MLP
from .density_modeling_utils import GaussianDiag

import os
from tqdm import tqdm

import pickle

class VAE1d(nn.Module):
    def __init__(self, input_dim, layers=3, latent_dim=64, interm_dim=1024):
        super(VAE1d, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP((input_dim, *([interm_dim] * layers), latent_dim * 2), batchnorm=False)
        self.decoder = MLP((latent_dim, *([interm_dim] * layers), input_dim), batchnorm=False)
        
        self.loggamma_x = nn.Parameter(torch.zeros(1,), requires_grad=True)
    
    def _split_dist_params(self, dist_params):
        m, logs = torch.split(dist_params, dist_params.size(1) // 2, dim=-1)
        return m, logs

    def _reparameterize(self, m, logs):
        return m + torch.exp(logs) * torch.randn_like(m).to(m.device)
    
    def _get_kld_loss(self, m, logs):
        kld_loss = 0.5 * (m.pow(2) + torch.exp(2 * logs) - 2 * logs - 1.).sum(-1)
        return kld_loss
    
    def forward(self, x):
        dist_params = self.encode(x)
        m, logs = self._split_dist_params(dist_params)
        latents = self._reparameterize(m, logs)
        rec = self.decode(latents)

        kld_loss = self._get_kld_loss(m, logs)

        return rec, kld_loss
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def sample(self, sample_size, device=torch.device('cuda')):
        prior_samples = torch.randn(sample_size, self.latent_dim).to(device)
        samples = self.decode(prior_samples)
        return samples

class Glow1d(nn.Module):
    def __init__(self, input_depth, interm_channels, steps):
        super(Glow1d, self).__init__()
        self.input_depth = input_depth
        self.steps = steps
        self.operators = nn.ModuleList([GlowRevNet1d(input_depth, interm_channels) for _ in range(steps)])

        self.prior = GaussianDiag(input_depth, 0., 0., trainable=False)
    
    def forward(self, x):
        b = x.size(0)

        logpk = 0.
            
        Fx = x.view(b, -1)
        for i in range(self.steps):
            Fx, logdet = self.operators[i](Fx, ignore_logdet=False)
            logpk = logpk + logdet

        prior_logp = self.prior.logp(Fx)
        logpk = logpk + prior_logp
        dim = float(np.prod(Fx.shape[1:]))
        logpk = logpk / dim

        return logpk
    
    def sample(self, n_samples, temp=1.):
        with torch.no_grad():
            sample = self.prior.sample(n_samples, temp)
            x = sample
            for i in reversed(range(self.steps)):
                x = self.operators[i].inverse(x)

        return x

class Glow2dStage(nn.Module):
    def __init__(self, in_channels, steps):
        super(Glow2dStage, self).__init__()
        self.ops = nn.ModuleList([GlowRevNet2d(in_channels, do_actnorm=False) for _ in range(steps)])
    
    def forward(self, x):
        Fx = x
        logdet = 0.
        for op in self.ops:
            Fx, _logdet = op(Fx)
            logdet = logdet + _logdet
        return Fx, logdet

    def inverse(self, Fx):
        x = Fx
        for op in reversed(self.ops):
            x = op.inverse(x)
        return x

class Glow2d(nn.Module):
    def __init__(self, input_shape, stages, steps):
        super(Glow2d, self).__init__()
        c, h, w = input_shape

        self.init_squeeze_factor = 2
        c *= (self.init_squeeze_factor ** 2)
        h = h // self.init_squeeze_factor
        w = w // self.init_squeeze_factor

        _stages = []
        for i in range(stages):
            _stages.append(Glow2dStage(c, steps[i]))
            c *= 4
            h = h // 2
            w = w // 2
        self.stages = nn.ModuleList(_stages)
        self.final_shape = (c, h, w)

        self.prior = GaussianDiag(np.prod(self.final_shape), 0., 0., trainable=False)
    
    @staticmethod
    def _squeeze(tensor, factor=2):
        b, c, h, w = tensor.shape
        tensor = (tensor.view(b, c, h // factor, factor, w // factor, factor)
                        .permute(0, 1, 3, 5, 2, 4)
                        .reshape(b, c * (factor ** 2), h // factor, w // factor))
        return tensor

    @staticmethod
    def _unsqueeze(tensor, factor=2):
        b, c, h, w = tensor.shape
        tensor = (tensor.view(b, c // (factor ** 2), factor, factor, h, w)
                        .permute(0, 1, 4, 2, 5, 3)
                        .reshape(b, c // (factor ** 2), h * factor, w * factor))
        return tensor
    
    def forward(self, x):
        b = x.size(0)

        logpk = 0.
        Fx = self._squeeze(x, self.init_squeeze_factor)
        for s in self.stages:
            Fx, logdet = s(Fx)
            Fx = self._squeeze(Fx, 2)
            logpk = logpk + logdet

        prior_logp = self.prior.logp(Fx.view(b, -1))
        logpk = logpk + prior_logp
        dim = float(np.prod(Fx.shape[1:]))
        logpk = logpk / dim

        return logpk
    
    def sample(self, n_samples, temp=1.):
        with torch.no_grad():
            sample = self.prior.sample(n_samples, temp).view(n_samples, *self.final_shape)
            x = sample
            for s in reversed(self.stages):
                x = self._unsqueeze(x, 2)
                x = s.inverse(x)
            x = self._unsqueeze(x, self.init_squeeze_factor)

        return x

class InvResNet1d(nn.Module):
    def __init__(self, input_depth, interm_channels, steps):
        super(InvResNet1d, self).__init__()
        self.input_depth = input_depth
        self.steps = steps
        self.operators = nn.ModuleList([InvertibleResNet1d(input_depth, interm_channels) for _ in range(steps)])

        self.prior = GaussianDiag(input_depth, 0., 0., trainable=False)
    
    def forward(self, x):
        b = x.size(0)

        logpk = 0.
            
        K1 = x.view(b, -1)
        K1.requires_grad = True
        for i in range(self.steps):
            K1, logdet = self.operators[i](K1, ignore_logdet=False)
            logpk = logpk + logdet

        prior_logp = self.prior.logp(K1)
        logpk = logpk + prior_logp
        dim = float(np.prod(K1.shape[1:]))
        logpk = logpk / dim

        return logpk
    
    def sample(self, n_samples, temp=1.):
        with torch.no_grad():
            sample = self.prior.sample(n_samples, temp)
            K0 = sample
            for i in reversed(range(self.steps)):
                K0 = self.operators[i].inverse(K0)

        return K0

        





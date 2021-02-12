import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import numpy as np
from .kernel_fn import *
from .basic_blocks import *
from .utils import normalize, VisualizeLayer, NegPad2d, Swish
from .loss import mix_rbf_mmd

import matplotlib.pyplot as plt

class BaseAE(nn.Module):
    def __init__(self, input_dim, encoding_depth, style='celeba64', use_sn=False, sn_decoder_only=False):
        """Base class for AE-like structures

        Args:
            input_dim (int): input dimension
            encoding_depth ([type]): latent code dimension
            style (str, optional): WAE implementation of autoencoder for vision datasets. https://arxiv.org/pdf/1711.01558.pdf 
                                   Available options are ['mnist', 'cifar10', 'celeba64']. Defaults to 'celeba64'.
            use_sn (bool, optional): use spectral normalization as regularizer. Defaults to False.
            sn_decoder_only (bool, optional): apply spectral normalization on decoder only if use_sn is true. Defaults to False.
        """
        super(BaseAE, self).__init__()
        in_channels = input_dim[0]
        
        self.groups = 1
        self.encoding_depth = encoding_depth

        if use_sn:
            def _get_conv(*args, **kwargs):
                return nn.Conv2d(*args, **kwargs)

            def _get_sn_conv(*args, **kwargs):
                return spectral_norm(nn.Conv2d(*args, **kwargs))
            
            def _get_sn_linear(*args, **kwargs):
                return spectral_norm(nn.Linear(*args, **kwargs))

            def _get_sn_conv_transpose(*args, **kwargs):
                return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

            if not sn_decoder_only:
                Conv = _get_sn_conv
            else:
                Conv = _get_conv
            Lin = _get_sn_linear
            ConvTranspose = _get_sn_conv_transpose
        else:
            def _get_conv(*args, **kwargs):
                return nn.Conv2d(*args, **kwargs)
            
            def _get_linear(*args, **kwargs):
                return nn.Linear(*args, **kwargs)

            def _get_conv_transpose(*args, **kwargs):
                return nn.ConvTranspose2d(*args, **kwargs)

            Conv = _get_conv
            Lin = _get_linear
            ConvTranspose = _get_conv_transpose
        
        if style == 'celeba64':
            self.mid_shape = (1024, 4, 4)
            self.encoder = nn.Sequential(
                Conv(in_channels, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                Conv(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                Conv(256, 512, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                Conv(512, 1024, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                ConvTranspose(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                ConvTranspose(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ConvTranspose(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                ConvTranspose(128, in_channels, kernel_size=5, stride=1, padding=2),
                nn.Tanh()
            )
        elif style in ['mnist', 'cifar10']:
            self.mid_shape = (1024, 2, 2)
            self.encoder = nn.Sequential(
                Conv(in_channels, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                Conv(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                Conv(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                Conv(512, 1024, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                ConvTranspose(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                ConvTranspose(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ConvTranspose(256, in_channels, kernel_size=4, stride=1, padding=1),
                NegPad2d((0, 1, 0, 1)),
                nn.Tanh()
            )
            
        self.latent_fn = None
        self.mid_dim = np.prod(self.mid_shape)

        self._get_encoding_mlp = lambda encoding_depth: Lin(self.mid_dim, encoding_depth)
        self._get_decoding_mlp = lambda decoding_depth: Lin(decoding_depth, 1024 * 8 * 8)

        self.loggamma_x = nn.Parameter(torch.zeros(1,), requires_grad=True)

        for m in self.modules():
            if m in (nn.Conv2d, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if m in (nn.Linear,):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents
    
    def encode(self, x):
        b = x.size(0)
        x = self.encoder(x)
        latent = self.encoding_mlp(x.view(b, -1))
        if self.latent_fn is not None:
            latent = self.latent_fn(latent)

        return [latent]
    
    def decode(self, features):
        b = features[0].size(0)
        x = self.decoding_mlp(features[0])
        output = self.decoder(x.reshape(b, 1024, 8, 8))
        return output
    
    # x should be a size 4 batch of samples
    def bilinear_interpolate(self, latents, steps):
        assert latents[0].size(0) >= 4
        latents = [l[:4] for l in latents]

        interpolated_latent = []
        for l in latents:
            interp_l_list = []
            for alpha in np.linspace(0., 1., num=steps):
                l_t = l[0] * (1 - alpha) + l[1] * alpha
                l_b = l[2] * (1 - alpha) + l[3] * alpha
                for beta in np.linspace(0., 1., num=steps):
                    interp_l = l_t * (1 - beta) + l_b * beta
                    interp_l_list.append(interp_l)
            interpolated_latent.append(torch.stack(interp_l_list, dim=0))

        if self.latent_fn is not None:
            # Note that for non identiy latent fn, this is an extrinsic mean
            interpolated_latent = [self.latent_fn(l) for l in interpolated_latent]
        ret = self.decode(interpolated_latent)
        return ret

class VanillaAE(BaseAE):

    def __init__(self, input_dim, encoding_depth, style='celeba64', use_sn=False, sn_decoder_only=False):
        """Implementation of vanilla autoencoder (see base class for argument specs)

        Args:
            input_dim (int): input dimension
            encoding_depth ([type]): latent code dimension
        """
        super(VanillaAE, self).__init__(input_dim=input_dim,
                                  encoding_depth=encoding_depth,
                                  style=style, 
                                  use_sn=use_sn,
                                  sn_decoder_only=sn_decoder_only)
        self.encoding_mlp = self._get_encoding_mlp(encoding_depth)
        self.decoding_mlp = self._get_decoding_mlp(encoding_depth)

    
class SAE(VanillaAE):
    def __init__(self, input_dim, encoding_depth, style='celeba64', use_sn=False, sn_decoder_only=False):
        """Implementation of sperical autoencoder (see base class for argument specs)

        Args:
            input_dim (int): input dimension
            encoding_depth ([type]): latent code dimension
        """
        super(SAE, self).__init__(input_dim=input_dim, 
                                  encoding_depth=encoding_depth, 
                                  style=style, 
                                  use_sn=use_sn,
                                  sn_decoder_only=sn_decoder_only)
        self.latent_fn = normalize


class VAE(BaseAE):
    def __init__(self, input_dim, encoding_depth, style='celeba64', use_sn=False, sn_decoder_only=False):
        """Implementation of vanilla VAE (see base class for argument specs)

        Args:
            input_dim (int): input dimension
            encoding_depth ([type]): latent code dimension (after reparameterization)
        """
        super(VAE, self).__init__(input_dim=input_dim, 
                                  encoding_depth=encoding_depth, 
                                  style=style, 
                                  use_sn=use_sn,
                                  sn_decoder_only=sn_decoder_only)
        self.encoding_mlp = self._get_encoding_mlp(encoding_depth * 2)
        self.decoding_mlp = self._get_decoding_mlp(encoding_depth)
    
    def _split_dist_params(self, dist_params):
        mus = []
        log_sigmas = []
        for p in dist_params:
            m, s = torch.split(p, p.size(1) // 2, dim=-1)
            mus.append(m)
            log_sigmas.append(5 * s.tanh())
        return mus, log_sigmas

    def _reparameterize(self, mus, log_sigmas):
        return [m + torch.exp(s) * torch.randn_like(m).to(m.device) for m, s in zip(mus, log_sigmas)]
    
    def _get_kld_loss(self, mus, log_sigmas):
        kld_loss = 0.
        for m, s in zip(mus, log_sigmas):
            kld_loss += 0.5 * (m.pow(2) + torch.exp(2 * s) - 2 * s - 1).sum(-1)
        return kld_loss
    
    def forward(self, x):
        latents, mus, log_sigmas = self.get_latents(x)
        kld_loss = self._get_kld_loss(mus, log_sigmas)

        reconstruction = self.decode(latents)
        return reconstruction, latents, kld_loss
    
    def get_latents(self, x):
        dist_params = self.encode(x)

        mus, log_sigmas = self._split_dist_params(dist_params)
        latents = self._reparameterize(mus, log_sigmas)
        if self.latent_fn is not None:
            latents = [self.latent_fn(l) for l in latents]
        return latents, mus, log_sigmas
    
    def sample(self, sample_size, device=torch.device('cuda')):
        sampled_latent = [torch.randn(sample_size, self.encoding_depth).to(device) for i in range(self.groups)]
        return self.decode(sampled_latent), sampled_latent

class TwoStageVAE(nn.Module):
    def __init__(self, base_vae, layers=3, latent_dim=64, interm_dim=1024):
        """Implementation of Two-Stage VAE. https://arxiv.org/pdf/1903.05789.pdf

        Args:
            base_vae ([type]): the (trained) first stage VAE
            layers (int, optional): # layers for encoder/decoder of the second stage VAE. Defaults to 3.
            latent_dim (int, optional): latent code dimension (after reparameterization) for the second stage VAE. Defaults to 64.
            interm_dim (int, optional): intermediate dimension of encoder/decoder. Defaults to 1024.
        """
        super(TwoStageVAE, self).__init__()
        super(TwoStageVAE, self).add_module('base_vae', base_vae)
        self.latent_dim = latent_dim
        self.encoding_layers = nn.ModuleList([MLP((base_vae.encoding_depth, *([interm_dim] * layers)), batchnorm=False) for i in range(base_vae.groups)])
        self.decoding_layers = nn.ModuleList([MLP((latent_dim, *([interm_dim] * layers)), batchnorm=False) for i in range(base_vae.groups)])

        self.encoding_output = nn.ModuleList([nn.Linear(base_vae.encoding_depth + interm_dim, latent_dim * 2) for i in range(base_vae.groups)])
        self.decoding_output = nn.ModuleList([nn.Linear(latent_dim + interm_dim, base_vae.encoding_depth) for i in range(base_vae.groups)])
        self.loggamma_x = nn.Parameter(torch.zeros(1,), requires_grad=True)
    
    def _split_dist_params(self, dist_params):
        mus = []
        log_sigmas = []
        for p in dist_params:
            m, s = torch.split(p, p.size(1) // 2, dim=-1)
            mus.append(m)
            log_sigmas.append(5 * s.tanh())
        return mus, log_sigmas

    def _reparameterize(self, mus, log_sigmas):
        return [m + torch.exp(s) * torch.randn_like(m).to(m.device) for m, s in zip(mus, log_sigmas)]
    
    def _get_kld_loss(self, mus, log_sigmas):
        kld_loss = 0.
        for m, s in zip(mus, log_sigmas):
            kld_loss += 0.5 * (m.pow(2) + torch.exp(2 * s) - 2 * s - 1.).sum(-1)
        return kld_loss
    
    def forward(self, x):
        base_latents = [l.detach() for l in self.base_vae.get_latents(x)[0]] # Stop gradients
        second_dist_params = self.encode(base_latents)
        second_mus, second_log_sigmas = self._split_dist_params(second_dist_params)
        second_latents = self._reparameterize(second_mus, second_log_sigmas)
        rec_latents = self.decode(second_latents)

        kld_loss = self._get_kld_loss(second_mus, second_log_sigmas)

        return base_latents, rec_latents, kld_loss
    
    def encode(self, base_latents):
        second_dist_params = []
        for i, l in enumerate(base_latents):
            e = self.encoding_layers[i](l)
            second_p = self.encoding_output[i](torch.cat((e, l), dim=-1))
            second_dist_params.append(second_p)
        return second_dist_params
    
    def decode(self, second_latents):
        rec_latents = []
        for i, l in enumerate(second_latents):
            e = self.decoding_layers[i](l)
            rec_l = self.decoding_output[i](torch.cat((e, l), dim=-1))
            rec_latents.append(rec_l)
        return rec_latents
    
    def sample(self, sample_size, device=torch.device('cuda')):
        sampled_second_latent = [torch.randn(sample_size, self.latent_dim).to(device) for i in range(self.base_vae.groups)]
        sampled_base_latent = self.decode(sampled_second_latent)
        samples = self.base_vae.decode(sampled_base_latent)
        return samples, sampled_base_latent, sampled_second_latent


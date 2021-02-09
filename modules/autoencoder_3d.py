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

class BaseAE3d(nn.Module):
    def __init__(self, input_dim, n_l, depths, encoding_depth=128, scale_factor=1, groups=None):
        super(BaseAE3d, self).__init__()
        in_channels = input_dim[0]
        input_hwd = input_dim[1:]

        def _enc_block(in_channels, n_l, downscale=2, out_channels=None):
            assert(n_l > 0)
            if out_channels is None:
                out_channels = in_channels * downscale

            layers = [
                ResNetDownscaleBlock3d(in_channels, out_channels, scale=downscale, bottleneck=False),
            ]
            for i in range(n_l - 1):
                layers.extend([
                    ResNetIdentityBlock3d(out_channels, bottleneck=False),
                ])
            return nn.Sequential(*layers)
        
        def _dec_block(in_channels, n_l, upscale=2, out_channels=None):
            assert(n_l > 0)
            if out_channels is None:
                out_channels = in_channels // upscale

            layers = []
            for i in range(n_l - 1):
                layers.extend([
                    ResNetIdentityBlock3d(in_channels, mid_channels=in_channels, bottleneck=False)
                ])
            
            layers.extend([
                ResNetUpscaleBlock3d(in_channels, out_channels, mid_channels=in_channels, scale=upscale, bottleneck=False)
            ])
            return nn.Sequential(*layers)

        self.scale_factor = scale_factor
        self.encoding_depth = encoding_depth
        self.n_l = n_l
        self.groups = len(n_l) if groups is None else groups
        
        self.input_conv = nn.Conv3d(in_channels, depths[0], kernel_size=5, stride=self.scale_factor, padding=2)
        self.input_norm = nn.BatchNorm3d(depths[0])
        self.input_act = Swish()
        self.enc_blocks = nn.ModuleList([_enc_block(depths[i], 
                                                    out_channels=depths[i + 1], 
                                                    n_l=n_l[i]) for i in range(len(n_l))])
        self.dec_blocks = nn.ModuleList([_dec_block(depths[i + 1] * 2 if len(n_l) - i - 1 < self.groups else depths[i + 1], 
                                                    out_channels=depths[i], 
                                                    n_l=n_l[i]) for i in reversed(range(len(n_l)))])

        self.output_upscale = PixelShuffle3d(self.scale_factor)
        self.output_conv = nn.Conv3d(depths[0] // (self.scale_factor ** 3), in_channels, kernel_size=5, padding=2)

        self.depths = depths
        self.latent_fn = None

        hw = [d // self.scale_factor for d in input_hwd]

        self.mid_shape = [(depths[i + 1], *[d // (2 ** (i + 1)) for d in hw]) for i in range(0, len(n_l))]
        self.mid_dim = [np.prod(s) for s in self.mid_shape]
        print(self.mid_shape)

        self.h = nn.Parameter(torch.ones(*self.mid_shape[-1]), requires_grad=True)

        self._get_encoding_mlp = lambda latent_dim: nn.ModuleList([nn.Linear(d, latent_dim) for d in self.mid_dim[-self.groups:]])
        self._get_decoding_mlp = lambda latent_dim: nn.ModuleList([nn.Linear(latent_dim, d) for d in self.mid_dim[-self.groups:]])

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
    
    def encode(self, x):
        b = x.size(0)
        x = self.input_conv(x)
        x = self.input_act(self.input_norm(x))
        latents = []
        for i, enc_b in enumerate(self.enc_blocks):
            x = enc_b(x)
            if len(self.n_l) - i - 1 < self.groups:
                l = self.encoding_mlp[- len(self.n_l) + i](x.view(b, -1))
                if self.latent_fn is not None:
                    l = self.latent_fn(l)
                latents.append(l)
        return latents
    
    def decode(self, features):
        b = features[0].size(0)
        x = 0.
        for i, dec_b in enumerate(self.dec_blocks):
            ind = -(i + 1)
            if i < self.groups:
                z = self.decoding_mlp[ind](features[ind]).view(b, *self.mid_shape[ind])
                if i == 0:
                    x = torch.cat((self.h.unsqueeze(0).expand(b, -1, -1, -1, -1), z), dim=1)
                else:
                    x = torch.cat((x, z), dim=1)
            x = dec_b(x)

        x = self.output_upscale(x)
        x = self.output_conv(x)
        x = torch.tanh(x)
        return x
    
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

class SAE3d(BaseAE3d):
    def __init__(self, input_dim, n_l, depths, encoding_depth, scale_factor=1, groups=None):
        super(SAE3d, self).__init__(input_dim, n_l, depths, encoding_depth, scale_factor, groups)
        self.encoding_mlp = self._get_encoding_mlp(encoding_depth)
        self.decoding_mlp = self._get_decoding_mlp(encoding_depth)
        self.latent_fn = normalize
    
    def forward(self, x):
        latents = self.encode(x)
        latents = [self.latent_fn(l) for l in latents]
        reconstruction = self.decode(latents)
        return reconstruction, latents

class VAE3d(BaseAE3d):
    def __init__(self, input_dim, n_l, depths, encoding_depth, scale_factor=1, groups=None, normalize_latent=False):
        super(VAE3d, self).__init__(input_dim, n_l, depths, encoding_depth, scale_factor, groups)
        self.encoding_mlp = self._get_encoding_mlp(encoding_depth * 2)
        self.decoding_mlp = self._get_decoding_mlp(encoding_depth)
        if normalize_latent:
            self.latent_fn = normalize
    
    def _split_dist_params(self, dist_params):
        mus = []
        log_sigmas = []
        for p in dist_params:
            m, s = torch.split(p, p.size(1) // 2, dim=-1)
            mus.append(m)
            log_sigmas.append(s)
        return mus, log_sigmas

    def _reparameterize(self, mus, log_sigmas):
        return [m + torch.exp(s) * torch.randn_like(m).to(m.device) for m, s in zip(mus, log_sigmas)]
    
    def _get_kld_loss(self, mus, log_sigmas):
        kld_loss = 0.
        for m, s in zip(mus, log_sigmas):
            kld_loss += 0.5 * (m.pow(2) + torch.exp(s).pow(2) - 2 * s - 1).sum(-1)
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


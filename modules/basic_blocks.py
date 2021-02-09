import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np

from .utils import spectral_norm_fc, Swish

bn_momentum = 0.1
    
class AbstractInvertibleStruct(nn.Module):
    def __init__(self):
        super(AbstractInvertibleStruct, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError

class ActNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm1d, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = nn.Parameter(torch.Tensor(num_channels))
        self._shift = nn.Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :]

    def shift(self):
        return self._shift[None, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() 
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())

class ActNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm2d, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = nn.Parameter(torch.Tensor(num_channels))
        self._shift = nn.Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() 
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())

class GlowRevNet1d(AbstractInvertibleStruct):
    def __init__(self, in_channels, interm_channels=None, do_actnorm=True):
        super(GlowRevNet1d, self).__init__()
        if interm_channels is None:
            interm_channels = in_channels * 16
        self.actnorm = ActNorm1d(in_channels) if do_actnorm else None
        self.f = MLP([in_channels // 2, interm_channels, interm_channels, interm_channels, in_channels], activation=nn.ReLU, batchnorm=False, zero_output=True)
        self.perm = InvertibleLinear(in_channels, qr_init=True, bias=False)
    
    def forward(self, x, ignore_logdet=False):
        b, c = x.shape
        assert c % 2 == 0

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.

        x, perm_logdet = self.perm(x, ignore_logdet)

        z1, z2, mean, scale = self._get_coupling_terms(x)
        z2 = (z2 + mean) * scale

        y = torch.cat([z1, z2], dim=-1)

        if not ignore_logdet:
            logdet = scale.log().view(b, -1).sum(-1) + perm_logdet + an_logdet
        else:
            logdet = 0.
        return y, logdet

    def inverse(self, y):
        b, c = y.shape
        assert c % 2 == 0

        z1, z2, mean, scale = self._get_coupling_terms(y)
        z2 = z2 / scale - mean

        x = torch.cat([z1, z2], dim=-1)

        x = self.perm.inverse(x)
        
        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        return x

    def _get_coupling_terms(self, x):
        c = x.size(1)
        z1 = x[:, :c // 2]
        z2 = x[:, c // 2:]

        h = self.f(z1)
        mean = h[:, :c // 2]
        scale = torch.sigmoid(h[:, c // 2:] + 4.) + 1e-10
        return z1, z2, mean, scale

class GlowRevNet2d(AbstractInvertibleStruct):
    def __init__(self, in_channels, interm_channels=None, do_actnorm=False):
        super(GlowRevNet2d, self).__init__()
        if interm_channels is None:
            interm_channels = in_channels * 4
        self.actnorm = ActNorm(in_channels) if do_actnorm else None
        self.f = nn.Sequential(
            nn.Conv2d(in_channels // 2, interm_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(interm_channels, interm_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(interm_channels, in_channels, 1, 1, 0)
        )
        self.perm = InvertibleLinear(in_channels, qr_init=True, bias=False)
    
    def forward(self, x, ignore_logdet=True):
        b, h, w, c = x.shape
        assert c % 2 == 0

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.

        x, perm_logdet = self.perm(x, ignore_logdet)

        z1, z2, mean, scale = self._get_coupling_terms(x)
        z2 = z2 * scale + mean

        y = torch.cat([z1, z2], dim=1)

        if not ignore_logdet:
            logdet = scale.log().view(b, -1).sum(-1) + perm_logdet + an_logdet
        else:
            logdet = 0.
        return y, logdet

    def inverse(self, y):
        b, h, w, c = y.shape
        assert c % 2 == 0

        z1, z2, mean, scale = self._get_coupling_terms(y)
        z2 = (z2 - mean) / scale

        x = torch.cat([z1, z2], dim=1)

        x = self.perm.inverse(x)
        
        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        return x

    def _get_coupling_terms(self, x):
        c = x.size(1)
        z1 = x[:, :c // 2]
        z2 = x[:, c // 2:]

        h = self.f(z1)
        mean = h[:, 0::2, :, :].contiguous()
        scale = torch.sigmoid(h[:, 1::2, :, :].contiguous() + 2.) # + 1e-5
        return z1, z2, mean, scale


class InvertibleResNet1d(AbstractInvertibleStruct):
    def __init__(self, in_channels, interm_channels, activation=nn.ELU, coeff=0.9, n_power_iterations=1, do_actnorm=False):
        super(InvertibleResNet1d, self).__init__()
        layers = [
            activation(),
            spectral_norm_fc(nn.Linear(in_channels, interm_channels), coeff, n_power_iterations=n_power_iterations),
            activation(),
            spectral_norm_fc(nn.Linear(interm_channels, interm_channels), coeff, n_power_iterations=n_power_iterations),
            activation(),
            spectral_norm_fc(nn.Linear(interm_channels, in_channels), coeff, n_power_iterations=n_power_iterations)
        ]
        self.residual = nn.Sequential(*layers)
        if do_actnorm:
            self.actnorm = ActNorm(in_channels)
        else:
            self.actnorm = None
    
    def forward(self, x, ignore_logdet=False):
        logdet = 0.
        if self.actnorm is not None:
            x, _logdet = self.actnorm(x)
            logdet = logdet + _logdet

        Fx = self.residual(x)
        if not ignore_logdet:
            logdet = self.power_series_matrix_logarithm_trace(Fx, x, 5, 10)
        return x + Fx, logdet
    
    def inverse(self, y, iters=5):
        x = y
        for i in range(iters):
            summand = self.residual(x)
            x = y - summand
        
        if self.actnorm is not None:
            x = self.actnorm.inverse(x)
        
        return x
    
    @staticmethod
    def power_series_matrix_logarithm_trace(Fx, x, k, n):
        # trace estimation including power series
        outSum = Fx.sum(dim=0)
        dim = list(outSum.shape)
        dim.insert(0, n)
        dim.insert(0, x.size(0))
        u = torch.randn(dim).to(x.device)
        trLn = 0
        for j in range(1, k + 1):
            if j == 1:
                vectors = u
            # compute vector-jacobian product
            vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                        retain_graph=True, create_graph=True)[0] for i in range(n)]
            # compute summand
            vectors = torch.stack(vectors, dim=1)
            vjp4D = vectors.view(x.size(0), n, 1, -1)
            u4D = u.view(x.size(0), n, -1, 1)
            summand = torch.matmul(vjp4D, u4D)
            # add summand to power series
            if (j + 1) % 2 == 0:
                trLn += summand / np.float(j)
            else:
                trLn -= summand / np.float(j)
        trace = trLn.mean(dim=1).squeeze()
        return trace
    
class InvertibleLinear(AbstractInvertibleStruct):
    def __init__(self, in_channels, bias=True, qr_init=False):
        super(InvertibleLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(in_channels)) if bias else None

        with torch.no_grad():
            if qr_init:
                self.weight.set_(torch.tensor(np.linalg.qr(np.random.randn(*self.weight.shape))[0]).float())
            else:
                nn.init.normal_(self.weight, 0., 0.02)
            if bias:
                nn.init.zeros_(self.bias)

    def forward(self, x, ignore_logdet=False):
        x_shape = x.shape
        if len(x_shape) == 2:
            Fx = x.mm(self.weight)
        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            Fx = x.permute(0, 2, 3, 1).reshape(-1, c).mm(self.weight)
            Fx = Fx.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Unsupported shape {:s}".format(x_shape))

        if self.bias is not None:
            Fx += self.bias

        logdet = 0.
        if not ignore_logdet:
            logdet = torch.slogdet(self.weight)[-1]
        return Fx, logdet
    
    def inverse(self, y):
        y_shape = y.shape
        if self.bias is not None:
            y -= self.bias
        if len(y_shape) == 2:
            x = y.mm(self.weight.inverse())
        elif len(y_shape) == 4:
            b, c, h, w = y_shape
            x = y.permute(0, 2, 3, 1).reshape(-1, y_shape[1]).mm(self.weight.inverse())
            x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x

class InvertibleResLinear(AbstractInvertibleStruct):
    def __init__(self, in_channels, bias=True):
        super(InvertibleResLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(in_channels)) if bias else None

        nn.init.normal_(self.weight, 0., 1 / in_channels)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        Fx = x.mm(self.weight)
        if self.bias is not None:
            Fx += self.bias
        return x + Fx
    
    def inverse(self, y):
        if self.bias is not None:
            x = y - self.bias
        else:
            x = y
        x = x.mm((self.weight + torch.eye(*self.weight.shape).to(self.weight.device)).inverse())
        return x

    
class ResNetIdentityBlock(nn.Module):
    def __init__(self, in_channels, mid_channels=None, activation=Swish, bottleneck=False, use_sn=False):
        super(ResNetIdentityBlock, self).__init__()
        if not bottleneck:
            if mid_channels is None:
                mid_channels = in_channels
            if use_sn:
                layers = [
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.Conv2d(in_channels, mid_channels, 3, 1, 1)),
                    nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.Conv2d(mid_channels, in_channels, 3, 1, 1)),
                    # SqueezeExcitation(in_channels),
                ]
            else:
                layers = [
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
                    nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                    activation(),
                    nn.Conv2d(mid_channels, in_channels, 3, 1, 1),
                    # SqueezeExcitation(in_channels),
                ]
        else:
            if mid_channels is None:
                mid_channels = in_channels // 4
            layers = [
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(),
                DepthwiseSeparableConv(mid_channels, mid_channels),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv2d(mid_channels, in_channels, 1, 1, 0),
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                # SqueezeExcitation(in_channels),
            ]
        self.residual = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.residual(x)

class ResNetDownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, scale=2, activation=Swish, bottleneck=False, use_sn=False):
        assert scale > 1

        super(ResNetDownscaleBlock, self).__init__()
        if not bottleneck:
            if mid_channels is None:
                mid_channels = out_channels
            if use_sn:
                layers = [
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.Conv2d(in_channels, mid_channels, 3, scale, 1)),
                    nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.Conv2d(mid_channels, out_channels, 3, 1, 1)),
                    # SqueezeExcitation(out_channels),
                ]
            else:
                layers = [
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    nn.Conv2d(in_channels, mid_channels, 3, scale, 1),
                    nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                    activation(),
                    nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
                    # SqueezeExcitation(out_channels),
                ]
        else:
            if mid_channels is None:
                mid_channels = out_channels * 4
            layers = [
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                nn.Conv2d(in_channels, mid_channels, 1, scale, 0),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(),
                DepthwiseSeparableConv(mid_channels, mid_channels),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv2d(mid_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                # SqueezeExcitation(out_channels),
            ]
        self.residual = nn.Sequential(*layers)
        if use_sn:
            self.skip = nn.Sequential(
                *[
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, 3, scale, 1))
                ]
            )
        else:
            self.skip = nn.Sequential(
                *[
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    nn.Conv2d(in_channels, out_channels, 3, scale, 1)
                ]
            )
    
    def forward(self, x):
        return self.skip(x) + self.residual(x)

class ResNetUpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, scale=2, activation=Swish, bottleneck=False, use_sn=False):
        assert scale > 1

        super(ResNetUpscaleBlock, self).__init__()
        if not bottleneck:
            if mid_channels is None:
                mid_channels = in_channels
            if use_sn:
                layers = [
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.Conv2d(in_channels, mid_channels, 3, 1, 1)),
                    nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.ConvTranspose2d(mid_channels, out_channels, 3, scale, 1, output_padding=1)),
                    # SqueezeExcitation(out_channels),
                ]
            else:
                layers = [
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
                    nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                    activation(),
                    nn.ConvTranspose2d(mid_channels, out_channels, 3, scale, 1, output_padding=1),
                    # SqueezeExcitation(out_channels),
                ]
        else:
            if mid_channels is None:
                mid_channels = in_channels * 4
            layers = [
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(),
                DepthwiseSeparableConv(mid_channels, mid_channels),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.ConvTranspose2d(mid_channels, out_channels, 3, scale, 1, output_padding=1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                # SqueezeExcitation(out_channels),
            ]
        self.residual = nn.Sequential(*layers)
        if use_sn:
            self.skip = nn.Sequential(
                *[
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 3, scale, 1, output_padding=1))
                ]
            )
        else:
            self.skip = nn.Sequential(
                *[
                    nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                    activation(),
                    nn.ConvTranspose2d(in_channels, out_channels, 3, scale, 1, output_padding=1)
                ]
            )
    
    def forward(self, x):
        return self.skip(x) + self.residual(x)

class ResNetIdentityBlock3d(nn.Module):
    def __init__(self, in_channels, mid_channels=None, activation=Swish, bottleneck=False):
        super(ResNetIdentityBlock3d, self).__init__()
        if not bottleneck:
            if mid_channels is None:
                mid_channels = in_channels
            layers = [
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(in_channels, mid_channels, 3, 1, 1),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, in_channels, 3, 1, 1),
                SqueezeExcitation3d(in_channels),
            ]
        else:
            if mid_channels is None:
                mid_channels = in_channels * 4
            layers = [
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, mid_channels, 1, 1, 0),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, mid_channels, 3, 1, 1),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, in_channels, 1, 1, 0),
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                SqueezeExcitation3d(in_channels),
            ]
        self.residual = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.residual(x)

class ResNetDownscaleBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, scale=2, activation=Swish, bottleneck=False):
        assert scale > 1

        super(ResNetDownscaleBlock3d, self).__init__()
        if not bottleneck:
            if mid_channels is None:
                mid_channels = out_channels
            layers = [
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(in_channels, mid_channels, 3, scale, 1),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, out_channels, 3, 1, 1),
                SqueezeExcitation3d(out_channels),
            ]
        else:
            if mid_channels is None:
                mid_channels = out_channels // 4
            layers = [
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, mid_channels, 1, scale, 0),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, mid_channels, 3, 1, 1),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, out_channels, 1, 1, 0),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
                SqueezeExcitation3d(out_channels),
            ]
        self.residual = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            *[
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(in_channels, out_channels, 3, scale, 1),
            ]
        )
    
    def forward(self, x):
        return self.skip(x) + self.residual(x)

class ResNetUpscaleBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, scale=2, activation=Swish, bottleneck=False):
        assert scale > 1

        super(ResNetUpscaleBlock3d, self).__init__()
        if not bottleneck:
            if mid_channels is None:
                mid_channels = in_channels
            layers = [
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(in_channels, mid_channels, 3, 1, 1),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.ConvTranspose3d(mid_channels, out_channels, 3, scale, 1, output_padding=1),
                SqueezeExcitation3d(out_channels),
            ]
        else:
            if mid_channels is None:
                mid_channels = in_channels // 4
            layers = [
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, mid_channels, 1, 1, 0),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.Conv3d(mid_channels, mid_channels, 3, 1, 1),
                nn.BatchNorm3d(mid_channels, momentum=bn_momentum),
                activation(),
                nn.ConvTranspose3d(mid_channels, out_channels, 3, scale, 1, output_padding=1),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
                SqueezeExcitation3d(out_channels),
            ]
        self.residual = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            *[
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                activation(),
                nn.ConvTranspose3d(in_channels, out_channels, 3, scale, 1, output_padding=1)
            ]
        )
    
    def forward(self, x):
        return self.skip(x) + self.residual(x)

class EncoderCell(nn.Module):
    def __init__(self, in_channels, mid_channels=None, activation=Swish):
        super(EncoderCell, self).__init__()
        layers = [
            nn.BatchNorm2d(in_channels, momentum=bn_momentum),
            activation(),
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
            activation(),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1),
            SqueezeExcitation(in_channels),
        ]
        self.residual = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.residual(x)

class StyleDecoderCell(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, mid_channels=None, activation=Swish):
        super(StyleDecoderCell, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.activation1 = activation()
        self.adaNoise1 = AdaNoise(mid_channels)
        self.adaIN1 = AdaIN(style_channels, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.activation2 = activation()
        self.adaNoise2 = AdaNoise(out_channels)
        self.adaIN2 = AdaIN(style_channels, out_channels)
        self.se = SqueezeExcitation(out_channels)
    
    def forward(self, x, style):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.adaNoise1(out)
        out = self.adaIN1(out, style)
        out = self.conv2(out)
        out = self.activation2(out)
        out = self.adaNoise2(out)
        out = self.adaIN2(out, style)
        out = self.se(out)
        return out

class StyleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, n_l, upsample=2, mid_channels=None):
        super(StyleDecoderBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if upsample > 1:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear')
        else:
            self.upsample = None
        
        self.cells = nn.ModuleList([StyleDecoderCell(in_channels if i == 0 else out_channels, 
                                                     out_channels, 
                                                     style_channels, 
                                                     mid_channels) for i in range(n_l)])
    
    def forward(self, x, style):
        if self.upsample is not None:
            x = self.upsample(x)
        
        for c in self.cells:
            x = c(x, style)
        
        return x

class AdaIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaIN, self).__init__()
        self.affine = nn.Linear(in_channels, out_channels * 2)
        self.norm = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x, style):
        b, c, h, w = x.shape
        x_normed = self.norm(x)
        mean, std = torch.split(self.affine(style).view(b, c * 2, 1, 1), c, dim=1)
        x_transformed = x_normed * torch.sigmoid(std) + mean
        return x_transformed

class AdaNoise(nn.Module):
    def __init__(self, in_channels):
        super(AdaNoise, self).__init__()
        self.scale = nn.Parameter(torch.zeros(in_channels,), requires_grad=True)
    
    def forward(self, x):
        b, c, h, w = x.shape
        noise = torch.randn(b, 1, h, w).to(x.device)
        scaled_noise = noise * self.scale.view(1, -1, 1, 1)
        return scaled_noise + x

class MLP(nn.Module):
    def __init__(self, in_channels, activation=Swish, batchnorm=True, zero_output=False):
        super(MLP, self).__init__()
        assert len(in_channels) >= 2
        layers = [nn.Linear(in_channels[0], in_channels[1])]
        for i in range(1, len(in_channels) - 1):
            if batchnorm:
                layers.append(nn.BatchNorm1d(in_channels[i], momentum=bn_momentum))
            layers.append(activation())
            layers.append(nn.Linear(in_channels[i], in_channels[i + 1]))
        self.layers = nn.ModuleList(layers)

        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0., 1. / np.sqrt(m.weight.size(0)))
                nn.init.zeros_(m.bias)
        
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, activation=Swish, factor=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // factor)
        self.activation = activation()
        self.fc2 = nn.Linear(in_channels // factor, in_channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        pooled = x.view(b, c, h * w).mean(-1)
        scale = torch.sigmoid(self.fc2(self.activation(self.fc1(pooled))))
        return x * scale.view(b, c, 1, 1)

class SqueezeExcitation3d(nn.Module):
    def __init__(self, in_channels, activation=Swish, factor=4):
        super(SqueezeExcitation3d, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // factor)
        self.activation = activation()
        self.fc2 = nn.Linear(in_channels // factor, in_channels)
    
    def forward(self, x):
        b, c, h, w, d = x.shape
        pooled = x.view(b, c, h * w * d).mean(-1)
        scale = torch.sigmoid(self.fc2(self.activation(self.fc1(pooled))))
        return x * scale.view(b, c, 1, 1, 1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Ref: https://github.com/gap370/pixelshuffle3d/blob/master/pixelshuffle3d.py
class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
        
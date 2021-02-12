import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalize

class GeodesicInterpPreimage(nn.Module):
    def __init__(self):
        super(GeodesicInterpPreimage, self).__init__()
        self.input_type = 'sim'
    
    def forward(self, K, y, topk=-1):
        if topk == -1:
            topk = K.size(-1)

        b = K.size(0)
        K_top = K.topk(topk, dim=-1)
        inds = K_top.indices
        weight = K_top.values

        mu = y[inds[:, 0].unsqueeze(1)].view(b, -1)
        _w_sum = weight[:, 0]
        for i in range(1, inds.size(1)):
            v = y[inds[:, i].unsqueeze(1)].view(b, -1)
            theta = torch.acos(torch.clamp((mu * v).view(b, -1).sum(-1), min=-1 + 1e-7, max=1 - 1e-7)).unsqueeze(1)
            _w_sum += weight[:, i]
            t = (weight[:, i] / _w_sum).unsqueeze(1)

            mu_prime = (torch.sin((1 - t) * theta) * mu + torch.sin(t * theta) * v) / torch.sin(theta)
            mu_prime = torch.where(theta == 0., mu, mu_prime)
            mu = mu_prime
        preimage = mu.view(b, *y.shape[1:])
        return preimage, inds

class WeightedMeanPreimage(nn.Module):
    def __init__(self):
        super(WeightedMeanPreimage, self).__init__()
        self.input_type = 'sim'

    def forward(self, K, y, topk=-1):
        K_top = K.topk(topk, dim=-1)
        inds = K_top.indices
        weight = K_top.values

        # print(weight)

        weight_mat = torch.zeros_like(K).to(K.device).scatter_(-1, inds, normalize(weight, p=1, dim=-1))
        preimage = weight_mat.mm(y)
        return preimage, inds

class IterativePreimage(nn.Module):
    def __init__(self, kernel_fn, n_iters=15):
        super(IterativePreimage, self).__init__()
        self.input_type = 'sim'
        self.kernel_fn = kernel_fn
        self.n_iters = 15

    def forward(self, K, y, topk=-1):
        K_top = K.topk(topk, dim=-1)
        inds = K_top.indices
        weight = K_top.values

        weight_mat = torch.zeros_like(K).to(K.device).scatter_(-1, inds, normalize(weight, p=1, dim=-1))
        preimage = weight_mat.mm(y)

        for i in range(self.n_iters):
            K_i = self.kernel_fn(preimage, y)
            K_top = K_i.topk(topk, dim=-1)
            inds = K_top.indices
            weight = K_top.values

            weight_mat = torch.zeros_like(K).to(K.device).scatter_(-1, inds, normalize(weight, p=1, dim=-1))
            preimage = weight_mat.mm(y)
        
        return preimage, inds


class MDSPreimage(nn.Module):
    def __init__(self):
        super(MDSPreimage, self).__init__()
        self.input_type = 'sq_dist'
    
    def forward(self, sq_dist, y, topk=-1):
        b = sq_dist.size(0)
        dist_nearest = sq_dist.topk(topk, dim=-1, largest=False)
        inds = dist_nearest.indices
        weight = dist_nearest.values

        n, c = y.shape
        y_nearest = y.unsqueeze(0).expand(b, n, c).gather(1, inds.unsqueeze(-1).expand(b, topk, c))
        y_mean = y_nearest.mean(dim=1)
        y_nearest_centered = y_nearest - y_mean.unsqueeze(1)

        y_outer = y_nearest_centered.transpose(-2, -1).bmm(y_nearest_centered)
        y_inner = y_nearest_centered.bmm(y_nearest_centered.transpose(-2, -1))

        y_outer_inv = torch.inverse(y_outer)
        y_inner_diag = y_inner.diagonal(dim1=-2, dim2=-1)

        # Computed with the MDS-based pre-image formula in https://hal.archives-ouvertes.fr/hal-01965582/file/11.spm_draft.pdf 
        preimage = 0.5 * (y_outer_inv.bmm(y_nearest_centered.transpose(-2, -1)\
                         .bmm((y_inner_diag - weight).unsqueeze(-1)))).squeeze(-1)
        preimage = preimage + y_mean
        return preimage, inds
import torch
import torch.nn as nn
import numpy as np
from .kernel_approx import NystromPFOperator
from .utils import normalize, LinearWrapper, batch_subset_kernel, sq_dist
from sklearn.mixture import GaussianMixture

class GMMGenerator(nn.Module):
    def __init__(self, latent_samples, n_components=10, device=torch.device('cuda')):
        super(GMMGenerator, self).__init__()
        self.models = []
        self.device = device
        self.groups = len(latent_samples)
        for i in range(self.groups):
            self.models.append(GaussianMixture(n_components=n_components, 
                                               max_iter=2000,
                                               verbose=2, 
                                               tol=1e-3).fit(latent_samples[i].detach().cpu().numpy()))
        n = latent_samples[0].size(0)
        self.y = nn.Parameter(torch.cat([_y.view(n, -1) for _y in latent_samples], dim=-1)).to(device)

    def sample(self, size, topk=10):
        samples = []
        for i in range(self.groups):
            sampled, _ = self.models[i].sample(size)
            samples.append(torch.tensor(sampled, dtype=torch.float, device=self.device))
        
        samples_cat = torch.cat([_x.view(size, -1) for _x in samples], dim=-1)

        dist = sq_dist(samples_cat, self.y)
        inds = torch.topk(dist, 10, largest=False).indices
        return samples, inds

class KernelPFGenerator(nn.Module):
    def __init__(self, input_kernel, output_kernel, output_samples, preimage_module, spherical=True, labels=None, prior_sample_fn=None, nystrom_compression=False, nystrom_points=1000, epsilon=0., p_dim=-1, device=torch.device('cuda')):
        super(KernelPFGenerator, self).__init__()
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.prior_sample_fn = prior_sample_fn
        self.spherical = spherical

        n = output_samples[0].size(0)
        self.y_list = nn.ParameterList([nn.Parameter(y, requires_grad=False) for y in output_samples])
        self.y = nn.Parameter(torch.cat([_y.view(n, -1) for _y in self.y_list], dim=-1)).to(device)
        self.labels = labels
        
        self.preimage_module = preimage_module

        if p_dim == -1:
            self.p_dim = np.sum([_y.size(1) for _y in self.y_list])
        else:
            self.p_dim = p_dim

        self.x = nn.Parameter(self.get_prior_sample(n)).to(device)

        with torch.no_grad():
            if nystrom_compression:
                op = NystromPFOperator(self.x, self.y, self.input_kernel, self.output_kernel, nystrom_points, epsilon=epsilon)
            else:
                # self.Gxx = batch_subset_kernel(self.x, self.input_kernel, np.arange(n), np.arange(n), 5000)
                # self.Gyy = batch_subset_kernel(self.y, self.output_kernel, np.arange(n), np.arange(n), 5000)
                self.Gxx = nn.Parameter(self.input_kernel(self.x, self.x), requires_grad=False)
                self.Gyy = nn.Parameter(self.output_kernel(self.y, self.y), requires_grad=False)

                # Compute Tikhonov-regularized inverse
                Gxx_reg = (self.Gxx + epsilon * n * torch.eye(n).to(self.Gxx.device))
                self.Gxx_inv = nn.Parameter(Gxx_reg.inverse(), requires_grad=False)

                op = LinearWrapper(self.Gyy.mm(self.Gxx_inv), trainable=False)
        self.operator = op
    
    def get_transferred_sq_dist(self, x_prime):
        Gxx_prime = self.input_kernel(self.x, x_prime)
        Kxy = self.operator(Gxx_prime)
        Kxx = Gxx_prime.T.mm(self.Gxx_inv).mm(Kxy).diag().unsqueeze(1)
        Kyy = self.Gyy.diag().unsqueeze(1)

        sq_dist = Kxx - 2 * Kxy.T + Kyy.T
        # import pdb; pdb.set_trace()
        # print(sq_dist)
        return sq_dist
    
    def get_transferred_similarity(self, x_prime):
        Gxx_prime = self.input_kernel(self.x, x_prime)
        K = self.operator(Gxx_prime).t()
        return K
    
    def get_prior_sample(self, size):
        if self.prior_sample_fn is None:
            if self.spherical:
                return normalize(torch.randn(size, self.p_dim), p=2, dim=-1)
            else:
                return torch.randn(size, self.p_dim)
        return self.prior_sample_fn(size)
    
    def forward(self, sample_size):
        return self.sample(sample_size)

    def sample(self, sample_size, topk=10):
        with torch.no_grad():
            x_prime = self.get_prior_sample(sample_size).to(self.x.device)

            if self.preimage_module.input_type == 'sim':
                K = self.get_transferred_similarity(x_prime)
            elif self.preimage_module.input_type == 'sq_dist':
                K = self.get_transferred_sq_dist(x_prime)

        y_prime = []
        for _y in self.y_list:
            _y, inds = self.preimage_module(K, _y, topk=topk)
            y_prime.append(_y)

        return y_prime, inds
    
    def get_density_operator(self, reference_samples, epsilon=1e-3):
        b = reference_samples.size(0)
        n = self.Gxx.size(0)
        z = reference_samples.to(self.y.device)

        Gzy = self.output_kernel(z, self.y)
        Gzz_inv2 = torch.matrix_power((self.output_kernel(z, z) + epsilon * torch.eye(b).to(z.device)), -2)
        Gxmu = (self.Gxx * (1 - torch.eye(n).to(self.Gxx.device))).sum(dim=-1).unsqueeze(-1) / (n - 1)

        beta = -Gzz_inv2.mm(Gzy.mm(self.Gxx_inv.mm(Gxmu))) / (b ** 2)

        def get_prob(points):
            G = self.output_kernel(points, z)
            prob = G.mm(beta).squeeze()
            return prob
        
        return get_prob
        
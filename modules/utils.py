import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize(tensor, p=2, dim=-1, eps=1e-15):
    return tensor / (tensor.norm(p=p, dim=dim).unsqueeze(dim) + eps)

def flatten_normalize(tensor, p=2, eps=1e-15):
    t_shape = tensor.shape
    tensor_flattened = tensor.view(t_shape[0], -1)
    tensor_norm = tensor_flattened.norm(p=p, dim=-1).view(t_shape[0], 1) + eps
    return (tensor_flattened / tensor_norm).view(*t_shape)

def sq_dist(x, y):
    x_sq = x.square().sum(-1).unsqueeze(-1)
    y_sq = y.square().sum(-1).unsqueeze(-1)

    xy = x.mm(y.T)
    return x_sq - 2 * xy + y_sq.T

class VisualizeLayer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def forward(self, x):
        n, c, h, w = x.shape
        x_img = x[:5, :5].reshape(25, 1, h, w)
        x_flat = x_img.view(25, h * w)
        x_min = x_flat.min(dim=1).values.view(25, 1, 1, 1)
        x_max = x_flat.max(dim=1).values.view(25, 1, 1, 1)
        x_img = (x_img - x_min) / (x_max - x_min + 1e-7)
        viz.images(x_img, nrow=5, win=self.name, opts={'title': self.name})
        return x

class NegPad2d(nn.Module):
    def __init__(self, size):
        super().__init__()
        if type(size) == int:
            self.size = (size, size, size, size)
        elif len(size) == 2:
            self.size = (size[0], size[0], size[1], size[1])
        else:
            assert len(size) == 4
            self.size = size
    
    def forward(self, x):
        _, _, h, w = x.shape
        return x[:, :, self.size[0]:h - self.size[1], self.size[2]:w - self.size[3]]

class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x.shape)
        return x
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class SpectralNorm(object):
    _version = 1

    def __init__(self, coeff, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.coeff = coeff
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        sigma_log = getattr(module, self.name + '_sigma') # for logging
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        sigma_det = sigma.detach()
        torch.max(torch.ones(1).to(weight.device), sigma_det / self.coeff,
                  out=sigma_log)
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, coeff, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(coeff, name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class SpectralNormLoadStateDictPreHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[prefix + fn.name + '_v'] = v

class SpectralNormStateDictHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


def spectral_norm_fc(module, coeff, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d,
                               torch.nn.Linear)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, coeff, n_power_iterations, dim, eps)
    return module

def mix_rbf_mmd(X, Y, sigma_list=[1, np.sqrt(2), 2, 2 * np.sqrt(2), 4], biased=True):
    from .kernel_fn import mix_rbf_kernel
    m = X.size(0)
    K = mix_rbf_kernel(X, Y, sigma_list)
    return mmd(K[:m, :m], K[:m, m:], K[m:, m:], biased=biased)

def poly_mmd(X, Y, deg=2, biased=True):
    from .kernel_fn import polynomial_kernel
    m = X.size(0)
    K = polynomial_kernel(X, Y, deg=deg)
    return mmd(K[:m, :m], K[:m, m:], K[m:, m:], biased=biased)

def mmd(K_XX, K_XY, K_YY, biased=False):
    m = K_XX.size(0)

    diag_X = torch.diag(K_XX)                       # (m,)
    diag_Y = torch.diag(K_YY)                       # (m,)
    sum_diag_X = torch.sum(diag_X)
    sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))
    
    return mmd

class SubsetDataset:
    def __init__(self, full_dataset, subset_size=None, subset_ratio=None, shuffle=False):
        if subset_size is not None:
            if subset_ratio is not None:
                raise ValueError("Subset size and subset ratio are mutually exclusive")

            if subset_size < 0:
                self.len = len(full_dataset)
            else:
                self.len = subset_size
        else:
            if subset_ratio is None:
                raise ValueError("At least one of subset size or subset ratio need to be set")
            
            self.len = len(full_dataset) * subset_ratio
        self.dataset = full_dataset

        if shuffle:
            self.index_dict = np.random.permutation(len(self.dataset))[:self.len]
        else:
            self.index_dict = np.arange(self.len)
        
    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError("Element index out of range")
        return self.dataset[self.index_dict[idx]]
    
    def __len__(self):
        return self.len

class LinearWrapper(nn.Module):
    def __init__(self, w_matrix, trainable=False):
        super(LinearWrapper, self).__init__()
        self.weight = nn.Parameter(w_matrix, requires_grad=trainable)
    
    def forward(self, x):
        return self.weight.mm(x)

def batch_subset_kernel(X, kernel_obj, row_inds, col_inds, batch_size, Y=None):
    K_row_list = []
    for i in range(0, int(np.ceil(len(row_inds) / batch_size))):
        K_col_list = []
        for j in range(0, int(np.ceil(len(col_inds) / batch_size))):
            K_col_list.append(subset_kernel(X, 
                                            kernel_obj, 
                                            row_inds[batch_size*i: min(len(row_inds), batch_size*(i + 1))], 
                                            col_inds[batch_size*j: min(len(col_inds), batch_size*(j + 1))],
                                            Y))
            K_col = torch.cat(K_col_list, dim=-1)
        K_row_list.append(K_col)
    return torch.cat(K_row_list, dim=0)

def subset_kernel(X, kernel_obj, row_inds, col_inds, Y=None):
    if Y is None:
        Y = X
    row_set = X[row_inds]
    col_set = Y[col_inds]
    return kernel_obj(row_set, col_set)

def unit_sphere_to_cartesian_3d(coord):
    assert coord.size(1) == 3

    XsqPlusYsq = coord[:,0]**2 + coord[:,1]**2
    theta = torch.atan2(torch.sqrt(XsqPlusYsq), coord[:,2])
    phi = torch.atan2(coord[:,1],coord[:,0])

    phi = phi + 2 * np.pi * (phi < 0)

    return torch.stack((theta, phi), dim=1)

def cartesian_to_unit_sphere_3d(coord):
    assert coord.size(1) == 2

    x = torch.sin(coord[:, 0]) * torch.cos(coord[:, 1])
    y = torch.sin(coord[:, 0]) * torch.sin(coord[:, 1])
    z = torch.cos(coord[:, 0])

    return torch.stack((x, y, z), dim=1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


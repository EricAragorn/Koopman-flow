import torch
import torch.nn as nn
import numpy as np

class GaussianDiag(nn.Module):
    def __init__(self, n_channels, mean, logsd, trainable=False):
        super(GaussianDiag, self).__init__()
        self.n_channels = n_channels
        self.mean = nn.Parameter(torch.tensor([mean] * n_channels).reshape(1, n_channels), requires_grad=trainable)
        self.logsd = nn.Parameter(torch.tensor([logsd] * n_channels).reshape(1, n_channels), requires_grad=trainable)

    def _flatten_sum(self, logps):
        b = logps.size(0)
        return logps.view(b, -1).sum(-1)
    
    def logps(self, x):
        return -0.5 * (2. * self.logsd) - ((x - self.mean) ** 2) / (2 * torch.exp(self.logsd))
    
    def logp(self, x):
        dim = float(np.prod(x.shape[1:]))
        Log2PIdim = -0.5* (dim*float(np.log(2 * np.pi)))
        return Log2PIdim + self._flatten_sum(self.logps(x))
    
    def sample(self, n_samples, temp=1.):
        return torch.randn(n_samples, self.n_channels).to(self.logsd.device) * torch.exp(self.logsd) * temp + self.mean
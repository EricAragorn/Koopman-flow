import torch
import torch.nn as nn
import numpy as np
from .utils import normalize, batch_subset_kernel

# Pytorch Implementation of https://arxiv.org/abs/1605.07583
# Original Matlab implementation https://github.com/cnmusco/recursive-nystrom
def recursiveNystrom(X, kernel_obj, samples):
    n = X.size(0)
    if samples >= n:
        raise ValueError("Samples greater than or equal to # of rows. Use direct kernel instead.")
    kernel_batch_size = 5000
    # import pdb; pdb.set_trace()

    oversamp = np.log(samples)
    k = int(np.ceil(samples / (4 * oversamp)))
    nLevels = int(np.ceil(np.log(n / samples) / np.log(2)))

    perm = torch.randperm(n).to(X.device)

    lSize = [n]
    for i in range(1, nLevels + 1):
        lSize.append(int(np.ceil(lSize[-1] / 2)))
    
    samp = torch.arange(lSize[-1], dtype=torch.long).to(X.device)
    rInd = perm[:lSize[-1]]
    with torch.no_grad():
        weights = torch.ones(len(rInd),).to(X.device)
        kDiag = kernel_obj.diag(X)

        for l in reversed(range(0, nLevels)):
            rIndCurr = perm[:lSize[l]]
            KS = batch_subset_kernel(X, kernel_obj, rIndCurr, rInd, kernel_batch_size)
            SKS = KS[samp, :]
            SKSn = SKS.size(0)

            if (k >= SKSn):
                _lambda = 1e-6
            else:
                eigs = (SKS * weights.unsqueeze(1) * weights.unsqueeze(0)).eig()[0]
                top_eig_sum = eigs[:, 0].abs().sort()[0][:k].sum()
                _lambda = ((SKS.diag() * weights).pow(2).sum() - top_eig_sum) / k
                del eigs
        
            R = (SKS + (_lambda * weights.pow(-2)).diag_embed()).inverse()
            if l > 0:
                levs = (
                    oversamp * (1 / _lambda) * 
                    torch.clamp_min((kDiag[rIndCurr] - (KS.mm(R) * KS).sum(-1)), 0.)
                ).clamp_max(1.)
                samp = (torch.rand(lSize[l],).to(levs.device) < levs).nonzero().squeeze()

                if samp.numel() == 0:
                    levs.fill_(samples / lSize[l])
                    samp = torch.randperm(lSize[l])[:samples].to(levs.device)
                weights = (1 / levs[samp]).sqrt()
            else:
                levs = (
                    (1 / _lambda) * 
                    torch.clamp_min((kDiag[rIndCurr] - (KS.mm(R) * KS).sum(-1)), 0.)
                ).clamp_max(1.)
                samp = np.random.choice(np.arange(n), samples, replace=False, p=normalize(levs, p=1).cpu().numpy())
            rInd = perm[samp]

            del KS, SKS, R, levs
            torch.cuda.empty_cache()

        C = batch_subset_kernel(X, kernel_obj, np.arange(n), rInd, kernel_batch_size)
        W = C[rInd, :]
    return C, W, rInd

class NystromPFOperator(nn.Module):
    def __init__(self, x, y, input_kernel, output_kernel, samples, epsilon=1e-8):
        """Implementation of Nystrom-approximation-based PF operator

        Args:
            x, y: input/output features
            input_kernel, output_kernel: input/output kernels
            samples (int): size of Nystrom landmark points
            epsilon (float, optional): regularization parameter for computing matrix inverse. Defaults to 1e-8.
        """
        super(NystromPFOperator, self).__init__()
        self.nx, self.ny = x.size(0), y.size(0)
        self.epsilon = epsilon
        KSxx, SKSxx, x_nystrom_point_inds = [t for t in recursiveNystrom(x, input_kernel, samples)]
        KSyy, SKSyy, y_nystrom_point_inds = [t for t in recursiveNystrom(y, output_kernel, samples)]

        def left_mul_inverse(KS, SKS):
            # Inverse using Woodbury matrix identity (currently not working due to numerical errors)
            # n = KS.size(0)
            # KS_param = nn.Parameter(KS, requires_grad=False)
            # reg = (self.epsilon * n * SKS) + KS.T.mm(KS)
            # print(reg.symeig()[0])
            # reg_inv_param = nn.Parameter(reg.inverse(), requires_grad=False)

            # def _map(K):
            #     return (K - KS_param.mm(reg_inv_param.mm(KS_param.T.mm(K)))) / (self.epsilon * n)

            # Inverse using pseudo-inverse
            KS_pinv_param = nn.Parameter(KS.pinverse(), requires_grad=False)
            SKS_param = nn.Parameter(SKS, requires_grad=False)
            def _map(K):
                return KS_pinv_param.T.mm(SKS_param.mm(KS_pinv_param.mm(K)))
            return _map
        
        def left_mul(KS, SKS):
            s = SKS.size(0)
            KS_param = nn.Parameter(KS, requires_grad=False)
            SKS_pinv_param = nn.Parameter((SKS + s * epsilon * torch.eye(samples).to(SKS.device)).inverse(), requires_grad=False)

            def _map(K):
                return KS_param.mm(SKS_pinv_param.mm(KS_param.T.mm(K)))
            return _map
            

        self.map1 = left_mul_inverse(KSxx, SKSxx)
        self.map2 = left_mul(KSyy, SKSyy)
    
    def forward(self, k_prime):
        out = self.map1(k_prime)
        out = self.map2(out)
        return out

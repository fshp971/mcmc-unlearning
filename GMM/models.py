import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GMM(nn.Module):
    def __init__(self, kk, dim, std, n):
        super(GMM, self).__init__()
        self.kk = kk
        self.dim = dim
        self.n = n
        assert len(std) == dim
        self.std = torch.Tensor(std)
        self.mk = Parameter(torch.Tensor(kk, dim))

        self.mk.data.normal_(0, 1)

    def forward(self, x):
        x = x.reshape(-1, 1, self.dim)
        out = ((x - self.mk)**2) * (-0.5) - np.log(np.sqrt(2*np.pi))
        out = out.sum(dim=2)
        return out.exp()

    def pa_loss(self):
        ''' i.e. log_prior '''
        out = (self.mk**2)*(-0.5) -self.std.log() -np.log(np.sqrt(2*np.pi))
        return -out.sum()

    def x_loss(self, x, _y):
        out = _y.mean(dim=1).log().sum()
        return -out

    def loss(self, x, _y):
        return self.pa_loss() + self.x_loss(x, _y) * (self.n/len(x))

    def dump_numpy_dict(self):
        return {'kk': self.kk,
                'dim': self.dim,
                'std': np.array(self.std.data.cpu()),
                'mk': np.array(self.mk.data.t().cpu())}

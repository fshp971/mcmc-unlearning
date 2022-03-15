import numpy as np
import torch
import torch.nn as nn


class mcmcModule(nn.Module):
    def log_prior(self):
        res = - (self.weight ** 2).sum() / (2 * self.prior_sig**2)
        if self.bias is not None:
            res += - (self.bias ** 2).sum() / (2 * self.prior_sig**2)
        return res


class mcmcLinear(nn.Linear, mcmcModule):
    def __init__(self, prior_sig, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.prior_sig = prior_sig


class mcmcConv2d(nn.Conv2d, mcmcModule):
    def __init__(self, prior_sig, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        self.prior_sig = prior_sig


class mcmcBatchNorm2d(nn.BatchNorm2d, mcmcModule):
    def __init__(self, prior_sig, *args, **kwargs):
        nn.BatchNorm2d.__init__(self, *args, **kwargs)
        self.prior_sig = prior_sig
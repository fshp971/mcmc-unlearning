import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from . import mcmc_modules

from .mcmc_modules import mcmcLinear as Linear
from .mcmc_modules import mcmcConv2d as Conv2d
from .mcmc_modules import mcmcBatchNorm2d as BatchNorm2d


class mcmcArch(nn.Module):
    def __init__(self):
        super(mcmcArch, self).__init__()
        self.__dict__['_mcmc_modules'] = dict()

    def __setattr__(self, name, value):
        if isinstance(value, mcmc_modules.mcmcLinear) or \
            isinstance(value, mcmc_modules.mcmcConv2d):
            self.__dict__['_mcmc_modules'][name] = value
        super(mcmcArch, self).__setattr__(name, value)

    def log_prior(self):
        # return 0
        res = 0
        for name, mod in self.__dict__['_mcmc_modules'].items():
            res += mod.log_prior()
        return res


class mcmcMLP(mcmcArch):
    def __init__(self, prior_sig=None, in_dims=1):
        super(mcmcMLP, self).__init__()

        self.linear1 = Linear(prior_sig, 32*32*in_dims, 512)
        self.linear2 = Linear(prior_sig, 512, 512)
        self.linear3 = Linear(prior_sig, 512, 10)

    def forward(self, x):
        x = self.linear1( x.view(len(x),-1) )
        x = F.relu(x)
        x = self.linear2( x.view(len(x),-1) )
        x = F.relu(x)
        x = self.linear3( x.view(len(x),-1) )
        return x


''' standard LeNet-5 (for 32*32 input) '''
class mcmcLeNet(mcmcArch):
    def __init__(self, prior_sig=None, in_dims=1, out_dims=10):
        super(mcmcLeNet, self).__init__()

        self.conv1 = Conv2d(prior_sig, in_dims, 6, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(prior_sig, 6, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = Linear(prior_sig, 5*5*16, 120)
        self.fc2 = Linear(prior_sig, 120, 84)
        self.fc3 = Linear(prior_sig, 84, out_dims)

    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ) )
        x = self.pool2( F.relu( self.conv2(x) ) )
        x = F.relu( self.fc1(x.view(len(x), -1)) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)
        return x


''' small cnn (for 32*32 input) '''
class mcmcSmallCNN(mcmcArch):
    def __init__(self, prior_sig=None, in_dims=1, out_dims=10):
        super(mcmcSmallCNN, self).__init__()

        self.conv1 = Conv2d(prior_sig, in_dims, 64, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(prior_sig, 64, 64, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = Linear(prior_sig, 5*5*64, 384)
        self.fc2 = Linear(prior_sig, 384, out_dims)

    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ) )
        x = self.pool2( F.relu( self.conv2(x) ) )
        x = F.relu( self.fc1(x.view(len(x), -1)) )
        x = self.fc2(x)
        return x

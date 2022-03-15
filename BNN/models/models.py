import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm2d


''' standard LeNet-5 (for 32*32 input) '''
class LeNet(nn.Module):
    def __init__(self, in_dims=1, out_dims=10):
        super(LeNet, self).__init__()

        self.conv1 = Conv2d(in_dims, 6, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = Linear(5*5*16, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, out_dims)

    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ) )
        x = self.pool2( F.relu( self.conv2(x) ) )
        x = F.relu( self.fc1(x.view(len(x), -1)) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)
        return x


''' small cnn (for 32*32 input) '''
class SmallCNN(nn.Module):
    def __init__(self, in_dims=1, out_dims=10):
        super(SmallCNN, self).__init__()

        self.conv1 = Conv2d(in_dims, 64, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = Linear(5*5*64, 384)
        self.fc2 = Linear(384, out_dims)

    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ) )
        x = self.pool2( F.relu( self.conv2(x) ) )
        x = F.relu( self.fc1(x.view(len(x), -1)) )
        x = self.fc2(x)
        return x


class VGG9(nn.Module):
    def __init__(self, prior_sig=None, in_dims=3):
        super(VGG9, self).__init__()
        self.conv1 = Conv2d(in_dims, 32, 3, padding=1)
        self.bn1 = BatchNorm2d(32)

        self.conv2 = Conv2d(32, 64, 3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = Conv2d(64, 128, 3, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.conv4 = Conv2d(128, 128, 3, padding=1)
        self.bn4 = BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = Conv2d(128, 256, 3, padding=1)
        self.bn5 = BatchNorm2d(256)
        self.conv6 = Conv2d(256, 256, 3, padding=1)
        self.bn6 = BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(2)

        self.fc1 = Linear(4096, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 10)

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu( self.bn1(self.conv1(x)) )
        x = self.pool2(F.relu( self.bn2(self.conv2(x)) ))
        x = F.relu( self.bn3(self.conv3(x)) )
        x = self.pool4(F.relu( self.bn4(self.conv4(x)) ))
        x = F.relu( self.bn5(self.conv5(x)) )
        x = self.pool6(F.relu( self.bn6(self.conv6(x)) ))

        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

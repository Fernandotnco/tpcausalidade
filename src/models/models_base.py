import torch
import torch.nn as nn
import math

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, before = None, after = False, bias=True, device=None, dtype=None):
        super(Dense, self).__init__()
        self.dense = nn.Linear(in_channels, out_channels, bias = bias)
        self.dense.apply(weights_init('xavier'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid
        elif after == 'softmax':
            self.after = nn.Softmax(dim = 1)

        if before=='ReLU':
            self.before = nn.ReLU(inplace=False)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)
          
    def forward(self, x):
      if hasattr(self, 'before'):
          x = self.before(x)

      x = self.dense(x)

      if hasattr(self, 'after'):
          x = self.after(x)

      return x

class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=True):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('orthogonal'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = nn.Sigmoid()
        elif after == 'softmax':
            self.after = nn.Softmax(dim = 1)
        elif(after == 'maxPooling'):
            self.after = nn.MaxPool2d(kernel_size = 2)
        elif(after == 'flatten'):
            self.after = nn.Flatten

        if before=='ReLU':
            self.before = nn.ReLU(inplace=False)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x
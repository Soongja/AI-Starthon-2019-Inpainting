import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class Inpaint(nn.Module):
    def __init__(self):
        super(Inpaint, self).__init__()
        nf = 64
        self.conv1 = nn.Conv2d(4, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = F.avg_pool2d(h, 2)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.conv5(h)
        h = F.interpolate(h, scale_factor=2)
        x_hat = self.conv6(h)
        return x_hat


class InpaintPlus(nn.Module):
    def __init__(self):
        super(InpaintPlus, self).__init__()
        nf = 64
        self.conv1 = nn.Conv2d(4, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = F.avg_pool2d(h, 2)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.conv5(h)
        h = F.interpolate(h, scale_factor=2)
        x_hat = self.conv6(h)
        return x_hat

########################################################################################################################

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


if __name__ == '__main__':
    net = Inpaint()
    _input = torch.rand(1, 4, 128, 128)
    out = net(_input)
    n_params = count_parameters(net)
    print('number of trainable parameters * 4 =', n_params * 4)

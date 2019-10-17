import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out


class CenterBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation=2):
        super(CenterBlock, self).__init__()
        # self.maxpool = nn.MaxPool2d((2, 2))
        self.pad = nn.ReflectionPad2d(dilation)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.maxpool(x)
        out = self.pad(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out) # 이거 없애는 것도 시도.

        return out


class DecoderBlock(nn.Module):
    def __init__(self, inplanes, planes, outplanes, se_block=False):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se_block = se_block
        self.squeeze_excitation = SqueezeExcitation(outplanes)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.se_block:
            x = self.squeeze_excitation(x)

        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, planes, reduction=8):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(planes, planes // reduction, bias=False)
        self.fc2 = nn.Linear(planes // reduction, planes, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avgpool(x)
        y = y.permute(0, 2, 3, 1)

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.permute(0, 3, 1, 2)

        return x * y


class DilatedUnetResnet(nn.Module):
    def __init__(self, res_n=18, inplanes=32, center_block=True, se_block=True):
        super(DilatedUnetResnet, self).__init__()

        if res_n < 50:
            block = BasicBlock
        else:
            block = Bottleneck

        resnet_spec = {18: [1, 1, 1, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3],
                       152: [3, 8, 36, 3]}
        layers = resnet_spec[res_n]

        self.inplanes = inplanes
        self.center_block = center_block
        self.se_block = se_block

        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = self._make_layer(block, layers[0], self.inplanes, self.inplanes*2, stride=2, se_block=se_block)
        self.encoder2 = self._make_layer(block, layers[1], self.inplanes*2, self.inplanes*4, stride=2, se_block=se_block)
        # self.encoder3 = self._make_layer(block, layers[2], self.inplanes * 2, self.inplanes * 4, stride=2, se_block=se_block)
        # self.encoder4 = self._make_layer(block, layers[3], self.inplanes * 4, self.inplanes * 8, stride=2, se_block=se_block)

        self.center = CenterBlock(self.inplanes * 4, self.inplanes * 4, dilation=2)
        self.center2 = CenterBlock(self.inplanes * 4, self.inplanes * 4, dilation=2)

        # self.decoder4 = DecoderBlock(self.inplanes * 8 + self.inplanes * 4, self.inplanes * 8, self.inplanes * 8, se_block=se_block)
        # self.decoder3 = DecoderBlock(self.inplanes * 8 + self.inplanes * 2, self.inplanes * 4, self.inplanes * 4, se_block=se_block)
        self.decoder2 = DecoderBlock(self.inplanes * 4 + self.inplanes * 2, self.inplanes * 4, self.inplanes * 2, se_block=se_block)
        self.decoder1 = DecoderBlock(self.inplanes * 2 + self.inplanes * 1, self.inplanes * 2, self.inplanes * 1, se_block=se_block)

        self.logit = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            self.relu,
            nn.Conv2d(self.inplanes, 3, kernel_size=1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        zero_init_residual = True
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, num_blocks, inplanes, planes, stride=1, se_block=False):
        layers = []
        layers.append(block(inplanes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes, stride=1))
        if se_block:
            layers.append(SqueezeExcitation(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_0 = x

        # x = self.maxpool(x)
        x = self.encoder1(x)
        skip_1 = x

        x = self.encoder2(x)
        # skip_2 = x

        # x = self.encoder3(x)
        # skip_3 = x

        # x = self.encoder4(x)
        x = self.center(x)
        x = self.center2(x)

        # x = self.decoder4(x, skip_3)
        # x = self.decoder3(x, skip_2)
        x = self.decoder2(x, skip_1)
        x = self.decoder1(x, skip_0)

        x = self.logit(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, bias=False):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=bias),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=bias),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class FuckUNet(nn.Module):
    def __init__(self, ch=64, residual_blocks=2, bias=True):
        super(FuckUNet, self).__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=ch, kernel_size=3, stride=2, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch, track_running_stats=False),
                                  nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=ch*2, kernel_size=3, stride=2, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch*2, track_running_stats=False),
                                  nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(in_channels=ch*2, out_channels=ch*2, kernel_size=3, stride=2, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch*2, track_running_stats=False),
                                  nn.ReLU(True))

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(ch*2, 2, bias)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.dec3 = nn.Sequential(nn.Conv2d(in_channels=ch*2+ch*2, out_channels=ch*2, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch*2, track_running_stats=False),
                                  nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv2d(in_channels=ch*2+ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch, track_running_stats=False),
                                  nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Conv2d(in_channels=ch+4, out_channels=ch//2, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch, track_running_stats=False),
                                  nn.ReLU(True))

        self.logit = nn.Sequential(nn.Conv2d(in_channels=ch//2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=bias),
                                   nn.ReLU(True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)  # ch
        e2 = self.enc2(e1)  # ch*2
        e3 = self.enc3(e2)

        out = self.middle(e3)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = torch.cat([out, e2], dim=1)
        out = self.dec3(out)

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = torch.cat([out, e1], dim=1)
        out = self.dec2(out)

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = torch.cat([out, x], dim=1)
        out = self.dec1(out)

        out = self.logit(out)

        return out


class FuckNet(nn.Module):
    def __init__(self, ch=64, residual_blocks=2, bias=True):
        super(FuckNet, self).__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=ch, kernel_size=3, stride=2, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch, track_running_stats=False),
                                  nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=ch*2, kernel_size=3, stride=2, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch*2, track_running_stats=False),
                                  nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(in_channels=ch*2, out_channels=ch*2, kernel_size=3, stride=2, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch*2, track_running_stats=False),
                                  nn.ReLU(True))

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(ch*2, 2, bias)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.dec3 = nn.Sequential(nn.Conv2d(in_channels=ch*2, out_channels=ch*2, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch*2, track_running_stats=False),
                                  nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv2d(in_channels=ch*2, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch, track_running_stats=False),
                                  nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.InstanceNorm2d(ch, track_running_stats=False),
                                  nn.ReLU(True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)  # ch
        e2 = self.enc2(e1)  # ch*2
        e3 = self.enc3(e2)

        out = self.middle(e3)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.dec3(out)

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.dec2(out)

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.dec1(out)

        return out
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
    # net = DilatedUnetResnet()
    net = FuckNet()
    _input = torch.rand(1, 4, 128, 128)
    out = net(_input)
    n_params = count_parameters(net)
    print('number of trainable parameters * 4 =', n_params * 4)

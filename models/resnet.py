"""
ResNet building block
"""

import torch.nn as nn

from ..utils.weight_utils import kaiming_init


class ResNet32x32(nn.Module):

    def __init__(self, block, layers, num_classes=10, weight_init=kaiming_init):
        self.inplanes = 64
        super(ResNet32x32, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(weight_init)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""
ResNet basic block for GAN
"""


def conv3x3(in_planes, out_planes, weight_norm=None):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
    if weight_norm:
        conv = weight_norm(conv)
    return conv


class ResBlockGAN(nn.Module):
    def __init__(self, in_plane, planes, resample=None, weight_norm=None):
        super(ResBlockGAN, self).__init__()
        assert resample in [None, 'up', 'down']
        self.conv1 = conv3x3(in_plane, planes, weight_norm=weight_norm)
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.conv2 = conv3x3(planes, planes, weight_norm=weight_norm)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.resample = resample
        self.conv3 = conv3x3(in_plane, planes, weight_norm=weight_norm)
        if self.resample == 'up':
            self.pool = nn.Upsample(scale_factor=2, mode='nearest')
        elif self.resample == 'down':
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x
        out = x

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        if self.resample == 'up':
            out = self.pool(out)

        out = self.bn2(out)
        out = self.conv2(out)

        if self.resample == 'down':
            out = self.pool(out)

        if self.resample is not None:
            residual = self.conv3(residual)
            residual = self.pool(residual)
        else:
            residual = self.conv3(residual)

        out += residual

        return out

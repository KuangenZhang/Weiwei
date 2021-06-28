import torch.nn as nn
import torch 
import functools
import torch.nn.functional as F


##############################################################################
# Classes
##############################################################################

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        '''2D CNN'''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(2, 2))
        self.bn5 = nn.BatchNorm2d(1)
        # self.conv6 = nn.Conv2d(16, 1, kernel_size=(1, 1))
        # self.bn6 = nn.BatchNorm2d(1)


    def forward(self, x):
        '''2D CNN'''
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = F.relu6(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        return x


class ModifiedModel(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ndf, n_layers, original_model, norm_layer, fc_relu_slope, fc_drop_out):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ModifiedModel, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        sequence = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 32)
            sequence += [
                nn.Linear(in_features=int(ndf/nf_mult_prev), out_features=int(ndf/nf_mult)),
                norm_layer(int(ndf/nf_mult)),
                nn.LeakyReLU(fc_relu_slope, True),
                nn.Dropout2d(p=fc_drop_out)
            ]

        sequence += [nn.Linear(in_features=int(ndf/nf_mult), out_features=1)]  # output 1 channel prediction map
        self.linear_group = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        x = self.features(input)    
        x = self.avg(x)
        x = x.view(x.size(0), -1)    

        return self.linear_group(x)

class ModifiedModel_old(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, ndf, n_layers, original_model, norm_layer, fc_relu_slope, fc_drop_out):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ModifiedModel_old, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        sequence = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 32)
            sequence += [
                nn.Linear(in_features=int(ndf/nf_mult_prev), out_features=int(ndf/nf_mult)),
                # norm_layer(int(ndf/nf_mult)),
                nn.LeakyReLU(fc_relu_slope, True),
                nn.Dropout2d(p=fc_drop_out)
            ]

        sequence += [nn.Linear(in_features=int(ndf/nf_mult), out_features=1)]  # output 1 channel prediction map
        self.linear_group = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        x = self.features(input)    
        x = self.avg(x)
        x = x.view(x.size(0), -1)    

        return self.linear_group(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm_layer, stride=1):
        super(BasicBlock, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=use_bias),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm_layer, stride=1):
        super(Bottleneck, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=use_bias)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=use_bias)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=use_bias)
        self.bn3 =norm_layer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=use_bias),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, norm_layer, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.bn1 = norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm_layer, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm_layer, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm_layer, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm_layer, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, norm_layer, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm_layer, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(norm_layer):
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)


def ResNet34(norm_layer):
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer)


def ResNet50(norm_layer):
    return ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer)


def ResNet101(norm_layer):
    return ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer)


def ResNet152(norm_layer):
    return ResNet(Bottleneck, [3, 8, 36, 3], norm_layer=norm_layer)
 

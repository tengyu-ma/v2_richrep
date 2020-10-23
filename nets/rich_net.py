import torch
import torch.nn as nn
import torchvision.models as models


net_candidates = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'inception_v3'
]


class RichNet(nn.Module):
    def __init__(self, nview_all, net_name, pretrained, mode, channel6):
        super().__init__()
        assert mode in ['sv', 'rich_max', 'rich_flatten', 'rich_flatten_extra'], ValueError(f'Invalid mode {mode}')

        self.pretrained = pretrained
        self.net_name = net_name
        self.mode = mode
        self.nview_all = nview_all
        self.channel6 = channel6

        if mode == 'rich_flatten_extra':
            self.feature, self.fc1, self.fc2 = self.get_classicnet(pretrained, net_name, channel6)
        else:
            self.feature, self.fc = self.get_classicnet(pretrained, net_name, channel6)

    def get_classicnet(self, pretrained, net_name, channel6=False):
        assert net_name in net_candidates, ValueError(f'Invalid net_name {net_name}')

        if 'inception' in net_name:
            net = getattr(models, net_name)(pretrained=pretrained, aux_logits=False)
        else:
            net = getattr(models, net_name)(pretrained=pretrained)

        if channel6:
            if 'resnet' in net_name or 'resnext' in net_name:
                net_conv1 = net.conv1
            elif 'inception' in net_name:
                net_conv1 = net.Conv2d_1a_3x3.conv
            elif 'vgg' in net_name:
                net_conv1 = net.features[0]
            else:
                raise ValueError(f'Invalid net name {net_name}')

            net_conv1_weight = torch.cat(2 * [net_conv1.weight], dim=1)
            channel6_conv1 = nn.Conv2d(
                6,
                net_conv1.out_channels,
                kernel_size=net_conv1.kernel_size,
                stride=net_conv1.stride,
                padding=net_conv1.padding,
                bias=False,
            )
            channel6_conv1.weight = nn.Parameter(net_conv1_weight)

            if 'resnet' in net_name or 'resnext' in net_name:
                net.conv1 = channel6_conv1
            elif 'inception' in net_name:
                net.Conv2d_1a_3x3.conv = channel6_conv1
            elif 'vgg' in net_name:
                net.features[0] = channel6_conv1
            else:
                raise ValueError(f'Invalid net name {net_name}')

        if 'vgg' in net_name:
            in_features = net.classifier[-1].in_features
        else:
            in_features = net.fc.in_features

        if self.mode == 'rich_flatten_extra':
            fc1 = nn.Linear(in_features * self.nview_all, 37)
            fc2 = nn.Linear(37, 40)
            return nn.Sequential(*list(net.children())[:-1]), fc1, fc2
        elif self.mode == 'rich_flatten':
            fc = nn.Linear(in_features * self.nview_all, 40)
        else:
            fc = nn.Linear(in_features, 40)

        return nn.Sequential(*list(net.children())[:-1]), fc

    def forward(self, x):
        x = self.feature(x)

        if self.mode == 'rich_max':
            n, c, h, w = x.shape
            bs = n // self.nview_all  # real batch size
            x = x.view([bs, self.nview_all, c, h, w])
            x = torch.max(x, 1)[0].view(bs, -1)
            y = self.fc(x)
            return y
        if self.mode == 'rich_flatten':
            n, c, h, w = x.shape
            x.reshape(n * c, h, w)
            bs = n // self.nview_all  # real batch size
            x = x.view([bs, self.nview_all * c * h * w])
            y = self.fc(x)
            return y
        if self.mode == 'rich_flatten_extra':
            n, c, h, w = x.shape
            x.reshape(n * c, h, w)
            bs = n // self.nview_all  # real batch size
            x = x.view([bs, self.nview_all * c * h * w])

            y_low = self.fc1(x)  # 37 lower class as to combine (flower_pot, vase, plant) and (desk and table)
            y_high = self.fc2(y_low)  # the original 40 class
            return y_high
        else:
            y = self.fc(x)
            return y

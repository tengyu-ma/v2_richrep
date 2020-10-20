import torch
import torch.nn as nn
import torchvision.models as models


net_candidates = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'inception_v3'
]


class RichNet(nn.Module):
    def __init__(self, nview_all, net_name, pretrained, mode):
        super().__init__()
        assert mode in ['sv', 'rich_max', 'rich_flatten'], ValueError(f'Invalid mode {mode}')

        self.pretrained = pretrained
        self.net_name = net_name
        self.mode = mode
        self.nview_all = nview_all

        self.feature, self.fc = self.get_classicnet(pretrained, net_name)

    def get_classicnet(self, pretrained, net_name):
        assert net_name in net_candidates, ValueError(f'Invalid net_name {net_name}')

        if 'inception' in net_name:
            net = getattr(models, net_name)(pretrained=pretrained, aux_logits=False)
        else:
            net = getattr(models, net_name)(pretrained=pretrained)

        if 'vgg' in net_name:
            if self.mode == 'rich_flatten_extra':
                in_features = net.classifier[-1].in_features
                net.classifier[-1] = nn.Sequential(
                    nn.Linear(in_features * self.nview_all, in_features),
                    nn.Linear(in_features, 40)
                )
            else:
                in_features = net.classifier[-1].in_features
                if self.mode == 'rich_flatten':
                    in_features *= self.nview_all
                net.classifier[-1] = nn.Linear(in_features, 40)
        else:
            if self.mode == 'rich_flatten_extra':
                in_features = net.fc.in_features
                net.fc = nn.Sequential(
                    nn.Linear(in_features * self.nview_all, in_features),
                    nn.Linear(in_features, 40)
                )
            else:
                in_features = net.fc.in_features
                if self.mode == 'rich_flatten':
                    in_features *= self.nview_all
                net.fc = nn.Linear(in_features, 40)

        return nn.Sequential(*list(net.children())[:-1]), net.fc

    def forward(self, x):
        x = self.feature(x)

        if self.mode == 'rich_max':
            n, c, h, w = x.shape
            bs = n // self.nview_all  # real batch size
            x = x.view([bs, self.nview_all, c, h, w])
            x = torch.max(x, 1)[0].view(bs, -1)
            y = self.fc(x)
        if self.mode == 'rich_flatten':
            n, c, h, w = x.shape
            x.reshape(n * c, h, w)
            bs = n // self.nview_all  # real batch size
            x = x.view([bs, self.nview_all * c * h * w])
            y = self.fc(x)
        else:
            y = self.fc(x)

        return y

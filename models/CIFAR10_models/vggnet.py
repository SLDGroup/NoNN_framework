import torch
import torch.nn as nn
from torch.autograd import Variable


def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert (depth in depth_lst), "Error : VGGnet depth should be either 11, 13, 16, 19"
    cf_dict = {
        '11': [
            64, 'mp',
            128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'],
        '13': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'
        ],
        '16': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 'mp',
            512, 512, 512, 'mp',
            512, 512, 512, 'mp'
        ],
        '19': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 256, 'mp',
            512, 512, 512, 512, 'mp',
            512, 512, 512, 512, 'mp'
        ],
    }

    return cf_dict[str(depth)]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class VGG(nn.Module):
    def __init__(self, depth, num_classes, mode="train"):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg(depth))
        self.linear = nn.Linear(512, num_classes)
        self.mode = mode

    def forward(self, x):
        layer_out = self.features(x)
        # print(f"#################SHAPE L1: {layer_1_out.shape}")
        out = layer_out.view(layer_out.size(0), -1)
        # print(f"#################SHAPE OUT L1: {out.shape}")
        out = self.linear(out)
        # print(f"#################SHAPE OUT: {out.shape}")

        if self.mode == "train":
            return out, layer_out
        elif self.mode == "deployment_files":
            return out
        else:
            print("ERROR: Wrong student model mode. Choose form [train, deployment_files]")
            exit(-1)

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3

        for x in cfg:
            if x == 'mp':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes, x), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = VGG(11, 10)
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"parameters 11 = {n_parameters}")
    # net = VGG(13, 10)
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"parameters 13 = {n_parameters}")
    # net = VGG(16, 10)
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"parameters  16 = {n_parameters}")
    # net = VGG(19, 10)
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"parameters 19 = {n_parameters}")
    # parameters VGG11 = 9 231 114
    # parameters VGG13 = 9 416 010
    # parameters VGG16 = 14 728 266
    # parameters VGG19 = 20 040 522
    net(torch.randn(1, 3, 32, 32))

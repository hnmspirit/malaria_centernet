import torch
import torch.nn as nn
import torch.nn.functional as F

from .res import res18_enc, res34_enc, res_inchannels
from .mobi2 import mobi2_enc, mobi2_inchannels
from .mobi3 import mobi3_enc, mobi3_inchannels
from .mnas import mnas_enc, mnas_inchannels
from .effi import effi0_enc, effi0_inchannels


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UpPyramid(nn.Module):
    def __init__(self, encoder, in_channels, heads, head_conv=64, hidden_channels=[256,128,64], bias=False):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

        self.in5 = nn.Conv2d(in_channels[-1], hidden_channels[0], 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], hidden_channels[0], 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], hidden_channels[1], 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], hidden_channels[2], 1, bias=bias)

        self.out4 = nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, padding=1, bias=bias)
        self.out3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], 3, padding=1, bias=bias)
        self.out2 = nn.Conv2d(hidden_channels[2], 64, 3, padding=1, bias=bias)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)

        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

        for head in self.heads:
            channels = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, channels, kernel_size=1, padding=0, bias=True)
                )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, channels, kernel_size=1, padding=0, bias=True)
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # nn.init.kaiming_normal_(m.weight.data)
            nn.init.normal_(m.weight.data, std=0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x):
        x2, x3, x4, x5 = self.encoder(x)

        in5 = self.in5(x5)
        in4 = self.in4(x4)
        in3 = self.in3(x3)
        in2 = self.in2(x2)

        out4 = F.interpolate(in5, scale_factor=2) + in4
        out4 = self.out4(out4)

        out3 = F.interpolate(out4, scale_factor=2) + in3
        out3 = self.out3(out3)

        out2 = F.interpolate(out3, scale_factor=2) + in2
        out2 = self.out2(out2)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(out2)
        return [ret]


def res18_prm(heads, head_conv=64):
    encoder = res18_enc()
    in_channels = res_inchannels
    net = UpPyramid(encoder, in_channels, heads, head_conv)
    return net

def res34_prm(heads, head_conv=64):
    encoder = res34_enc()
    in_channels = res_inchannels
    net = UpPyramid(encoder, in_channels, heads, head_conv)
    return net

def mobi2_prm(heads, head_conv=64):
    encoder = mobi2_enc()
    in_channels = mobi2_inchannels
    net = UpPyramid(encoder, in_channels, heads, head_conv)
    return net

def mobi3_prm(heads, head_conv=32):
    encoder = mobi3_enc()
    in_channels = mobi3_inchannels
    net = UpPyramid(encoder, in_channels, heads, head_conv=32, hidden_channels=[128, 64, 32])
    return net

def mnas_prm(heads, head_conv=64):
    encoder = mnas_enc()
    in_channels = mnas_inchannels
    net = UpPyramid(encoder, in_channels, heads, head_conv)
    return net

def effi0_prm(heads, head_conv=64):
    encoder = effi0_enc()
    in_channels = effi0_inchannels
    net = UpPyramid(encoder, in_channels, heads, head_conv)
    return net

import math
import torch
import torch.nn as nn
import torchvision.ops as ops

from .res import res18_enc, res_inchannels

BN_MOMENTUM = 0.1

class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, groups=1, offset_groups=1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv2d_offset = nn.Conv2d(in_channels, offset_channels * offset_groups,
            kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
        )
        self.conv2d = ops.DeformConv2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=dilation,
            dilation=dilation, groups=groups, bias=False
        )

    def forward(self, x):
        offset = self.conv2d_offset(x)
        return self.conv2d(x, offset)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UpDCN(nn.Module):

    def __init__(self, encoder, inplanes, heads, head_conv):
        super().__init__()
        self.encoder = encoder
        self.heads = heads
        self.inplanes = inplanes
        self.deconv_with_bias = False

        self.decoders = self._make_deconv_layer(
            3, [256, 128, 64], [4, 4, 4]
        )
        self.init_weights_deconv()

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

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
                    kernel_size=3, stride=1,
                    dilation=1, groups=1)
            # fill_fc_weights(fc)

            up = nn.ConvTranspose2d(planes, planes,
                    kernel_size=kernel, stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, _, x = self.encoder(x)
        x = self.decoders(x)

        # return x
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights_deconv(self):
        for name, m in self.decoders.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def res18_dcn(heads, head_conv=64):
    encoder = res18_enc()
    inplanes = res_inchannels[-1]
    net = UpDCN(encoder, inplanes, heads, head_conv)
    return net

from torch.utils import model_zoo
from ..backbones import mobilenetv3


class Mobi3Enc(mobilenetv3.MobileNetV3):
    def __init__(self, cfgs, mode='large', width_mult=1., outputs=[16], url=None):
        self.url = url
        super().__init__(cfgs, mode, width_mult=width_mult)
        self.outputs = outputs
        del self.conv
        del self.avgpool
        del self.classifier

    def initialize(self):
        self.load_state_dict(model_zoo.load_url(self.url), strict=False)

    def forward(self, x):
        outputs = []
        for idx, feat in enumerate(self.features):
            x = feat(x)
            if idx in self.outputs:
                outputs.append(x)
        return outputs


def mobi3_enc():
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    mode = 'large'
    width_mult = 1.
    outputs = [3, 6, 12, 15]
    url = 'https://github.com/d-li14/mobilenetv3.pytorch/raw/master/pretrained/mobilenetv3-large-1cd25616.pth'
    encoder = Mobi3Enc(cfgs, mode, width_mult, outputs, url)
    encoder.initialize()
    return encoder

mobi3_inchannels = [24, 40, 112, 160]
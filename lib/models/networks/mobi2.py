from torch import nn
from ..backbones import mobilenetv2
from torch.utils import model_zoo


class Mobi2Enc(mobilenetv2.MobileNetV2):
    def __init__(self, outputs=[17], url=None):
        self.url = url
        super().__init__()
        self.outputs = outputs
        del self.classifier

    def initialize(self):
        self.load_state_dict(model_zoo.load_url(self.url), strict=False)

    def forward(self, x):
        outputs = []
        for idx, feat in enumerate(self.features[:-1]):
            x = feat(x)
            if idx in self.outputs:
                outputs.append(x)
        return outputs


def mobi2_enc():
    url = mobilenetv2.model_urls['mobilenet_v2']
    encoder = Mobi2Enc(outputs=[3, 6, 13, 17], url=url)
    encoder.initialize()
    return encoder

mobi2_inchannels = [24, 32, 96, 320]
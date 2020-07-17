from ..backbones import mnasnet
from torch.utils import model_zoo


class MnasEnc(mnasnet.MNASNet):
    def __init__(self, outputs, url):
        self.url = url
        super().__init__(alpha=1.0)
        self.outputs = outputs

        del self.layers[-1]
        del self.layers[-1]
        del self.layers[-1]
        del self.classifier

    def initialize(self):
        self.load_state_dict(model_zoo.load_url(self.url), strict=False)

    def forward(self, x):
        outputs = []
        for idx, module in enumerate(self.layers):
            x = module(x)
            if idx in self.outputs:
                outputs.append(x)
        return outputs


def mnas_enc():
    url = 'https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth'
    encoder = MnasEnc(outputs=[8, 9, 11, 13], url=url)
    encoder.initialize()
    return encoder

mnas_inchannels = [24, 40, 96, 320]
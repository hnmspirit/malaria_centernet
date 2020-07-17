from torch import nn
from torchvision.models import resnet
from torch.utils import model_zoo


class ResEnc(resnet.ResNet):
    def __init__(self, block, layers, url=None):
        self.url = url
        super().__init__(block, layers)
        del self.avgpool
        del self.fc

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url), strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


def res18_enc():
    encoder = ResEnc(resnet.BasicBlock, [2, 2, 2, 2], resnet.model_urls['resnet18'])
    encoder.initialize()
    return encoder

def res34_enc():
    encoder = ResEnc(resnet.BasicBlock, [3, 4, 6, 3], resnet.model_urls['resnet34'])
    encoder.initialize()
    return encoder

res_inchannels = [64, 128, 256, 512]
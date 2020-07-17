from ..backbones import effinet


class EffiNetE(effinet.EfficientNet):
    def __init__(self, outputs=[16], blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
        self.outputs = outputs

        del self._conv_head
        del self._bn1
        del self._avg_pooling
        del self._dropout
        del self._fc

    def forward(self, inputs):
        outputs = []
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.outputs:
                outputs.append(x)
        return outputs

    @classmethod
    def from_name(cls, model_name, outputs, in_channels=3, **override_params):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = effinet.get_model_params(model_name, override_params)
        model = cls(outputs, blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, outputs, in_channels=3, **override_params):
        model = cls.from_name(model_name, outputs, **override_params)
        effinet.load_pretrained_weights(model, model_name)
        model._change_in_channels(in_channels)
        return model


def effi0_enc():
    outputs = [2, 4, 10, 15]
    encoder = EffiNetE.from_pretrained('efficientnet-b0', outputs)
    return encoder


effi0_inchannels = [24, 40, 112, 320]
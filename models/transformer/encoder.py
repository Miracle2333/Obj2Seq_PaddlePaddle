import paddle
from .attention_modules import DeformableEncoderLayer, _get_clones


class TransformerEncoder(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        encoder_layer = DeformableEncoderLayer(args.ENCODER_LAYER)
        self.encoder_layers = _get_clones(encoder_layer, args.enc_layers)

    def forward(self, tgt, *args, **kwargs):
        for layer in self.encoder_layers:
            tgt = layer(tgt, *args, **kwargs)
        return tgt


def build_encoder(args):
    return TransformerEncoder(args)

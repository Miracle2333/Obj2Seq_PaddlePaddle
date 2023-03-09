from .transformer import Transformer


def build_transformer(args):
    return Transformer(args=args)

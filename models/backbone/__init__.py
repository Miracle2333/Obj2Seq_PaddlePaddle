from .resnet import build_backbone as build_resnet
from .swin_transformer import build_backbone as build_swin_transformer


def build_backbone(args):
    if args.backbone.startswith('resnet'):
        args.defrost()
        args.RESNET.train_backbone = args.train_backbone
        args.RESNET.num_feature_levels = args.num_feature_levels
        args.freeze()
        return build_resnet(args.RESNET)
    elif 'swin' in args.backbone:
        args.defrost()
        args.SWIN.train_backbone = args.train_backbone
        args.SWIN.num_feature_levels = args.num_feature_levels
        args.freeze()
        return build_swin_transformer(args.SWIN)
    else:
        raise NotImplemented

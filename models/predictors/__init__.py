from .separate_detect_head import SeparateDetectHead
from .separate_keypoint_head import SeparateKeypointHead
from .unified_seq_head import UnifiedSeqHead


def build_detect_predictor(args):
    if args.type == 'SeparateDetectHead':
        return SeparateDetectHead(args)
    elif args.type == 'SeparateKeypointHead':
        return SeparateKeypointHead(args)
    elif args.type == 'SeqHead':
        return UnifiedSeqHead(args)

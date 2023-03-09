from .losses import sigmoid_focal_loss
from .asl_losses import AsymmetricLoss, AsymmetricLossOptimized


def build_asymmetricloss(args):
    lossClass = (AsymmetricLossOptimized if args.asl_optimized else
        AsymmetricLoss)
    return lossClass(gamma_neg=args.asl_gamma_neg, gamma_pos=args.
        asl_gamma_pos, clip=args.asl_clip, disable_torch_grad_focal_loss=True)

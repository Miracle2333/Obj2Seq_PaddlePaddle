import paddle
"""
Backbone modules.
"""
from collections import OrderedDict
from typing import Dict, List
from util.misc import NestedTensor, is_main_process


class FrozenBatchNorm2d(paddle.nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-05):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', paddle.ones(shape=[n]))
        self.register_buffer('bias', paddle.zeros(shape=[n]))
        self.register_buffer('running_mean', paddle.zeros(shape=[n]))
        self.register_buffer('running_var', paddle.ones(shape=[n]))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)

    def forward(self, x):
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        w = self.weight.reshape(1, -1, 1, 1)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        b = self.bias.reshape(1, -1, 1, 1)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        rv = self.running_var.reshape(1, -1, 1, 1)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(paddle.nn.Layer):

    def __init__(self, backbone: paddle.nn.Layer, train_backbone: bool,
        return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (not train_backbone or 'layer2' not in name and 'layer3' not in
                name and 'layer4' not in name):
                """Class Method: *.requires_grad_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': '0'}
            self.strides = [32]
            self.num_channels = [2048]
>>>        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone,
            return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            mask = torch.nn.functional.interpolate(m[None].float(), size=x.
                shape[-2:]).to('bool')[0]
            out.append(NestedTensor(x, mask))
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, args):
        name = args.name
        pretrained = is_main_process() and args.pretrained
        train_backbone = args.train_backbone
        return_interm_layers = args.num_feature_levels > 1
        dilation = args.dilation
        norm_layer = NORM_DICT[args.norm_layer]
>>>        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'
            ), 'number of channels are hard coded'
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


>>>NORM_DICT = {'FrozenBN': FrozenBatchNorm2d, 'BN': torch.nn.BatchNorm2d}


def build_backbone(args):
    backbone = Backbone(args)
    return backbone

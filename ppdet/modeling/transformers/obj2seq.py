# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Anchor DETR (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Main framework for Obj2Seq
"""
import os
import paddle
import paddle.nn.functional as F
from paddle import nn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from .meta_arch import BaseArch

from ppdet.core.workspace import register, create
#from ..layers import MultiHeadAttention
#from .position_encoding import PositionEmbedding
#from .utils import _get_clones, get_valid_ratio
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

__all__ = ['OBJ2SEQ']

#from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                       accuracy, get_world_size, interpolate,
#                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .models import build_transformer
from .initializer import xavier_uniform_, constant_


@register
class OBJ2SEQ(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 with_mask=False,
                 exclude_post_process=False):
        super(OBJ2SEQ, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        output = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()


class Obj2Seq_old(nn.Layer):
    """ This is Obj2Seq, our main framework """

    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            args: refers _C.MODEL in config.yaml.
        """
        super(Obj2Seq, self).__init__()
        self.backbone = build_backbone(args.BACKBONE)
        self.transformer = build_transformer(args)
        num_feature_levels=args.BACKBONE.num_feature_levels
        hidden_dim = self.transformer.d_model

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2D(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2D(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.LayerList(input_proj_list)
        else:
            self.input_proj = nn.LayerList([
                nn.Sequential(
                    nn.Conv2D(self.backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        for proj in self.input_proj:
            xavier_uniform_(proj[0].weight, gain=1)
            constant_(proj[0].bias, 0)

    def forward(self, samples, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        #if not isinstance(samples, NestedTensor):
        #    samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].cast("float32"), size=src.shape[-2:])[0]
                srcs.append(src)
                masks.append(mask)

        outputs, loss_dict = self.transformer(srcs, masks, targets=targets)
        return outputs, loss_dict


def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path


def build_model(args):
    model = Obj2Seq(args.MODEL)

    # load-pretrained
    if args.MODEL.pretrained:
        weight = args.MODEL.pretrained
        path = _strip_postfix(weight)
        pdparam_path = path + '.pdparams'
        if not os.path.exists(pdparam_path):
            raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))
        pretrained_dict = paddle.load(args.MODEL.pretrained, map_location="cpu")
        param_state_dict = paddle.load(pdparam_path)

        if "model" in param_state_dict:
            param_state_dict = param_state_dict["model"]
        model_dict = model.state_dict()
        model_weight = {}
        incorrect_keys = 0
        for key in model_dict.keys():
            if key in param_state_dict.keys():
                model_weight[key] = param_state_dict[key]
            else:
                print('Unmatched key: {}'.format(key))
                incorrect_keys += 1
        assert incorrect_keys == 0, "Load weight {} incorrectly, \
                {} keys unmatched, please check again.".format(weight,
                                                               incorrect_keys)
        print('Finish resuming model weights: {}'.format(pdparam_path))
        model.set_dict(model_weight)

       
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    if len(args.MODEL.fixed_params) > 0:
        for n, p in model.named_parameters():
            if match_name_keywords(n, args.MODEL.fixed_params):
                # p.requires_grad_(False)
                p.stop_gradient = True
    return model
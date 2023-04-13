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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import paddle
import paddle.nn.functional as F
from paddle import nn


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
#from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

__all__ = ['OBJ2SEQ']


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

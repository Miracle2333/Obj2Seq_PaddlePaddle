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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .meta_arch import BaseArch

from ppdet.core.workspace import register, create


__all__ = ['OBJ2SEQ']

@register
class OBJ2SEQ(BaseArch):
    __category__ = 'architecture'
    __inject__ = []
    __shared__ = ['with_mask']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 post_process=None,
                 with_mask=False,
                 exclude_post_process=False):
        super(OBJ2SEQ, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        postprocessor = create(cfg['post_process'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            'post_process': postprocessor
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        output = self.transformer(body_feats, pad_mask, self.inputs)
        # DETR Head
        preds, loss_dict = output
        losses_total = sum(loss_dict[k] for k in loss_dict.keys())
        loss_dict['loss'] = losses_total

        if self.training:
            return loss_dict
        else:
            bbox, bbox_num = self.post_process(preds, self.inputs['orig_size'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}
            return output


    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

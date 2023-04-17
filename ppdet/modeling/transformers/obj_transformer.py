# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tkinter import N
from venv import create
from numpy import arange
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register
from .attention_layers import DeformableTransformerEncoderLayer,\
     DeformableTransformerEncoder, DeformableTransformerDecoderLayer,\
         DeformableTransformerDecoder
from .position_encoding import PositionEmbedding
from .utils import _get_clones, get_valid_ratio
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_
from .prompt_indicator import PromptIndicator
from .predictors import build_detect_predictor

#from .models.object_decoder import ObjectDecoder

__all__ = ['Obj_Transformer']


@register
class Obj_Transformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=300,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 in_feats_channel=[512, 1024, 2048],
                 num_feature_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 lr_mult=0.1,
                 pe_temperature=10000,
                 pe_offset=-0.5,
                 model=None):
        super(Obj_Transformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(in_feats_channel) <= num_feature_levels
       
        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries


        args = model.MODEL
        self.spatial_prior = args.OBJECT_DECODER.spatial_prior
        # prompt_indicator
        if args.with_prompt_indicator:
            self.prompt_indicator = PromptIndicator(args.PROMPT_INDICATOR)
        else:
            self.prompt_indicator = None

        # object decoder
        #if args.with_object_decoder:
        #    self.object_decoder = ObjectDecoder(self.d_model, args=args.OBJECT_DECODER)
        #else:
        #    self.object_decoder = None
        self.num_layers = args.OBJECT_DECODER.num_layers
        num_decoder_layers = self.num_layers 
        self.num_queries = args.OBJECT_DECODER.num_query_position
        self.refine_reference_points = args.OBJECT_DECODER.refine_reference_points
        self.detect_head = build_detect_predictor(args.OBJECT_DECODER.HEAD)
        self.with_query_pos_embed = args.OBJECT_DECODER.with_query_pos_embed
        num_queries = self.num_queries
        if self.refine_reference_points:
            self.detect_head = _get_clones(self.detect_head, self.num_layers)
            # reset params
            for head in self.detect_head[1:]:
                head.reset_parameters_as_refine_head()
        else:
            self.detect_head = nn.ModuleList([self.detect_head for _ in range(self.num_layers)])



        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_encoder_points, lr_mult)
        self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                    num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_feature_levels, num_decoder_points)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(
            hidden_dim,
            2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.input_proj = nn.LayerList()
        for in_channels in in_feats_channel:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim)))
        in_channels = in_feats_channel[-1]
        for _ in range(num_feature_levels - len(in_feats_channel)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            temperature=pe_temperature,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=pe_offset,
            eps=1e-4)

        self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        normal_(self.tgt_embed.weight)
        normal_(self.query_pos_embed.weight)
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
         #input from args
        args = cfg['model'] 
        return {'in_feats_channel': [i.channels for i in input_shape],
                }

    def forward(self, src_feats, src_mask=None, inputs=None):
        srcs = []
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))
        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
            bs, _, h, w = paddle.shape(src)
            spatial_shapes.append(paddle.concat([h, w]))
            src = src.flatten(2).transpose([0, 2, 1])
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(src_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            pos_embed = self.position_embedding(mask).flatten(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[level]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = None if src_mask is None else paddle.concat(mask_flatten,
                                                                   1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [l, 2]
        spatial_shapes = paddle.to_tensor(
            paddle.stack(spatial_shapes).astype('int64'))
        # [l], 每一个level的起始index
        level_start_index = paddle.concat([
            paddle.zeros(
                [1], dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        # [b, l, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)

        #for indicator and object decoder
        #enc_kwargs = dict(spatial_shapes = spatial_shapes,
        #                  level_start_index = level_start_index,
        #                  reference_points = reference_points_enc,
        #                  pos = lvl_pos_embed_flatten)
        cls_kwargs = dict(src_level_start_index=level_start_index)
        obj_kwargs = dict(src_spatial_shapes=spatial_shapes,
                          src_level_start_index=level_start_index,
                          src_valid_ratios=valid_ratios)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              mask_flatten, lvl_pos_embed_flatten, valid_ratios)


        # decoder
        #hs = self.decoder(tgt, reference_points_input, memory, spatial_shapes,
        #                  level_start_index, mask_flatten, query_embed)

        #return (hs, memory, reference_points)

        outputs, loss_dict = {}, {}
        #targets = inputs[1]
        inputs, targets = self.generate_targets(inputs)
        self.targets = targets
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
       
        if self.prompt_indicator is not None:
            cls_outputs, cls_loss_dict = self.prompt_indicator(memory, mask_flatten, targets=targets, kwargs=cls_kwargs)
            outputs.update(cls_outputs)
            loss_dict.update(cls_loss_dict)
            additional_object_inputs = dict(
                bs_idx = outputs["bs_idx"] if "bs_idx" in outputs else None,
                cls_idx = outputs["cls_idx"] if "cls_idx" in outputs else None,
                class_vector = outputs["tgt_class"], # cs_all, d
                previous_logits = outputs["cls_label_logits"], # bs, 80
            )
        else:
            additional_object_inputs = {}

        obj_outputs = []
        obj_loss_dict = {}
        # prepare input for decoder
        
        tgt_object, query_embed, reference_points, predictor_kwargs, kwargs =\
             self.prepare_decoder_input(memory, additional_object_inputs, obj_kwargs)

        memory_spatial_shapes = kwargs.pop('src_spatial_shapes')
        memory_level_start_index = kwargs.pop('src_level_start_index')
        cs_batch = kwargs.pop('cs_batch', None)
        memory_valid_ratios = kwargs.pop('src_valid_ratios')
    
        dec_outputs = self.decoder(tgt_object, reference_points, memory, mask_flatten,\
            memory_spatial_shapes, memory_level_start_index, cs_batch, memory_valid_ratios, query_embed)
        

        for lid, tgt_object in enumerate(dec_outputs):
            if self.training or self.refine_reference_points or lid == self.num_layers - 1:
                predictor_kwargs["rearrange"] = not self.refine_reference_points
                # TODO: move arrange into prompt indicator
                layer_outputs, layer_loss = self.detect_head[lid](tgt_object, query_embed,\
                     reference_points, srcs=memory, src_padding_masks=mask_flatten, **predictor_kwargs)
                # refine_reference_points modify in detect head
                if self.refine_reference_points:
                    reference_points = layer_outputs["detection"]["pred_boxes"].clone().detach()
                obj_outputs.append(layer_outputs)
                if self.training:
                    for key in layer_loss:
                        obj_loss_dict[f"{key}_{lid}"] = layer_loss[key]
        obj_outputs = obj_outputs.pop()

        outputs.update(obj_outputs)
        loss_dict.update(obj_loss_dict)

        return outputs, loss_dict

    
    def prepare_decoder_input(self, srcs, additional_info, kwargs={}):
        #bs, _, c = memory.shape
        #query_embed = self.query_pos_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        #tgt = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        #reference_points = F.sigmoid(self.reference_points(query_embed))
        #reference_points_input = reference_points.unsqueeze(
        #    2) * valid_ratios.unsqueeze(1)
        class_vector = additional_info.pop("class_vector", None)
        previous_logits = additional_info.pop("previous_logits", None)

        bs = srcs.shape[0]
        bs_idx = additional_info.pop("bs_idx", paddle.arange(bs))
        cls_idx = additional_info.pop("cls_idx", paddle.zeros([bs], dtype=paddle.int32))

        # modify srcs
        cs_batch = [(bs_idx==i).sum().item() for i in range(bs)]
        cs_all = sum(cs_batch)
        # src is not modified, but in ops

        tgt_object = self.get_object_queries(class_vector, cls_idx, bs_idx)
        reference_points, query_pos_embed = self.get_reference_points(cs_all)

        # modify kwargs
        kwargs["src_valid_ratios"] = paddle.concat([
            vs.expand([cs ,-1, -1]) for cs, vs in zip(cs_batch, kwargs["src_valid_ratios"])
        ])
        kwargs["cs_batch"] = cs_batch

        # prepare kwargs for predictor
        predictor_kwargs = {}
        for key in kwargs:
            predictor_kwargs[key] = kwargs[key]
        predictor_kwargs["class_vector"] = class_vector
        predictor_kwargs["bs_idx"] = bs_idx
        predictor_kwargs["cls_idx"] = cls_idx
        predictor_kwargs["previous_logits"] = previous_logits
        predictor_kwargs["targets"] = self.targets

        return tgt_object, query_pos_embed, reference_points, predictor_kwargs, kwargs


    def get_object_queries(self, class_vector, cls_idx, bs_idx):
        #tgt = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        c = self.d_model
        if class_vector is None:
            cs_all = cls_idx.shape[0]
            tgt_object = self.tgt_embed.weight.reshape([1 , self.num_queries, c]).tile([cs_all, 1, 1])
        elif class_vector.dim() == 3:
            bs, cs, c = class_vector.shape
            tgt_pos = self.tgt_embed.weight.reshape([1 , self.num_queries, c]).tile([bs*cs, 1, 1])
            tgt_object = tgt_pos + class_vector.reshape([bs*cs, 1, c]) # bs*cs, nobj, c
        elif class_vector.dim() == 2:
            cs_all, c = class_vector.shape
            tgt_pos = self.tgt_embed.weight.reshape([1 , self.num_queries, c]).tile([cs_all, 1, 1])
            tgt_object = tgt_pos + class_vector.reshape([cs_all, 1, c]) # cs_all, nobj, c
        return tgt_object

    def get_reference_points(self, bs):
        #query_embed = self.query_pos_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        #reference_points = F.sigmoid(self.reference_points(query_embed))
        #reference_points_input = reference_points.unsqueeze(2) * valid_ratios.unsqueeze(1)
        # Generate srcs
        if self.spatial_prior == "learned":
            reference_points = self.query_pos_embed.weight.unsqueeze(0).tile([bs, 1, 1])
            query_embed = None
        elif self.spatial_prior == "grid":
            nx=ny=round(math.sqrt(self.num_queries))
            self.num_queries=nx*ny
            x = (paddle.arange(nx) + 0.5) / nx
            y = (paddle.arange(ny) + 0.5) / ny
            xy=paddle.meshgrid(x,y)
            reference_points=paddle.concat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
            reference_points = reference_points.unsqueeze(0).tile(bs, 1, 1)
            query_embed = None
        elif self.spatial_prior == "sigmoid":
            query_embed = self.query_pos_embed.weight.unsqueeze(0).expand([bs, -1, -1])
            reference_points = F.sigmoid(self.reference_points(query_embed))
            if not self.with_query_pos_embed:
                query_embed = None
        else:
            raise ValueError(f'unknown {self.spatial_prior} spatial prior')
        return reference_points, query_embed

    def generate_targets(self, inputs):
        targets = self.generateClassificationResults(inputs)
        targets = self.rearrangeByCls(targets)
        #for target_key in self.keep_keys:
        #    inputs.pop(target_key)

        return inputs, targets

    
    def generateClassificationResults(self, sample):
        num_cats = 80
        keep_keys = ["size", "orig_size", "image_id", "multi_label_onehot", "multi_label_weights", "force_sample_probs", "num_classes"]
        self.keep_keys = keep_keys
        targets = []
        
        for i, _ in enumerate(sample['gt_class']):
            target = {}
            target["num_classes"] = paddle.to_tensor(num_cats)
            target["labels"] = sample['gt_class'][i]
            target["boxes"] = sample['gt_bbox'][i]
            target["image_id"] = sample['im_id'][i]
            target["iscrowd"] = sample['is_crowd'][i]
            target["area"] = sample['gt_area'][i]
            target["size"] =  sample['im_shape'][i]
            target['orig_size'] = sample['orig_size'][i]
            multi_labels = target["labels"].unique()
            multi_label_onehot = paddle.zeros(paddle.to_tensor(num_cats))
            multi_label_onehot[multi_labels] = 1
            multi_label_weights = paddle.ones_like(paddle.to_tensor(multi_label_onehot))

            # filter crowd items
            keep = paddle.where(target["iscrowd"] == 0)
            #keep = target["iscrowd"][i] == 0
            fields = ["labels", "area", "iscrowd"]
            if "boxes" in target:
                fields.append("boxes")
            if "masks" in target:
                fields.append("masks")
            if "keypoints" in target:
                fields.append("keypoints")
            for field in fields:
                tmp = target[field]
                if field == "boxes":
                    tmp = tmp[keep[0]][:, 0, :]
                else:
                    tmp = tmp[keep] 
                target[field] = tmp

            sample_prob = paddle.zeros_like(multi_label_onehot) - 1
            sample_prob[target["labels"].unique()] = 1
            target["multi_label_onehot"] = multi_label_onehot
            target["multi_label_weights"] = multi_label_weights
            target["force_sample_probs"] = sample_prob
            targets.append(target)
        return targets
    

    def rearrangeByCls(self, samples):
        min_keypoints_train = 0
        new_targets = []

        for i, sample in enumerate(samples):
            new_target = {}
            sample["class_label"] = []
            sample["class_label"] = sample["labels"].unique()
            
            for icls in sample["labels"].unique():
                icls = icls.item()
                new_target[icls] = {}
                where = paddle.where(sample["labels"] == icls)
                tmp = sample["boxes"]
                new_target[icls]["boxes"] = tmp[where[0]][:, 0, :]
                
                if icls == 0 and "keypoints" in sample:
                    new_target[icls]["keypoints"] = sample["keypoints"][where[0]][:, 0, :]

            for key in self.keep_keys:
                new_target[key] = sample[key]
            new_targets.append(new_target)

        return new_targets

        

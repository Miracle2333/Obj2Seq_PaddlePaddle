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
from venv import create
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from .utils import _get_clones, get_valid_ratio
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FFN(nn.Layer):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu',\
         weight_attr=None, bias_attr=None, normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src




class MSDeformableAttention(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        self.value = None
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        try:
            # use cuda op
            from deformable_detr_ops import ms_deformable_attn
        except:
            # use paddle func
            from .utils import deformable_attention_core_func as ms_deformable_attn
        self.ms_deformable_attn_core = ms_deformable_attn

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def preprocess_value(self, input_flatten, input_padding_mask=None, cs_batch=None, bs_idx=None):
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            #value = value.masked_fill(input_padding_mask[..., None], float(0))
            input_padding_mask = input_padding_mask.astype(value.dtype).unsqueeze(-1)
            value *= input_padding_mask
        self.value = value.reshape([N, Len_in, self.num_heads, self.head_dim])
        if bs_idx is not None:
            self.value = self.value[bs_idx]
        elif cs_batch is not None:
            self.value = paddle.concat([
                v.expand([cs ,-1, -1, -1]) for cs, v in zip(cs_batch, self.value)
            ]) # cs_all, *, *, *

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None,
                cs_batch=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor(int64)): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        if self.value is None:
            bs, Len_v = value.shape[:2]
            assert int(value_spatial_shapes.prod(1).sum()) == Len_v

            value = self.value_proj(value)
            if value_mask is not None:
                value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
                value *= value_mask
            value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])

            if cs_batch is not None:
                value = paddle.concat([
                        v.expand([cs ,-1, -1, -1]) for cs, v in zip(cs_batch, value)
                    ]) # cs_all, *, *, *
                bs = value.shape[0]
        else:
            value = self.value
            assert (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum() == value.shape[1]

        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class DeformableTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                               n_points, lr_mult)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        # ffn
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation, weight_attr, bias_attr)
        #self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.activation = getattr(F, activation)
        #self.dropout2 = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(dim_feedforward, d_model)
        #self.dropout3 = nn.Dropout(dropout)
        #self.norm2 = nn.LayerNorm(
        #    d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        #self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src = self.ffn(src)
        return src

    def forward(self,
                src,
                reference_points,
                spatial_shapes,
                level_start_index,
                src_mask=None,
                query_pos_embed=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, query_pos_embed), reference_points, src,
            spatial_shapes, level_start_index, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(DeformableTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, offset=0.5):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(
                paddle.arange(end=H) + offset, paddle.arange(end=W) + offset)
            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] *
                                                    H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] *
                                                    W)
            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self,
                feat,
                spatial_shapes,
                level_start_index,
                feat_mask=None,
                query_pos_embed=None,
                valid_ratios=None):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [feat.shape[0], spatial_shapes.shape[0], 2])
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios)
        for layer in self.layers:
            feat = layer(feat, reference_points, spatial_shapes,
                         level_start_index, feat_mask, query_pos_embed)

        return feat


class MultiHeadDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadDecoderLayer, self).__init__()

        # self attention
        #self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        #self.dropout1 = nn.Dropout(dropout)
        #self.norm1 = nn.LayerNorm(
        #   d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # cross attention
        self.n_head = n_head
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # ffn
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation, weight_attr, bias_attr)
        #self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.activation = getattr(F, activation)
        #self.dropout3 = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(dim_feedforward, d_model)
        #self.dropout4 = nn.Dropout(dropout)
        #self.norm3 = nn.LayerNorm(
        #    d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        #self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt = self.ffn(tgt)
        return tgt

    def self_attn_forward(self, tgt, query_pos):
        if query_pos is not None and query_pos.shape[0] != tgt.shape[0]: # False
            cs = tgt.shape[0] // query_pos.shape[0]
            query_pos_self = query_pos.repeat_interleave(repeats=cs, axis=0)
        else:
            query_pos_self = query_pos
        #q = k = self.with_pos_embed(tgt, query_pos_self)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        #tgt2 = self.self_attn(q.transpose([1, 0, 2]), k.transpose([1, 0, 2]), tgt.transpose([1, 0, 2]))[0].transpose([1, 0, 2])
        q = k = self.with_pos_embed(tgt, query_pos_self)
        tgt2 = self.self_attn(q, k, value=tgt)
        return tgt2

    def cross_attn_forward(self, tgt, query_pos, srcs, src_padding_masks, posemb_2d=None):
        bs_all, seq, c = tgt.shape
        #srcs = kwargs["srcs"]
        bs = srcs.shape[0]
        if bs_all > bs:
            tgt = tgt.reshape([bs, -1, c])
            cs = bs_all // bs
        src_padding_masks = src_padding_masks
        posemb_2d = 0 if posemb_2d is None else posemb_2d
        query_pos = paddle.zeros_like(tgt) if query_pos is None else query_pos.tile([1, cs, 1])

        # tgt2 = self.self_attn(q, k, value=tgt)
        #issue no transpose and [0]
        seq_len = src_padding_masks.shape[-1]
        src_padding_masks =  src_padding_masks.reshape([bs, -1])
        src_padding_masks = src_padding_masks.reshape([bs, 1, 1, seq_len])
        src_padding_masks = src_padding_masks.expand([-1, self.n_head, -1, -1])

        tgt2 = self.cross_attn((tgt + query_pos),
                                (srcs + posemb_2d).reshape([bs, -1, c]),
                                srcs.reshape([bs, -1, c]), attn_mask=src_padding_masks)
        return tgt2.reshape([bs_all, seq, c])

    def forward(self,
                tgt,
                memory,
                memory_mask=None,
                query_pos_embed=None,
                ):

        # self attention
        ## no self-attention here
        #tgt2 = self.self_attn_forward(tgt, query_pos_embed)
        #tgt = tgt + self.dropout1(tgt2)
        #tgt = self.norm1(tgt)

        # cross attention
        # self.self_attn(q, k, value=tgt)
        #cross_attn_forward(self, tgt, query_pos, srcs, src_padding_masks, posemb_2d):
        tgt2 = self.cross_attn_forward(
            tgt, query_pos_embed, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableTransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points, lr_mult)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # ffn
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation, weight_attr, bias_attr)
        #self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.activation = getattr(F, activation)
        #self.dropout3 = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(dim_feedforward, d_model)
        #self.dropout4 = nn.Dropout(dropout)
        #self.norm3 = nn.LayerNorm(
        #    d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        #self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt = self.ffn(tgt)
        return tgt

    def self_attn_forward(self, tgt, query_pos):
        if query_pos is not None and query_pos.shape[0] != tgt.shape[0]: # False
            cs = tgt.shape[0] // query_pos.shape[0]
            query_pos_self = query_pos.repeat_interleave(repeats=cs, axis=0)
        else:
            query_pos_self = query_pos
        #q = k = self.with_pos_embed(tgt, query_pos_self)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        #tgt2 = self.self_attn(q.transpose([1, 0, 2]), k.transpose([1, 0, 2]), tgt.transpose([1, 0, 2]))[0].transpose([1, 0, 2])
        q = k = self.with_pos_embed(tgt, query_pos_self)
        tgt2 = self.self_attn(q, k, value=tgt)
        return tgt2

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask=None,
                memory_valid_ratios=None,
                query_pos_embed=None,
                cs_batch=None):

            # output = layer(output, reference_points, memory,
             #              memory_spatial_shapes, memory_level_start_index, memory_valid_ratio    

        if reference_points.shape[-1] == 4:
            memory_valid_ratios = paddle.concat([memory_valid_ratios, memory_valid_ratios], axis=-1)
        if memory_valid_ratios.shape[0] != reference_points.shape[0]:
            repeat_times = (reference_points.shape[0] // memory_valid_ratios.shape[0])
            memory_valid_ratios = memory_valid_ratios.repeat_interleave(repeat_times, axis=0)
        memory_valid_ratios = memory_valid_ratios[:, None] if reference_points.dim() == 3 else memory_valid_ratios[:, None, None]
        reference_points = reference_points[..., None, :] * memory_valid_ratios

        # self attention
        ## issue no transpose
        tgt2 = self.self_attn_forward(tgt, query_pos_embed)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask, cs_batch=cs_batch)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(DeformableTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate


    #self.decoder(tgt_object, reference_points, memory, mask_flatten, query_embed, **kwargs)
    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_mask=None,
                memory_spatial_shapes=None,
                memory_level_start_index=None,
                cs_batch=None,
                memory_valid_ratios=None,
                query_pos_embed=None,
                ):

                
        output = tgt
        intermediate = []
        for lid, layer in enumerate(self.layers):
            #forward(feat, query_pos, forward_reference_points, srcs, src_padding_masks, **kwargs)
            output = layer(output, reference_points, memory,
                           memory_spatial_shapes, memory_level_start_index, memory_mask, memory_valid_ratios,
                           query_pos_embed, cs_batch)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        #issue no unsqueeze
        #return output.unsqueeze(0)
        return output

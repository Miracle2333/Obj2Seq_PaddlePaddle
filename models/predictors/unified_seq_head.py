import paddle
import numpy as np
import math
from util.misc import inverse_sigmoid
from .classifiers import build_label_classifier
from .seq_postprocess import build_sequence_postprocess
from ..transformer.attention_modules import DeformableDecoderLayer
from models.ops.functions import MSDeformAttnFunction
from models.losses.classwise_criterion import ClasswiseCriterion


class Attention(paddle.nn.Layer):

    def __init__(self, dim, num_heads=8, dropout=0.0, proj=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = paddle.nn.Linear(in_features=dim, out_features=dim * 3)
        self.attn_drop = paddle.nn.Dropout(p=dropout)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim
            ) if proj else paddle.nn.Identity()

    def forward(self, x, pre_kv=None, attn_mask=None):
        N, B, C = x.shape
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        qkv = self.qkv(x).reshape(N, B, 3, self.num_heads, C // self.num_heads
            ).transpose(perm=[2, 1, 3, 0, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        if pre_kv is not None:
            k = paddle.concat(x=[pre_kv[0], k], axis=2)
            v = paddle.concat(x=[pre_kv[1], v], axis=2)
        pre_kv = paddle.stack(x=[k, v], axis=0)
        x = k
        perm_20 = list(range(x.ndim))
        perm_20[-2] = -1
        perm_20[-1] = -2
        attn = q @ x.transpose(perm=perm_20) * self.scale
        if attn_mask is not None:
            """Class Method: *.masked_fill_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            attn.masked_fill_(attn_mask, float('-inf'))
        """Class Method: *.softmax, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        x = (attn @ v).transpose(perm=[2, 0, 1, 3]).reshape(N, B, C)
        x = self.proj(x)
        return x, pre_kv


def update_reference_points_xy(output_signals, reference_points, id_step):
    if id_step < 2:
        new_reference_points = inverse_sigmoid(reference_points)
        new_reference_points[:, :, (id_step)] += output_signals[-1]
        new_reference_points = new_reference_points.sigmoid()
        return new_reference_points
    else:
        return reference_points


class UnifiedSeqHead(DeformableDecoderLayer):

    def __init__(self, args):
        super().__init__(args)
        if args.no_ffn:
            del self.ffn
            self.ffn = paddle.nn.Identity()
        if self.self_attn:
            del self.self_attn
            self.self_attn = Attention(self.d_model, self.n_heads, dropout=
                args.dropout, proj=args.self_attn_proj)
        self.classifier = build_label_classifier(args.CLASSIFIER)
        self.num_steps = args.num_steps
        self.output_embeds = paddle.nn.LayerList(sublayers=[MLP(self.
            d_model, self.d_model, c_out, 1) for c_out in [1] * self.num_steps]
            )
        self.reset_parameters_as_first_head()
        self.adjust_reference_points = update_reference_points_xy
        if args.pos_emb:
>>>            self.pos_emb = torch.nn.Embedding(self.num_steps, self.d_model)
>>>            timm.models.layers.trunc_normal_(self.pos_emb.weight, std=0.02)
        else:
            self.pos_emb = None
        self.post_process = build_sequence_postprocess(args)
        self.sg_previous_logits = args.sg_previous_logits
        self.combine_method = args.combine_method
        self.criterion = ClasswiseCriterion(args.LOSS)

    def reset_parameters_as_first_head(self):
        for i in range(self.num_steps):
>>>            torch.nn.init.constant_(self.output_embeds[i].layers[-1].weight
                .data, 0)
>>>            torch.nn.init.constant_(self.output_embeds[i].layers[-1].bias.
                data, 0.0 if i < 2 or i >= 4 else -2.0)

    def reset_parameters_as_refine_head(self):
        for i in range(self.num_steps):
>>>            torch.nn.init.constant_(self.output_embeds[i].layers[-1].weight
                .data, 0)
>>>            torch.nn.init.constant_(self.output_embeds[i].layers[-1].bias.
                data, 0)

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        bs, l, c = tgt.shape
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        tgt2, self.pre_kv = self.self_attn(tgt.view(1, bs * l, c), pre_kv=
            self.pre_kv)
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        return tgt2.view(bs, l, c)

    def forward(self, feat, query_pos, reference_points, srcs,
        src_padding_masks, **kwargs):
        cs_all, nobj, c = feat.shape
        previous_logits = kwargs.pop('previous_logits', None)
        class_vector = kwargs.pop('class_vector', None)
        bs_idx, cls_idx = kwargs.pop('bs_idx', None), kwargs.pop('cls_idx',
            None)
        if kwargs.pop('rearrange', False):
            num_steps, cls_idx, feat, class_vector, bs_idx, kwargs[
                'src_valid_ratios'
                ] = self.post_process.taskCategory.arrangeBySteps(cls_idx,
                feat, class_vector, bs_idx, kwargs['src_valid_ratios'])
        else:
            num_steps = self.post_process.taskCategory.getNumSteps(cls_idx)
        output_classes = self.classifier(feat, class_vector=class_vector.
            unsqueeze(axis=1) if class_vector is not None else None)
        output_classes = self.postprocess_logits(output_classes,
            previous_logits, bs_idx, cls_idx)
        input_feat = feat
        output_signals = []
        original_reference_points = reference_points
        self.pre_kv = None
        self.cross_attn.preprocess_value(srcs, src_padding_masks, bs_idx=bs_idx
            )
        for id_step, output_embed in enumerate(self.output_embeds):
            if self.pos_emb is not None:
                feat = feat + self.pos_emb.weight[id_step]
            forward_reference_points = reference_points.detach()
            output_feat = super().forward(feat, query_pos,
                forward_reference_points, srcs, src_padding_masks, **kwargs)
            output_signal = output_embed(output_feat).squeeze(axis=-1)
            output_signals.append(output_signal)
            feat = self.generate_feat_for_next_step(output_feat,
                output_signal, reference_points, None, id_step)
            reference_points = self.adjust_reference_points(output_signals,
                reference_points, id_step)
            if (num_steps == id_step + 1).sum(
                ) > 0 and id_step < self.num_steps:
                count_needs = (num_steps > id_step + 1).sum()
                old_cs = feat.shape[0]
                feat = feat[:count_needs]
                reference_points = reference_points[:count_needs]
                """Class Method: *.unflatten, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                self.pre_kv = self.pre_kv.unflatten(1, (old_cs, nobj))[:, :
                    count_needs].flatten(start_axis=1, stop_axis=2)
                self.cross_attn.value = self.cross_attn.value[:count_needs]
                kwargs['src_valid_ratios'] = kwargs['src_valid_ratios'][:
                    count_needs]
        outputs = self.post_process(output_signals, output_classes,
            original_reference_points, bs_idx, cls_idx)
        targets = kwargs.pop('targets', None)
        if targets is not None:
            loss_dict = self.criterion(outputs, targets)
        else:
            assert not self.training, 'Targets are required for training mode (unified_seq_head.py)'
            loss_dict = {}
        return outputs, loss_dict

    def generate_feat_for_next_step(self, output_feat, output_signal,
        reference_logits, boxes, id_step):
        feat = output_feat.clone().detach()
        return feat

    def postprocess_logits(self, outputs_logits, previous_logits, bs_idx,
        cls_idx):
        if previous_logits is not None:
            previous_logits = previous_logits[bs_idx, cls_idx]
            previous_logits = previous_logits.unsqueeze(axis=-1)
            if self.sg_previous_logits:
                previous_logits = previous_logits.detach()
        if self.combine_method == 'none':
            return outputs_logits
        elif self.combine_method == 'add':
            return outputs_logits.sigmoid() + previous_logits.sigmoid()
        elif self.combine_method == 'multiple':
            return inverse_sigmoid(outputs_logits.sigmoid() *
                previous_logits.sigmoid())
        else:
            raise KeyError


class MLP(paddle.nn.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.LayerList(sublayers=(paddle.nn.Linear(
            in_features=n, out_features=k) for n, k in zip([input_dim] + h,
            h + [output_dim])))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = paddle.nn.functional.relu(x=layer(x)
                ) if i < self.num_layers - 1 else layer(x)
        return x

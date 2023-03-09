import paddle
import numpy as np
from util.misc import NestedTensor
from .encoder import build_encoder
from .prompt_indicator import PromptIndicator
from .object_decoder import ObjectDecoder
from .position_encoding import build_position_encoding


class Transformer(paddle.nn.Layer):

    def __init__(self, args=None):
        super().__init__()
        self.d_model = args.hidden_dim
        self.encoder = build_encoder(args) if args.enc_layers > 0 else None
        self.position_embed = build_position_encoding(args)
        if self.encoder is not None:
>>>            self.level_embed = torch.nn.Parameter(torch.Tensor(args.
                num_feature_levels, self.d_model))
>>>            torch.nn.init.normal_(self.level_embed)
        if args.with_prompt_indicator:
            self.prompt_indicator = PromptIndicator(args.PROMPT_INDICATOR)
        else:
            self.prompt_indicator = None
        if args.with_object_decoder:
            self.object_decoder = ObjectDecoder(self.d_model, args=args.
                OBJECT_DECODER)
        else:
            self.object_decoder = None

    def forward(self, srcs, masks, targets=None):
        bs = srcs[0].shape[0]
        srcs, mask, enc_kwargs, cls_kwargs, obj_kwargs = (self.
            prepare_for_deformable(srcs, masks))
        srcs = self.encoder(srcs, padding_mask=mask, **enc_kwargs
            ) if self.encoder is not None else srcs
        outputs, loss_dict = {}, {}
        if self.prompt_indicator is not None:
            cls_outputs, cls_loss_dict = self.prompt_indicator(srcs, mask,
                targets=targets, kwargs=cls_kwargs)
            outputs.update(cls_outputs)
            loss_dict.update(cls_loss_dict)
            additional_object_inputs = dict(bs_idx=outputs['bs_idx'] if 
                'bs_idx' in outputs else None, cls_idx=outputs['cls_idx'] if
                'cls_idx' in outputs else None, class_vector=outputs[
                'tgt_class'], previous_logits=outputs['cls_label_logits'])
        else:
            additional_object_inputs = {}
        if self.object_decoder is not None:
            obj_outputs, obj_loss_dict = self.object_decoder(srcs, mask,
                targets=targets, additional_info=additional_object_inputs,
                kwargs=obj_kwargs)
            outputs.update(obj_outputs)
            loss_dict.update(obj_loss_dict)
        return outputs, loss_dict

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = paddle.sum(x=~mask[:, :, (0)], axis=1)
        valid_W = paddle.sum(x=~mask[:, (0), :], axis=1)
        """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        valid_ratio_h = valid_H.float() / H
        """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        valid_ratio_w = valid_W.float() / W
        valid_ratio = paddle.stack(x=[valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
>>>            ref_y, ref_x = torch.meshgrid(paddle.linspace(start=0.5, stop=
                H_ - 0.5, num=H_).astype('float32'), paddle.linspace(start=
                0.5, stop=W_ - 0.5, num=W_).astype('float32'))
            """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, (None), (lvl
                ), (1)] * H_)
            """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, (None), (lvl
                ), (0)] * W_)
            ref = paddle.stack(x=(ref_x, ref_y), axis=-1)
            reference_points_list.append(ref)
        reference_points = paddle.concat(x=reference_points_list, axis=1)
        reference_points = reference_points[:, :, (None)] * valid_ratios[:,
            (None)]
        return reference_points

    def prepare_for_deformable(self, srcs, masks):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            if self.encoder is not None:
                """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                pos_embed = self.position_embed(NestedTensor(src, mask)).to(src
                    .dtype)
                x = pos_embed.flatten(start_axis=2)
                perm_18 = list(range(x.ndim))
                perm_18[1] = 2
                perm_18[2] = 1
                pos_embed = x.transpose(perm=perm_18)
                """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1
                    )
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            x = src.flatten(start_axis=2)
            perm_19 = list(range(x.ndim))
            perm_19[1] = 2
            perm_19[2] = 1
            src = x.transpose(perm=perm_19)
            mask = mask.flatten(start_axis=1)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(x=src_flatten, axis=1)
        mask_flatten = paddle.concat(x=mask_flatten, axis=1)
        lvl_pos_embed_flatten = paddle.concat(x=lvl_pos_embed_flatten, axis=1
            ) if self.encoder is not None else None
        spatial_shapes = paddle.to_tensor(data=spatial_shapes, place=
            src_flatten.place).astype('int64')
        """Class Method: *.new_zeros, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        level_start_index = paddle.concat(x=(spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(axis=1).cumsum(axis=0)[:-1]))
        valid_ratios = paddle.stack(x=[self.get_valid_ratio(m) for m in
            masks], axis=1)
        reference_points_enc = self.get_reference_points(spatial_shapes,
            valid_ratios, device=src.place)
        enc_kwargs = dict(spatial_shapes=spatial_shapes, level_start_index=
            level_start_index, reference_points=reference_points_enc, pos=
            lvl_pos_embed_flatten)
        cls_kwargs = dict(src_level_start_index=level_start_index)
        obj_kwargs = dict(src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index, src_valid_ratios=
            valid_ratios)
        return src_flatten, mask_flatten, enc_kwargs, cls_kwargs, obj_kwargs

import paddle
import copy
from models.ops.modules import MSDeformAttn


class BasicEncoderLayer(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.normalize_before = args.pre_norm
        self.build_self_attn(args)
        self.dropout1 = paddle.nn.Dropout(p=args.dropout)
>>>        self.norm1 = torch.nn.LayerNorm(self.d_model)
        self.ffn = FFN(self.d_model, args.dim_feedforward, args.dropout,
            args.activation, normalize_before=self.normalize_before)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, **kwargs):
        src2 = self.self_attn_forward(src, **kwargs)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.ffn(src)
        return src

    def forward_pre(self, src, **kwargs):
        src2 = self.norm1(src)
        src2 = self.self_attn_forward(src2, **kwargs)
        src = src + self.dropout1(src2)
        src = self.ffn(src)
        return src

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class DeformableEncoderLayer(BasicEncoderLayer):

    def build_self_attn(self, args):
        self.self_attn = MSDeformAttn(self.d_model, args.n_levels, args.
            nheads, args.n_points)

    def self_attn_forward(self, src, **kwargs):
        pos = kwargs.pop('pos', None)
        reference_points = kwargs.pop('reference_points')
        spatial_shapes = kwargs.pop('spatial_shapes')
        level_start_index = kwargs.pop('level_start_index')
        padding_mask = kwargs.pop('padding_mask', None)
        src2 = self.self_attn(self.with_pos_embed(src, pos),
            reference_points, src, spatial_shapes, level_start_index,
            padding_mask)
        return src2


class BasicDecoderLayer(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.n_heads = args.nheads
        self.normalize_before = args.pre_norm
        self.build_cross_attn(args)
        self.dropout1 = paddle.nn.Dropout(p=args.dropout)
>>>        self.norm1 = torch.nn.LayerNorm(self.d_model)
        self.self_attn = not args.no_self_attn
        if self.self_attn:
>>>            self.self_attn = torch.nn.MultiheadAttention(self.d_model, self
                .n_heads, dropout=args.dropout)
            self.dropout2 = paddle.nn.Dropout(p=args.self_attn_dropout)
>>>            self.norm2 = torch.nn.LayerNorm(self.d_model)
        self.ffn = FFN(self.d_model, args.dim_feedforward, args.dropout,
            args.activation, normalize_before=self.normalize_before)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        if query_pos is not None and query_pos.shape[0] != tgt.shape[0]:
            cs = tgt.shape[0] // query_pos.shape[0]
            """Class Method: *.repeat_interleave, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            query_pos_self = query_pos.repeat_interleave(repeats=cs, dim=0)
        else:
            query_pos_self = query_pos
        q = k = self.with_pos_embed(tgt, query_pos_self)
        x = q
        perm_10 = list(range(x.ndim))
        perm_10[0] = 1
        perm_10[1] = 0
        x = k
        perm_11 = list(range(x.ndim))
        perm_11[0] = 1
        perm_11[1] = 0
        x = tgt
        perm_12 = list(range(x.ndim))
        perm_12[0] = 1
        perm_12[1] = 0
        x = self.self_attn(x.transpose(perm=perm_10), x.transpose(perm=
            perm_11), x.transpose(perm=perm_12))[0]
        perm_13 = list(range(x.ndim))
        perm_13[0] = 1
        perm_13[1] = 0
        tgt2 = x.transpose(perm=perm_13)
        return tgt2

    def forward_post(self, tgt, query_pos, **kwargs):
        if self.self_attn:
            tgt2 = self.self_attn_forward(tgt, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        tgt2 = self.cross_attn_forward(tgt, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.ffn(tgt)
        return tgt

    def forward_pre(self, tgt, query_pos, **kwargs):
        if self.self_attn:
            tgt2 = self.norm2(tgt)
            tgt2 = self.self_attn_forward(tgt2, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn_forward(tgt2, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.ffn(tgt)
        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class MultiHeadDecoderLayer(BasicDecoderLayer):

    def build_cross_attn(self, args):
>>>        self.cross_attn = torch.nn.MultiheadAttention(self.d_model, self.
            n_heads, dropout=args.dropout)

    def cross_attn_forward(self, tgt, query_pos, **kwargs):
        bs_all, seq, c = tgt.shape
        srcs = kwargs['srcs']
        bs = srcs.shape[0]
        if bs_all > bs:
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            tgt = tgt.view(bs, -1, c)
            cs = bs_all // bs
        src_padding_masks = kwargs.pop('src_padding_masks')
        posemb_2d = kwargs.pop('posemb_2d', 0)
        query_pos = paddle.zeros_like(x=tgt
            ) if query_pos is None else query_pos.tile(repeat_times=[1, cs, 1])
        x = tgt + query_pos
        perm_14 = list(range(x.ndim))
        perm_14[0] = 1
        perm_14[1] = 0
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        x = (srcs + posemb_2d).reshape(bs, -1, c)
        perm_15 = list(range(x.ndim))
        perm_15[0] = 1
        perm_15[1] = 0
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        x = srcs.reshape(bs, -1, c)
        perm_16 = list(range(x.ndim))
        perm_16[0] = 1
        perm_16[1] = 0
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        x = self.cross_attn(x.transpose(perm=perm_14), x.transpose(perm=
            perm_15), x.transpose(perm=perm_16), key_padding_mask=
            src_padding_masks.reshape(bs, -1))[0]
        perm_17 = list(range(x.ndim))
        perm_17[0] = 1
        perm_17[1] = 0
        tgt2 = x.transpose(perm=perm_17)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        return tgt2.reshape(bs_all, seq, c)

    def forward(self, tgt, query_pos, reference_points, srcs,
        src_padding_masks, **kwargs):
        return super().forward(tgt, query_pos, srcs=srcs, src_padding_masks
            =src_padding_masks, **kwargs)


class DeformableDecoderLayer(BasicDecoderLayer):

    def build_cross_attn(self, args):
        self.cross_attn = MSDeformAttn(self.d_model, args.n_levels, args.
            nheads, args.n_points, no_value_proj=args.cross_attn_no_value_proj)

    def cross_attn_forward(self, tgt, query_pos, reference_points, srcs,
        src_padding_masks, **kwargs):
        bs_all, seq, c = tgt.shape
        num_levels = reference_points.shape[-2]
        bs = srcs.shape[0]
        cs_batch = kwargs.pop('cs_batch', None)
        src_spatial_shapes = kwargs.pop('src_spatial_shapes')
        level_start_index = kwargs.pop('src_level_start_index')
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
            reference_points, srcs, src_spatial_shapes, level_start_index,
            src_padding_masks, cs_batch=cs_batch)
        return tgt2

    def forward(self, tgt, query_pos, reference_points, srcs,
        src_padding_masks, **kwargs):
        src_valid_ratios = kwargs.pop('src_valid_ratios')
        if reference_points.shape[-1] == 4:
            src_valid_ratios = paddle.concat(x=[src_valid_ratios,
                src_valid_ratios], axis=-1)
        if src_valid_ratios.shape[0] != reference_points.shape[0]:
            repeat_times = reference_points.shape[0] // src_valid_ratios.shape[
                0]
            """Class Method: *.repeat_interleave, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            src_valid_ratios = src_valid_ratios.repeat_interleave(repeat_times,
                dim=0)
        src_valid_ratios = src_valid_ratios[:, (None)] if reference_points.dim(
            ) == 3 else src_valid_ratios[:, (None), (None)]
        reference_points_input = reference_points[(...), (None), :
            ] * src_valid_ratios
        return super().forward(tgt, query_pos, reference_points=
            reference_points_input, srcs=srcs, src_padding_masks=
            src_padding_masks, **kwargs)


class FFN(paddle.nn.Layer):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0, activation=
        'relu', normalize_before=False):
        super().__init__()
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=d_ffn
            )
        self.activation = _get_activation_fn(activation)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=d_ffn, out_features=d_model
            )
        self.dropout3 = paddle.nn.Dropout(p=dropout)
>>>        self.norm2 = torch.nn.LayerNorm(d_model)
        self.normalize_before = normalize_before

    def forward_post(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src):
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)
        return src

    def forward(self, src):
        if self.normalize_before:
            return self.forward_pre(src)
        return self.forward_post(src)


def _get_clones(module, N):
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for i in
        range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return paddle.nn.functional.relu
    if activation == 'gelu':
        return paddle.nn.functional.gelu
    if activation == 'glu':
        return paddle.nn.functional.glu
    if activation == 'prelu':
        return paddle.nn.PReLU()
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')

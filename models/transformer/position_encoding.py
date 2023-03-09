import paddle
"""
Various positional encodings for the transformer.
"""
import math
from util.misc import NestedTensor


class PositionEmbeddingSine(paddle.nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False,
        scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(axis=1).astype('float32')
        x_embed = not_mask.cumsum(axis=2).astype('float32')
        if self.normalize:
            eps = 1e-06
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = paddle.arange(start=self.num_pos_feats).astype('float32')
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, (None)] / dim_t
        pos_y = y_embed[:, :, :, (None)] / dim_t
        pos_x = paddle.stack(x=(pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 
            1::2].cos()), axis=4).flatten(start_axis=3)
        pos_y = paddle.stack(x=(pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 
            1::2].cos()), axis=4).flatten(start_axis=3)
        pos = paddle.concat(x=(pos_y, pos_x), axis=3).transpose(perm=[0, 3,
            1, 2])
        return pos


class PositionEmbeddingLearned(paddle.nn.Layer):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
>>>        self.row_embed = torch.nn.Embedding(50, num_pos_feats)
>>>        self.col_embed = torch.nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
>>>        torch.nn.init.uniform_(self.row_embed.weight)
>>>        torch.nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = paddle.arange(start=w)
        j = paddle.arange(start=h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = paddle.concat(x=[x_emb.unsqueeze(axis=0).tile(repeat_times=[h,
            1, 1]), y_emb.unsqueeze(axis=1).tile(repeat_times=[1, w, 1])],
            axis=-1).transpose(perm=[2, 0, 1]).unsqueeze(axis=0).tile(
            repeat_times=[x.shape[0], 1, 1, 1])
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f'not supported {args.position_embedding}')
    return position_embedding

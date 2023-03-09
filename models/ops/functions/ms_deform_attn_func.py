import paddle
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import MultiScaleDeformableAttention as MSDA


>>>class MSDeformAttnFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
            value_level_start_index, sampling_locations, attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
            value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
>>>    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = (MSDA.
            ms_deform_attn_backward(value, value_spatial_shapes,
            value_level_start_index, sampling_locations, attention_weights,
            grad_output, ctx.im2col_step))
        return (grad_value, None, None, grad_sampling_loc, grad_attn_weight,
            None)


def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
    sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes],
        dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        x = value_list[lid_].flatten(start_axis=2)
        perm_6 = list(range(x.ndim))
        perm_6[1] = 2
        perm_6[2] = 1
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        value_l_ = x.transpose(perm=perm_6).reshape(N_ * M_, D_, H_, W_)
        x = sampling_grids[:, :, :, (lid_)]
        perm_7 = list(range(x.ndim))
        perm_7[1] = 2
        perm_7[2] = 1
        sampling_grid_l_ = x.transpose(perm=perm_7).flatten(start_axis=0,
            stop_axis=1)
        sampling_value_l_ = paddle.nn.functional.grid_sample(x=value_l_,
            grid=sampling_grid_l_, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    x = attention_weights
    perm_8 = list(range(x.ndim))
    perm_8[1] = 2
    perm_8[2] = 1
    """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    attention_weights = x.transpose(perm=perm_8).reshape(N_ * M_, 1, Lq_, 
        L_ * P_)
    """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    output = (paddle.stack(x=sampling_value_list, axis=-2).flatten(
        start_axis=-2) * attention_weights).sum(axis=-1).view(N_, M_ * D_, Lq_)
    x = output
    perm_9 = list(range(x.ndim))
    perm_9[1] = 2
    perm_9[2] = 1
    """Class Method: *.contiguous, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    return x.transpose(perm=perm_9).contiguous()

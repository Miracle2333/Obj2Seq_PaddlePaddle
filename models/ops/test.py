import paddle
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import time
from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
"""Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>shapes = paddle.to_tensor(data=[(6, 4), (3, 2)]).astype('int64').cuda()
"""Class Method: *.new_zeros, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>level_start_index = paddle.concat(x=(shapes.new_zeros((1,)), shapes.prod(
    axis=1).cumsum(axis=0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])
paddle.seed(seed=3)


@paddle.no_grad()
def check_forward_equal_with_pytorch_double():
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    value = paddle.rand(shape=[N, S, M, D]).cuda() * 0.01
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    sampling_locations = paddle.rand(shape=[N, Lq, M, L, P, 2]).cuda()
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    attention_weights = paddle.rand(shape=[N, Lq, M, L, P]).cuda() + 1e-05
    attention_weights /= attention_weights.sum(axis=-1, keepdim=True).sum(axis
        =-2, keepdim=True)
    im2col_step = 2
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes,
        sampling_locations.double(), attention_weights.double()).detach().cpu()
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes,
        level_start_index, sampling_locations.double(), attention_weights.
        double(), im2col_step).detach().cpu()
    fwdok = paddle.allclose(x=output_cuda, y=output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()
        ).max()
    print(
        f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}'
        )


@paddle.no_grad()
def check_forward_equal_with_pytorch_float():
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    value = paddle.rand(shape=[N, S, M, D]).cuda() * 0.01
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    sampling_locations = paddle.rand(shape=[N, Lq, M, L, P, 2]).cuda()
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    attention_weights = paddle.rand(shape=[N, Lq, M, L, P]).cuda() + 1e-05
    attention_weights /= attention_weights.sum(axis=-1, keepdim=True).sum(axis
        =-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes,
        sampling_locations, attention_weights).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value, shapes,
        level_start_index, sampling_locations, attention_weights, im2col_step
        ).detach().cpu()
    fwdok = paddle.allclose(x=output_cuda, y=output_pytorch, rtol=0.01,
        atol=0.001)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()
        ).max()
    print(
        f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}'
        )


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc
    =True, grad_attn_weight=True):
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    value = paddle.rand(shape=[N, S, M, channels]).cuda() * 0.01
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    sampling_locations = paddle.rand(shape=[N, Lq, M, L, P, 2]).cuda()
    """Class Method: *.cuda, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    attention_weights = paddle.rand(shape=[N, Lq, M, L, P]).cuda() + 1e-05
    attention_weights /= attention_weights.sum(axis=-1, keepdim=True).sum(axis
        =-2, keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply
    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    """Class Method: *.double, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    gradok = torch.autograd.gradcheck(func, (value.double(), shapes,
        level_start_index, sampling_locations.double(), attention_weights.
        double(), im2col_step))
    print(f'* {gradok} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()
    for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
        check_gradient_numerical(channels, True, True, True)

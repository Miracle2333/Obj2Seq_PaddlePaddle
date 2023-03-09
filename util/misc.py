import paddle
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
from packaging import version
>>>if version.parse(torchvision.__version__) < version.parse('0.5'):
    import math

    def _check_size_scale_factor(dim, size, scale_factor):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError(
                'only one of size or scale_factor should be defined')
        if not (scale_factor is not None and len(scale_factor) != dim):
            raise ValueError(
                'scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'
                .format(dim, len(scale_factor)))

    def _output_size(dim, input, size, scale_factor):
        assert dim == 2
        _check_size_scale_factor(dim, size, scale_factor)
        if size is not None:
            return size
        assert scale_factor is not None and isinstance(scale_factor, (int,
            float))
        scale_factors = [scale_factor, scale_factor]
        return [int(math.floor(input.shape[i + 2] * scale_factors[i])) for
            i in range(dim)]
>>>elif version.parse(torchvision.__version__) < version.parse('0.7'):


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
>>>        t = torch.tensor([self.count, self.total], dtype='float64', device=
            'cuda')
>>>        torch.distributed.barrier()
>>>        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
>>>        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
>>>        d = torch.tensor(list(self.deque), dtype='float32')
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg
            =self.global_avg, max=self.max, value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    buffer = pickle.dumps(data)
>>>    storage = torch.ByteStorage.from_buffer(buffer)
    """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    tensor = torch.ByteTensor(storage).to('cuda')
>>>    local_size = torch.tensor([tensor.size()], device='cuda')
>>>    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
>>>    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = []
    for _ in size_list:
        tensor_list.append(paddle.empty(shape=(max_size,), dtype='uint8'))
    if local_size != max_size:
        padding = paddle.empty(shape=(max_size - local_size,), dtype='uint8')
        tensor = paddle.concat(x=(tensor, padding), axis=0)
>>>    torch.distributed.all_gather(tensor_list, tensor)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with paddle.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = paddle.stack(x=values, axis=0)
>>>        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
>>>        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header, '[{0' + space_fmt +
                '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}',
                'data: {data}', 'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header, '[{0' + space_fmt +
                '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}',
                'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
>>>                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(
>>>                        data_time), memory=torch.cuda.max_memory_allocated(
                        ) / MB))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(
                        data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
            total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip(
            )
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'has uncommited changes' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f'sha: {sha}, status: {diff}, branch: {branch}'
    return message


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[paddle.Tensor],
    fix_input=None, input_divisor=None):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        if fix_input is not None:
            if isinstance(fix_input, int):
                max_size = [3, fix_input, fix_input]
            elif len(fix_input) == 2:
                max_size = [3, fix_input[0], fix_input[1]]
            else:
                raise ValueError('not supported DATA.COLLECT_FN.fix_input')
        if input_divisor is not None:
            max_size[1] = int(max_size[1] // input_divisor + 1
                ) * input_divisor if max_size[1] % input_divisor else max_size[
                1]
            max_size[2] = int(max_size[2] // input_divisor + 1
                ) * input_divisor if max_size[2] % input_divisor else max_size[
                2]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].place
        tensor = paddle.zeros(shape=[batch_shape], dtype=dtype)
        mask = paddle.ones(shape=(b, h, w), dtype='bool')
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            """Class Method: *.copy_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[paddle.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        """Class Method: *.record_stream, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            """Class Method: *.record_stream, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
>>>    if not torch.distributed.is_available():
        return False
>>>    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
>>>    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
>>>    return torch.distributed.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
>>>        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.rank > -1:
        args.gpu = args.rank % paddle.device.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['LOCAL_SIZE'] = str(paddle.device.cuda.device_count())
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(paddle.device.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = paddle.device.cuda.device_count()
        addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.
            format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    paddle.device.set_device(device=args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.
        dist_url), flush=True)
>>>    torch.distributed.init_process_group(backend=args.dist_backend,
        init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
>>>    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@paddle.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.size() == 0:
        return [paddle.zeros(shape=[])]
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(k=maxk, axis=1, largest=True, sorted=True)
    pred = pred.t()
    """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    correct = pred.equal(y=target.view(1, -1).expand_as(y=pred))
    res = []
    for k in topk:
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        correct_k = correct[:k].view(-1).float().sum(axis=0)
        res.append(correct_k.scale_(scale=100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode='nearest',
    align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
>>>    if float(torchvision.__version__[:3]) < 0.7:
        if input.size() > 0:
>>>            return torch.nn.functional.interpolate(input, size,
                scale_factor, mode, align_corners)
>>>        output_shape = torchvision.ops.misc._output_size(2, input, size,
            scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
>>>        if float(torchvision.__version__[:3]) < 0.5:
>>>            return torchvision.ops.misc._NewEmptyTensorOp.apply(input,
                output_shape)
>>>        return torchvision.ops._new_empty_tensor(input, output_shape)
    else:
>>>        return torchvision.ops.misc.interpolate(input, size, scale_factor,
            mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.place
    """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    total_norm = paddle.linalg.norm(x=paddle.stack(x=[paddle.linalg.norm(x=
        p.grad.detach(), p=norm_type).to(device) for p in parameters]), p=
        norm_type)
    return total_norm


def inverse_sigmoid(x, eps=1e-05):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x=x1 / x2)

import paddle


def item_cuda(item, device):
    if isinstance(item, list):
        return [item_cuda(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: item_cuda(v, device) for k, v in item.items()}
    elif isinstance(item, paddle.Tensor):
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        return item.to(device, non_blocking=True)
    else:
        raise TypeError


def to_cuda(samples, targets, device):
    """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    samples = samples.to(device, non_blocking=True)
    targets = item_cuda(targets, device)
    return samples, targets


class data_prefetcher:

    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.place = device
        if prefetch and self.place == 'cuda':
>>>            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
>>>        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.
                next_samples, self.next_targets, self.place)

    def next(self):
        if self.prefetch and self.place == 'cuda':
            paddle.device.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                """Class Method: *.record_stream, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                samples.record_stream(paddle.device.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        """Class Method: *.record_stream, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                        v.record_stream(paddle.device.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.place)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets

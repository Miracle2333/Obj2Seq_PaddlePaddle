import paddle
import os
import math


>>>class DistributedSampler(torch.utils.data.sampler.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=
        None, local_size=None, shuffle=True, fix_split=False):
        if num_replicas is None:
>>>            if not torch.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
>>>            num_replicas = torch.distributed.get_world_size()
        if rank is None:
>>>            if not torch.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
>>>            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.
            num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        if fix_split:
            assert hasattr(self.dataset, 'split_length'
                ), 'The dataset does not have split_length attribute'
            assert shuffle, 'sampler_fix_split only works with shuffle'
            self.split_length = self.dataset.split_length
        else:
            self.split_length = None

    def __iter__(self):
        if self.shuffle:
>>>            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.split_length is None:
                indices = paddle.randperm(n=len(self.dataset)).tolist()
            else:
                indices = []
                sub_indices = [paddle.randperm(n=l).tolist() for l in self.
                    split_length]
                for sub in sub_indices:
                    indices.extend(sub)
        else:
            indices = paddle.arange(start=len(self.dataset)).tolist()
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


>>>class NodeDistributedSampler(torch.utils.data.sampler.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=
        None, local_size=None, shuffle=True):
        if num_replicas is None:
>>>            if not torch.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
>>>            num_replicas = torch.distributed.get_world_size()
        if rank is None:
>>>            if not torch.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
>>>            rank = torch.distributed.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.
            num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.total_size_parts = (self.num_samples * self.num_replicas //
            self.num_parts)

    def __iter__(self):
        if self.shuffle:
>>>            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = paddle.randperm(n=len(self.dataset)).tolist()
        else:
            indices = paddle.arange(start=len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]
        indices += indices[:self.total_size_parts - len(indices)]
        assert len(indices) == self.total_size_parts
        indices = indices[self.rank // self.num_parts:self.total_size_parts
            :self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

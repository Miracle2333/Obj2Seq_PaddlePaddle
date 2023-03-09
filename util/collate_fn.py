import paddle
from .misc import nested_tensor_from_tensor_list


class BaseCollator:

    def __init__(self, fix_input=None, input_divisor=None):
        self.fix_input = fix_input
        self.input_divisor = input_divisor
        assert self.fix_input is None or self.input_divisor is None

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0], fix_input=self.
            fix_input, input_divisor=self.input_divisor)
        return tuple(batch)


class CLSCollator:

    def __init__(self, fix_input=None, input_divisor=None):
        self.fix_input = fix_input
        self.input_divisor = input_divisor
        assert self.fix_input is None or self.input_divisor is None

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0], fix_input=self.
            fix_input, input_divisor=self.input_divisor)
        if 'boxes_by_cls' in batch[1][0]:
            overall_batch = dict()
            overall_batch['image_id'] = paddle.stack(x=[item['image_id'] for
                item in batch[1]])
            overall_batch['size'] = paddle.stack(x=[item['size'] for item in
                batch[1]])
            overall_batch['orig_size'] = paddle.stack(x=[item['orig_size'] for
                item in batch[1]])
            """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            overall_batch['class_label'] = [item['labels'].unique() for
                item in batch[1]]
            overall_batch['multi_label'] = [item['multi_label'] for item in
                batch[1]]
            overall_batch['super_label'] = [item['super_label'] for item in
                batch[1]]
            overall_batch['boxes_by_cls'] = [item['boxes_by_cls'] for item in
                batch[1]]
            if 'keypoints_by_cls' in batch[1][0]:
                overall_batch['keypoints_by_cls'] = [item[
                    'keypoints_by_cls'] for item in batch[1]]
            if 'object_class' in batch[1][0]:
                overall_batch['object_class'] = paddle.stack(x=[item[
                    'object_class'] for item in batch[1]])
            batch[1] = overall_batch
        return tuple(batch)


def collate_fn_imnet(batch, is_train=True):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    if is_train:
        batch[1] = {'multi_label': [paddle.to_tensor(data=[item]) for item in
            batch[1]]}
    else:
        batch[1] = [{'multi_label': paddle.to_tensor(data=[item])} for item in
            batch[1]]
    return tuple(batch)


def build_collate_fn(args):
    if args.type == 'COCObyCLS':
        return CLSCollator(args.fix_input, args.input_divisor)
    elif args.type == 'none':
        return BaseCollator(args.fix_input, args.input_divisor)

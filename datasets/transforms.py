import paddle
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import numpy as np
import PIL
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
>>>    cropped_image = torchvision.transforms.functional.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
>>>    target['size'] = torch.tensor([h, w])
    fields = ['labels', 'area', 'iscrowd']
    if 'boxes' in target:
        boxes = target['boxes']
        max_size = paddle.to_tensor(data=[w, h]).astype('float32')
        cropped_boxes = boxes - paddle.to_tensor(data=[j, i, j, i])
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        cropped_boxes = paddle.minimum(x=cropped_boxes.reshape(-1, 2, 2), y
            =max_size)
        cropped_boxes = cropped_boxes.clip(min=0)
        area = (cropped_boxes[:, (1), :] - cropped_boxes[:, (0), :]).prod(axis
            =1)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        target['boxes'] = cropped_boxes.reshape(-1, 4)
        target['area'] = area
        fields.append('boxes')
    if 'masks' in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append('masks')
    if 'keypoints' in target:
        keypoints = target['keypoints']
        keypoints = keypoints - paddle.to_tensor(data=[j, i, 0])
        target['keypoints'] = keypoints
        fields.append('keypoints')
    if 'boxes' in target or 'masks' in target:
        if 'boxes' in target:
            """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = paddle.all(x=cropped_boxes[:, (1), :] > cropped_boxes[:,
                (0), :], dim=1)
        else:
            keep = target['masks'].flatten(start_axis=1).any(axis=1)
        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, target


KPT_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def hflip(image, target):
>>>    flipped_image = torchvision.transforms.functional.hflip(image)
    w, h = image.size
    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        boxes = boxes[:, ([2, 1, 0, 3])] * paddle.to_tensor(data=[-1, 1, -1, 1]
            ) + paddle.to_tensor(data=[w, 0, w, 0])
        target['boxes'] = boxes
    if 'masks' in target:
        target['masks'] = target['masks'].flip(axis=-1)
    if 'keypoints' in target:
        keypoints = target['keypoints']
        keypoints = keypoints * paddle.to_tensor(data=[-1, 1, 1]
            ) + paddle.to_tensor(data=[w, 0, 0])
        keypoints = keypoints[:, (KPT_FLIP_INDEX)]
        target['keypoints'] = keypoints
    return flipped_image, target


def resize(image, target, size, max_size=None):

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if w < h:
            ow = size
            oh = int(size * h / w)
            if max_size is not None and oh > max_size:
                oh = max_size
                ow = int(max_size * w / h)
        else:
            oh = size
            ow = int(size * w / h)
            if max_size is not None and ow > max_size:
                ow = max_size
                oh = int(max_size * h / w)
        return oh, ow

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    size = get_size(image.size, size, max_size)
>>>    rescaled_image = torchvision.transforms.functional.resize(image, size)
    if target is None:
        return rescaled_image, None
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(
        rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        scaled_boxes = boxes * paddle.to_tensor(data=[ratio_width,
            ratio_height, ratio_width, ratio_height])
        target['boxes'] = scaled_boxes
    if 'area' in target:
        area = target['area']
        scaled_area = area * (ratio_width * ratio_height)
        target['area'] = scaled_area
    h, w = size
>>>    target['size'] = torch.tensor([h, w])
    if 'masks' in target:
        """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        target['masks'] = interpolate(target['masks'][:, (None)].float(),
            size, mode='nearest')[:, (0)] > 0.5
    if 'keypoints' in target:
        keypoints = target['keypoints']
        scaled_keypoints = keypoints * paddle.to_tensor(data=[ratio_width,
            ratio_height, 1])
        target['keypoints'] = scaled_keypoints
    return rescaled_image, target


def pad(image, target, padding):
>>>    padded_image = torchvision.transforms.functional.pad(image, (0, 0,
        padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
>>>    target['size'] = torch.tensor(padded_image[::-1])
    if 'masks' in target:
        target['masks'] = paddle.nn.functional.pad(x=target['masks'], pad=(
            0, padding[0], 0, padding[1]))
    if 'keypoints' in target:
        keypoints = target['keypoints']
        keypoints = keypoints + paddle.to_tensor(data=[padding[0], padding[
            1], 0])
        target['keypoints'] = keypoints
    return padded_image, target


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
>>>        region = torchvision.transforms.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
>>>        region = torchvision.transforms.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width)
            )


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):

    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):

    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
>>>        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale
        target = target.copy()
        target['size'] = scaled_size
        if 'boxes' in target:
            boxes = target['boxes']
            scaled_boxes = boxes * paddle.to_tensor(data=[ratio_width,
                ratio_height, ratio_width, ratio_height])
            target['boxes'] = scaled_boxes
        if 'area' in target:
            area = target['area']
            scaled_area = area * (ratio_width * ratio_height)
            target['area'] = scaled_area
        if 'masks' in target:
            masks = target['masks']
            """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            masks = interpolate(masks[:, (None)].float(), scaled_size, mode
                ='nearest')[:, (0)] > 0.5
            target['masks'] = masks
        if 'keypoints' in target:
            keypoints = target['keypoints']
            scaled_keypoints = keypoints * paddle.to_tensor(data=[
                ratio_width, ratio_height, 1])
            target['keypoints'] = scaled_keypoints
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ['labels', 'area', 'iscrowd']
        target = target.copy()
>>>        target['size'] = torch.tensor([h, w])
        if 'boxes' in target:
            boxes = target['boxes']
            max_size = paddle.to_tensor(data=[w, h]).astype('float32')
            cropped_boxes = boxes - paddle.to_tensor(data=[j, i, j, i])
            """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            cropped_boxes = paddle.minimum(x=cropped_boxes.reshape(-1, 2, 2
                ), y=max_size)
            cropped_boxes = cropped_boxes.clip(min=0)
            area = (cropped_boxes[:, (1), :] - cropped_boxes[:, (0), :]).prod(
                axis=1)
            """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            target['boxes'] = cropped_boxes.reshape(-1, 4)
            target['area'] = area
            fields.append('boxes')
        if 'masks' in target:
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append('masks')
        if 'keypoints' in target:
            keypoints = target['keypoints']
            keypoints = keypoints - paddle.to_tensor(data=[j, i, 0])
            target['keypoints'] = keypoints
            fields.append('keypoints')
        if 'boxes' in target or 'masks' in target:
            if 'boxes' in target:
                """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = paddle.all(x=cropped_boxes[:, (1), :] >
                    cropped_boxes[:, (0), :], dim=1)
            else:
                keep = target['masks'].flatten(start_axis=1).any(axis=1)
            for field in fields:
                target[field] = target[field][keep]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if 'masks' in target:
            target['masks'] = paddle.nn.functional.pad(x=target['masks'],
                pad=(0, padding[1], 0, padding[0]))
        if 'keypoints' in target:
            keypoints = target['keypoints']
            keypoints = keypoints + paddle.to_tensor(data=[padding[1],
                padding[0], 0])
            target['keypoints'] = keypoints
        return target

    def __call__(self, image, target=None):
        image_size = image.size
>>>        image_size = torch.tensor(image_size[::-1])
        """Class Method: *.round, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.int, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        out_desired_size = (self.desired_size * image_size / max(image_size)
            ).round().int()
        random_scale = paddle.rand(shape=[1]) * (self.aug_scale_max - self.
            aug_scale_min) + self.aug_scale_min
        """Class Method: *.round, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        scaled_size = (random_scale * self.desired_size).round()
        scale = paddle.minimum(x=scaled_size / image_size[0], y=scaled_size /
            image_size[1])
        """Class Method: *.round, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.int, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        scaled_size = (image_size * scale).round().int()
>>>        scaled_image = torchvision.transforms.functional.resize(image,
            scaled_size.tolist())
        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)
        if random_scale > 1:
            max_offset = scaled_size - out_desired_size
            """Class Method: *.int, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            offset = (max_offset * paddle.rand(shape=[2])).floor().int()
            region = offset[0].item(), offset[1].item(), out_desired_size[0
                ].item(), out_desired_size[1].item()
>>>            output_image = torchvision.transforms.functional.crop(scaled_image,
                *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            padding = out_desired_size - scaled_size
>>>            output_image = torchvision.transforms.functional.pad(scaled_image,
                [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)
        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5
        ):
        self.prob = prob
>>>        self.tfm = torchvision.transforms.ColorJitter(brightness, contrast,
            saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target


class ToTensor(object):

    def __call__(self, img, target):
>>>        return torchvision.transforms.functional.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
>>>        self.eraser = torchvision.transforms.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
>>>        image = torchvision.transforms.functional.normalize(image, mean=
            self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = box_xyxy_to_cxcywh(boxes)
>>>            boxes = boxes / torch.tensor([w, h, w, h], dtype='float32')
            target['boxes'] = boxes
        if 'keypoints' in target:
            keypoints = target['keypoints']
>>>            keypoints = keypoints / torch.tensor([w, h, 1], dtype='float32')
            target['keypoints'] = keypoints
        return image, target


class GenerateClassificationResults(object):

    def __init__(self, num_cats):
        self.num_cats = num_cats

    def __call__(self, image, target):
        """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        multi_labels = target['labels'].unique()
        multi_label_onehot = paddle.zeros(shape=[self.num_cats])
        multi_label_onehot[multi_labels] = 1
        multi_label_weights = paddle.ones_like(x=multi_label_onehot)
        keep = target['iscrowd'] == 0
        fields = ['labels', 'area', 'iscrowd']
        if 'boxes' in target:
            fields.append('boxes')
        if 'masks' in target:
            fields.append('masks')
        if 'keypoints' in target:
            fields.append('keypoints')
        for field in fields:
            target[field] = target[field][keep]
        if 'neg_category_ids' in target:
            not_exhaustive_category_ids = [self.
                json_category_id_to_contiguous_id[idx] for idx in img_info[
                'not_exhaustive_category_ids'] if idx in self.
                json_category_id_to_contiguous_id]
            neg_category_ids = [self.json_category_id_to_contiguous_id[idx] for
                idx in img_info['neg_category_ids'] if idx in self.
                json_category_id_to_contiguous_id]
            multi_label_onehot[not_exhaustive_category_ids] = 1
            multi_label_weights = multi_label_onehot.clone()
            multi_label_weights[neg_category_ids] = 1
        else:
            sample_prob = paddle.zeros_like(x=multi_label_onehot) - 1
            """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            sample_prob[target['labels'].unique()] = 1
        target['multi_label_onehot'] = multi_label_onehot
        target['multi_label_weights'] = multi_label_weights
        target['force_sample_probs'] = sample_prob
        return image, target


class RearrangeByCls(object):

    def __init__(self, keep_keys=['size', 'orig_size', 'image_id',
        'multi_label_onehot', 'multi_label_weights', 'force_sample_probs'],
        min_keypoints_train=0):
        self.min_keypoints_train = min_keypoints_train
        self.keep_keys = keep_keys

    def __call__(self, image, target=None):
        """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        target['class_label'] = target['labels'].unique()
        new_target = {}
        """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        for icls in target['labels'].unique():
            icls = icls.item()
            new_target[icls] = {}
            where = target['labels'] == icls
            new_target[icls]['boxes'] = target['boxes'][where]
            if icls == 0 and 'keypoints' in target:
                new_target[icls]['keypoints'] = target['keypoints'][where]
        for key in self.keep_keys:
            new_target[key] = target[key]
        return image, new_target


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

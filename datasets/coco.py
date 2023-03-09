import paddle
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import os, cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pycocotools import mask as coco_mask
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann['keypoints'][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in anno)


def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False
    if 'keypoints' not in anno[0]:
        return True
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class CocoDetection(TvCocoDetection):

    def __init__(self, img_folder, ann_file, transforms, return_masks,
        cache_mode=False, local_rank=0, local_size=1, is_train=False,
        remove_empty_annotations=False):
        super(CocoDetection, self).__init__(img_folder, ann_file,
            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size
            )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(self.coco, return_masks)
        if is_train and remove_empty_annotations:
            self.ids = sorted(self.ids)
            ids = []
            obj_counts = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
                    obj_counts.append(len([obj for obj in anno if obj[
                        'iscrowd'] == 0 and obj['bbox'][2] > 0 and obj[
                        'bbox'][3] > 0]))
            self.ids = ids
        if str(self.root)[:5] == 's3://':
            conf_path = '~/petreloss.conf'
            if conf_path:
                from petrel_client.client import Client
                self.cclient = Client(conf_path)

    def ceph_read(self, filename, image_type):
        img_bytes = self.cclient.get(filename)
        assert img_bytes is not None
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        result = cv2.imdecode(img_array, image_type)
        return result

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        filename = os.path.join(self.root, path)
        if str(self.root)[:5] == 's3://':
            image = self.ceph_read(filename, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        else:
            return Image.open(filename).convert('RGB')

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target['num_classes'] = paddle.to_tensor(data=[len(self.coco.cats)])
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = paddle.to_tensor(data=mask).astype('uint8')
        mask = mask.any(axis=2)
        masks.append(mask)
    if masks:
        masks = paddle.stack(x=masks, axis=0)
    else:
        masks = paddle.zeros(shape=(0, height, width), dtype='uint8')
    return masks


class ConvertCocoPolysToMask(object):

    def __init__(self, coco, return_masks=False):
        self.return_masks = return_masks
        self.categories = {cat['id']: cat['name'] for cat in coco.cats.values()
            }
        self.json_category_id_to_contiguous_id = {v: i for i, v in
            enumerate(coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.
            json_category_id_to_contiguous_id.items()}

    def __call__(self, image, target):
        w, h = image.size
        image_id = target['image_id']
>>>        image_id = torch.tensor([image_id])
        anno = target['annotations']
        boxes = [obj['bbox'] for obj in anno]
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        boxes = paddle.to_tensor(data=boxes).astype('float32').reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clip_(min=0, max=w)
        boxes[:, 1::2].clip_(min=0, max=h)
        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
>>>        classes = torch.tensor(classes, dtype='int64')
        if self.return_masks:
            segmentations = [obj['segmentation'] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = paddle.to_tensor(data=keypoints).astype('float32')
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                keypoints = keypoints.view(num_keypoints, -1, 3)
        keep = (boxes[:, (3)] > boxes[:, (1)]) & (boxes[:, (2)] > boxes[:, (0)]
            )
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        if self.return_masks:
            target['masks'] = masks
        target['image_id'] = image_id
        if keypoints is not None:
            target['keypoints'] = keypoints
>>>        area = torch.tensor([obj['area'] for obj in anno])
>>>        iscrowd = torch.tensor([(obj['iscrowd'] if 'iscrowd' in obj else 0) for
            obj in anno])
        target['area'] = area[keep]
        target['iscrowd'] = iscrowd[keep]
        target['orig_size'] = paddle.to_tensor(data=[int(h), int(w)])
        target['size'] = paddle.to_tensor(data=[int(h), int(w)])
        return image, target


def build(image_set, transform, args):
    root = args.COCO.coco_path
    ann_file = (args.COCO.anno_train if image_set == 'train' else args.COCO
        .anno_val)
    if str(root)[:3] != 's3:':
        root = Path(root)
        assert root.exists(), f'provided COCO path {root} does not exist'
        img_root = root if (root / 'val2017').exists() else root / 'images'
    else:
        img_root = root
    img_folder = os.path.join(img_root, f'{image_set}2017')
    dataset = CocoDetection(img_folder, ann_file, transforms=transform,
        return_masks=args.COCO.masks, cache_mode=args.cache_mode,
        local_rank=get_local_rank(), local_size=get_local_size(), is_train=
        image_set == 'train', remove_empty_annotations=args.COCO.
        remove_empty_annotations)
    return dataset

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
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from .torchvision_datasets import CocoDetection as TvCocoDetection
from .coco import CocoDetection
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


class CocoHybrid(TvCocoDetection):

    def __init__(self, img_folder, det_ann, kpt_ann, transforms,
        return_masks, cache_mode=False, local_rank=0, local_size=1,
        is_train=False):
        super(CocoHybrid, self).__init__(img_folder, det_ann, cache_mode=
            cache_mode, local_rank=local_rank, local_size=local_size)
        self.dset_kps = CocoDetection(img_folder, kpt_ann, transforms,
            return_masks, cache_mode, local_rank, local_size, is_train,
            remove_empty_annotations=True)
        self.coco_kps = COCO(kpt_ann)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(self.coco, self.coco_kps,
            return_masks)
        if is_train:
            self.ids = sorted(self.ids)
            ids_w_kps = []
            ids_wo_kps = []
            for img_id in self.ids:
                ann_ids = self.coco_kps.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco_kps.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids_w_kps.append(img_id)
                else:
                    ids_wo_kps.append(img_id)
            self.ids = ids_w_kps + ids_wo_kps
            self.split_length = [len(ids_w_kps), len(ids_wo_kps)]
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
        if idx < 0:
            img, target = self.dset_kps[-idx]
            target['num_classes'] = paddle.to_tensor(data=[1])
            return img, target
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target['num_classes'] = paddle.to_tensor(data=[80])
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

    def __init__(self, coco, coco_kps, return_masks=False):
        self.return_masks = return_masks
        self.coco_kps = coco_kps
        self.categories = {cat['id']: cat['name'] for cat in coco.cats.values()
            }
        self.json_category_id_to_contiguous_id = {v: i for i, v in
            enumerate(coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.
            json_category_id_to_contiguous_id.items()}
        super_categorys = {i: coco.cats[i]['supercategory'] for i in coco.
            getCatIds()}
        super_categorys = {self.json_category_id_to_contiguous_id[k]: v for
            k, v in super_categorys.items()}
        self.super_to_super_id = {supername: idx for idx, supername in
            enumerate(set(super_categorys.values()), 80)}
        self.category_id_to_super_id = {k: self.super_to_super_id[v] for k,
            v in super_categorys.items()}

    def __call__(self, image, target):
        w, h = image.size
        image_id = target['image_id']
>>>        image_id = torch.tensor([image_id])
        anno = target['annotations']
        multi_labels = [self.json_category_id_to_contiguous_id[item[
            'category_id']] for item in anno]
        """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        super_labels = paddle.to_tensor(data=[self.category_id_to_super_id[
            k] for k in multi_labels]).astype('int64').unique()
        """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        multi_labels = paddle.to_tensor(data=multi_labels).astype('int64'
            ).unique()
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj[
            'iscrowd'] == 0]
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
        keypoints = paddle.zeros(shape=[classes.shape[0], 17 * 3])
        anno_kps = self.coco_kps.loadAnns([obj['id'] for obj in anno if obj
            ['category_id'] == 1])
        if anno_kps:
            keypoints_these = paddle.to_tensor(data=[obj['keypoints'] for
                obj in anno_kps]).astype('float32')
            keypoints[classes == 0] = keypoints_these
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        keypoints = keypoints.view(classes.shape[0], 17, 3)
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
        target['super_label'] = super_labels
        target['multi_label'] = multi_labels
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
    root = args.COCO_HYBRID.coco_path
    det_ann = args.COCO_HYBRID.detection_anno.format(image_set)
    kpt_ann = args.COCO_HYBRID.keypoint_anno.format(image_set)
    if str(root)[:3] != 's3:':
        root = Path(root)
        assert root.exists(), f'provided COCO path {root} does not exist'
        img_root = root if (root / 'val2017').exists() else root / 'images'
    else:
        img_root = root
    img_folder = os.path.join(img_root, f'{image_set}2017')
    dataset = CocoHybrid(img_folder, det_ann, kpt_ann, transforms=transform,
        return_masks=False, cache_mode=args.cache_mode, local_rank=
        get_local_rank(), local_size=get_local_size(), is_train=image_set ==
        'train')
    return dataset

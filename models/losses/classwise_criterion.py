import paddle
from util.misc import nested_tensor_from_tensor_list, interpolate, is_main_process, get_world_size, is_dist_avail_and_initialized
from util.task_category import TaskCategory
from .unified_single_class_criterion import UnifiedSingleClassCriterion


class ClasswiseCriterion(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        self.set_criterion = UnifiedSingleClassCriterion(args)
        self.taskCategory = TaskCategory(args.task_category, args.num_classes)
        self.need_keypoints = 'pose' in [it.name for it in self.
            taskCategory.tasks]

    def forward(self, outputs, targets):
        loss_dicts_all = []
        for tKey, output in outputs.items():
            device = output['pred_logits'].place
            cs_all, num_obj = output['pred_logits'].shape
            num_boxes = self.get_num_boxes(targets, device)
            num_pts = self.get_num_pts(targets, device
                ) if self.need_keypoints else 1
            num_people = self.get_num_people(targets, device
                ) if self.need_keypoints else 1
            bs_idx, cls_idx = output['batch_index'], output['class_index']
            task_info = self.taskCategory[tKey]
            target = []
            for id_b, id_c in zip(bs_idx, cls_idx):
                tgtThis = {}
                id_c = id_c.item()
                if id_c in targets[id_b]:
                    tgtOrigin = targets[id_b][id_c]
                    for key in task_info.required_targets:
                        tgtThis[key] = tgtOrigin[key]
                else:
                    for key in task_info.required_targets:
                        default_shape = task_info.required_targets[key]
                        tgtThis[key] = paddle.zeros(shape=[default_shape])
                target.append(tgtThis)
            loss_dicts_all.append(self.set_criterion(output, target,
                task_info.losses, num_boxes, num_pts, num_people))
        loss_dict = {}
        for idict in loss_dicts_all:
            for k in idict:
                if k in loss_dict:
                    loss_dict[k] += idict[k]
                else:
                    loss_dict[k] = idict[k]
        return loss_dict

    def get_num_boxes(self, targets, device):
        num_boxes = sum(sum(t[key]['boxes'].shape[0] for key in t if
            isinstance(key, int)) for t in targets)
        num_boxes = paddle.to_tensor(data=[num_boxes], place=device).astype(
            'float32')
        if is_dist_avail_and_initialized():
>>>            torch.distributed.all_reduce(num_boxes)
        num_boxes = paddle.clip(x=num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_num_pts(self, targets, device):
        kps = [t[0]['keypoints'] for t in targets if 0 in t]
        if len(kps) > 0:
            kps = paddle.concat(x=kps, axis=0)
            kps = (kps[..., 2] > 0) * (kps[(...), :2] >= 0).all(axis=-1) * (kps
                [(...), :2] <= 1).all(axis=-1)
            num_pts = kps.sum()
        else:
            num_pts = paddle.to_tensor(data=0.0, place=device)
        if is_dist_avail_and_initialized():
>>>            torch.distributed.all_reduce(num_pts)
        num_pts = paddle.clip(x=num_pts / get_world_size(), min=1).item()
        return num_pts

    def get_num_people(self, targets, device):
        num_people = sum(t[0]['boxes'].shape[0] for t in targets if 0 in t)
        num_people = paddle.to_tensor(data=[num_people], place=device).astype(
            'float32')
        if is_dist_avail_and_initialized():
>>>            torch.distributed.all_reduce(num_people)
        num_people = paddle.clip(x=num_people / get_world_size(), min=1).item()
        return num_people

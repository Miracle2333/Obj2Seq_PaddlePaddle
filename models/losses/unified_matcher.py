import paddle
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from scipy.optimize import linear_sum_assignment
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
KPS_OKS_SIGMAS = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 
    0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0


def joint_oks(src_joints, tgt_joints, tgt_bboxes, joint_sigmas=
    KPS_OKS_SIGMAS, with_center=True, eps=1e-15):
    tgt_flags = tgt_joints[:, :, (2)]
    tgt_joints = tgt_joints[:, :, 0:2]
    tgt_wh = tgt_bboxes[(...), 2:]
    tgt_areas = tgt_wh[..., 0] * tgt_wh[..., 1]
    num_gts, num_kpts = tgt_joints.shape[0:2]
    """Class Method: *.expand, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    areas = tgt_areas.unsqueeze(axis=1).expand(num_gts, num_kpts)
>>>    sigmas = torch.tensor(joint_sigmas).astype(dtype=tgt_joints.dtype)
    """Class Method: *.expand, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    sigmas_sq = paddle.square(x=2 * sigmas).unsqueeze(axis=0).expand(num_gts,
        num_kpts)
    d_sq = paddle.square(x=src_joints.unsqueeze(axis=1) - tgt_joints.
        unsqueeze(axis=0)).sum(axis=-1)
    """Class Method: *.expand, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    tgt_flags = tgt_flags.unsqueeze(axis=0).expand(*d_sq.shape)
    oks = paddle.exp(x=-d_sq / (2 * areas * sigmas_sq + eps))
    oks = oks * tgt_flags
    oks = oks.sum(axis=-1) / (tgt_flags.sum(axis=-1) + eps)
    return oks


class HungarianMatcher(paddle.nn.Layer):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, args):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_normalization = args.set_class_normalization
        self.box_normalization = args.set_box_normalization
        self.keypoint_normalization = args.set_keypoint_normalization
        self.oks_normalization = args.set_oks_normalization

    def forward(self, outputs, targets, weight_dict, num_box, num_pts,
        num_people):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            match_args

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with paddle.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            NORMALIZER = {'num_box': num_box, 'num_pts': num_pts,
                'num_people': num_people, 'mean': num_queries, 'none': 1,
                'box_average': num_box}
            with_boxes = True
            with_keypoints = ('loss_kps_l1' in weight_dict or 'loss_oks' in
                weight_dict)
            out_logit = outputs['pred_logits'].flatten(start_axis=0,
                stop_axis=1)
            out_prob = out_logit.sigmoid()
            if with_boxes:
                out_bbox = outputs['pred_boxes'].flatten(start_axis=0,
                    stop_axis=1)
            if with_keypoints:
                out_keypoints = outputs['pred_keypoints'].flatten(start_axis
                    =0, stop_axis=1)
            tgt_bbox = paddle.concat(x=[v['boxes'] for v in targets])
            sizes = [t['boxes'].shape[0] for t in targets]
            num_local = sum(sizes)
            if num_local == 0:
                return [(paddle.to_tensor(data=[]).astype('int64'), paddle.
                    to_tensor(data=[]).astype('int64')) for _ in sizes]
            assert ('loss_bce' in weight_dict) ^ ('loss_ce' in weight_dict)
            if 'loss_bce' in weight_dict:
                cost_class = -out_prob * weight_dict['loss_bce']
            elif 'loss_ce' in weight_dict:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 -
                    out_prob + 1e-08).log()
                pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob +
                    1e-08).log()
                cost_class = pos_cost_class - neg_cost_class
                cost_class = cost_class * weight_dict['loss_ce']
            cost_class = cost_class[..., None].tile(repeat_times=[1, num_local]
                )
            C = cost_class / NORMALIZER[self.class_normalization]
            if with_boxes:
>>>                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) / NORMALIZER[
                    self.box_normalization]
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(
                    out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) / NORMALIZER[
                    self.box_normalization]
                C_box = weight_dict['loss_bbox'] * cost_bbox + weight_dict[
                    'loss_giou'] * cost_giou
                C = C + C_box
            if with_keypoints:
                tgt_kps = paddle.concat(x=[v['keypoints'] for v in targets])
                tgt_visible = tgt_kps[..., -1]
                tgt_kps = tgt_kps[(...), :2]
                tgt_visible = (tgt_visible > 0) * (tgt_kps >= 0).all(axis=-1
                    ) * (tgt_kps <= 1).all(axis=-1)
                if 'loss_kps_l1' in weight_dict:
                    out_kps = out_keypoints.unsqueeze(axis=1)
                    tgt_kps_t = tgt_kps.unsqueeze(axis=0)
                    cost_kps_l1 = paddle.abs(x=out_kps - tgt_kps_t).sum(axis=-1
                        ) * tgt_visible
                    cost_kps_l1 = cost_kps_l1.sum(axis=-1)
                    if self.keypoint_normalization == 'box_average':
                        cost_kps_l1 = cost_kps_l1 / tgt_visible.sum(axis=-1
                            ).clip(min=1.0)
                    C_kps_l1 = weight_dict['loss_kps_l1'
                        ] * cost_kps_l1 / NORMALIZER[self.
                        keypoint_normalization]
                    C = C + C_kps_l1
                if 'loss_oks' in weight_dict:
                    cat_tgt_kps = paddle.concat(x=[tgt_kps, tgt_visible.
                        unsqueeze(axis=-1)], axis=-1)
                    cost_oks = -joint_oks(out_keypoints, cat_tgt_kps, tgt_bbox)
                    C_kps_oks = weight_dict['loss_oks'
                        ] * cost_oks / NORMALIZER[self.oks_normalization]
                    C = C + C_kps_oks
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            C = C.view(bs, num_queries, -1).cpu()
            """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.
                split(sizes, -1))]
            return [(paddle.to_tensor(data=i).astype('int64'), paddle.
                to_tensor(data=j).astype('int64')) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(args)

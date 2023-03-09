import paddle
from .unified_matcher import build_matcher
from .losses import sigmoid_focal_loss
from util import box_ops
from util.misc import nested_tensor_from_tensor_list, interpolate, get_world_size, is_dist_avail_and_initialized
import numpy as np
KPS_OKS_SIGMAS = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 
    0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0


class UnifiedSingleClassCriterion(paddle.nn.Layer):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args):
        """ Create the criterion.
        Parameters:
            args.MATCHER: module able to compute a matching between targets and proposals
            args.focal_alpha: dict containing as key the names of the losses and as values their relative weight.
            args.*_loss_coef
            args.*_normalization
            args.bce_negative_weight
        """
        super().__init__()
        self.matcher = build_matcher(args.MATCHER)
        self.focal_alpha = args.focal_alpha
        all_weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bce': args.
            cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou':
            args.giou_loss_coef, 'loss_kps_l1': args.keypoint_l1_loss_coef,
            'loss_oks': args.keypoint_oks_loss_coef}
        self.all_weight_dict = all_weight_dict
        self.bce_negative_weight = args.bce_negative_weight
        self.class_normalization = args.class_normalization
        self.box_normalization = 'num_box'
        self.keypoint_normalization = args.keypoint_normalization
        self.oks_normalization = args.oks_normalization

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_onehot = paddle.zeros(shape=[src_logits.shape[0],
            src_logits.shape[1]], dtype=src_logits.dtype)
        target_classes_onehot[idx] = 1
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,
            num_boxes=None, alpha=self.focal_alpha, gamma=2
            ) / self.loss_normalization[self.class_normalization]
        losses = {'loss_ce': loss_ce}
        if False:
            losses['class_error'] = 100
        return losses

    def loss_bce(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        srcs_idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits'].sigmoid()
        target_logits = paddle.zeros_like(x=src_logits)
        target_logits[srcs_idx] = 1.0
        weight_matrix = paddle.full_like(x=src_logits, fill_value=self.
            bce_negative_weight)
        weight_matrix[srcs_idx] = 1.0
>>>        loss_bce = torch.nn.functional.binary_cross_entropy(src_logits,
            target_logits, weight=weight_matrix, reduction='sum')
        loss_bce = loss_bce / self.loss_normalization[self.class_normalization]
        losses = {'loss_bce': loss_bce}
        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = paddle.concat(x=[t['boxes'][i] for t, (_, i) in zip(
            targets, indices)], axis=0)
>>>        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes,
            reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / self.loss_normalization[
            self.box_normalization]
        loss_giou = 1 - paddle.diag(x=box_ops.generalized_box_iou(box_ops.
            box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(
            target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / self.loss_normalization[
            self.box_normalization]
        return losses

    def loss_oks(self, outputs, targets, indices, with_center=True, eps=1e-15):
        idx = self._get_src_permutation_idx(indices)
        src_joints = outputs['pred_keypoints'][idx]
        tgt_joints = paddle.concat(x=[t['keypoints'][i] for t, (_, i) in
            zip(targets, indices)], axis=0)
        tgt_bboxes = paddle.concat(x=[t['boxes'][i] for t, (_, i) in zip(
            targets, indices)], axis=0)
        tgt_flags = tgt_joints[..., 2]
        tgt_joints = tgt_joints[(...), 0:2]
        tgt_flags = (tgt_flags > 0) * (tgt_joints >= 0).all(axis=-1) * (
            tgt_joints <= 1).all(axis=-1)
        tgt_wh = tgt_bboxes[(...), 2:]
        tgt_areas = tgt_wh[..., 0] * tgt_wh[..., 1]
        sigmas = KPS_OKS_SIGMAS
>>>        sigmas = torch.tensor(sigmas).astype(dtype=tgt_joints.dtype)
        d_sq = paddle.square(x=src_joints - tgt_joints).sum(axis=-1)
        loss_oks = 1 - paddle.exp(x=-1 * d_sq / (2 * tgt_areas[:, (None)] *
            sigmas[(None), :] + 1e-15))
        loss_oks = loss_oks * tgt_flags
        loss_oks = loss_oks.sum(axis=-1) / (tgt_flags.sum(axis=-1) + eps)
        loss_oks = loss_oks.sum() / (self.loss_normalization[self.
            oks_normalization] + eps)
        losses = {'loss_oks': loss_oks}
        return losses

    def loss_keypoints(self, outputs, targets, indices):
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_kps = outputs['pred_keypoints'][idx]
        target_kps = paddle.concat(x=[t['keypoints'][i] for t, (_, i) in
            zip(targets, indices)], axis=0)
        tgt_kps = target_kps[(...), :2]
        tgt_visible = (target_kps[..., 2] > 0) * (tgt_kps >= 0).all(axis=-1
            ) * (tgt_kps <= 1).all(axis=-1)
        src_loss, tgt_loss = src_kps[tgt_visible], tgt_kps[tgt_visible]
>>>        loss_keypoint = torch.nn.functional.l1_loss(src_loss, tgt_loss,
            reduction='sum')
        loss_keypoint = loss_keypoint / self.loss_normalization[self.
            keypoint_normalization]
        return {'loss_kps_l1': loss_keypoint}

    def _get_src_permutation_idx(self, indices):
        batch_idx = paddle.concat(x=[paddle.full_like(x=src, fill_value=i) for
            i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat(x=[src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = paddle.concat(x=[paddle.full_like(x=tgt, fill_value=i) for
            i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat(x=[tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {'labels': self.loss_labels, 'bce': self.loss_bce,
            'boxes': self.loss_boxes, 'keypoints': self.loss_keypoints,
            'oks': self.loss_oks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices)

    def build_weight_dict(self, losses):
        weight_dict = {}
        if 'labels' in losses:
            weight_dict['loss_ce'] = self.all_weight_dict['loss_ce']
        if 'bce' in losses:
            weight_dict['loss_bce'] = self.all_weight_dict['loss_bce']
        if 'boxes' in losses:
            weight_dict['loss_bbox'] = self.all_weight_dict['loss_bbox']
            weight_dict['loss_giou'] = self.all_weight_dict['loss_giou']
        if 'keypoints' in losses:
            weight_dict['loss_kps_l1'] = self.all_weight_dict['loss_kps_l1']
        if 'oks' in losses:
            weight_dict['loss_oks'] = self.all_weight_dict['loss_oks']
        return weight_dict

    def forward(self, outputs, targets, losses, num_boxes, num_pts, num_people
        ):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                  pred_logits: bs, nobj
                  pred_boxes:  bs, nobj, 4
                  (optional):  bs, nobj, mngts
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                  keypoints: ngts, 17, 3
        """
        weight_dict = self.build_weight_dict(losses)
        self.loss_normalization = {'num_box': num_boxes, 'num_pts': num_pts,
            'num_people': num_people, 'mean': outputs['pred_logits'].shape[
            1], 'none': 1}
        indices = self.matcher(outputs, targets, weight_dict, num_boxes,
            num_pts, num_people)
        loss_dict = {}
        for loss in losses:
            loss_dict.update(self.get_loss(loss, outputs, targets, indices))
        return self.rescale_loss(loss_dict, weight_dict)

    def rescale_loss(self, loss_dict, weight_dict):
        return {k: (loss_dict[k] * weight_dict[k]) for k in loss_dict.keys(
            ) if k in weight_dict}

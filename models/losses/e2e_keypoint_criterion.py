import paddle
from .matcher_kps import build_matcher
from .losses import sigmoid_focal_loss
from util import box_ops
from util.misc import nested_tensor_from_tensor_list, interpolate, get_world_size, is_dist_avail_and_initialized
import numpy as np
KPS_OKS_SIGMAS = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 
    0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0


class FixedMatcher(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
>>>        self.matcher_train = torch.load(args.fix_match_train)
>>>        self.matcher_val = torch.load(args.fix_match_val)

    def forward(self, outputs, targets, num_box, num_pts, save_print=False):
        if self.training:
            return [self.matcher_train[itgt['image_id']] for itgt in targets]
        else:
            return [self.matcher_val[itgt['image_id']] for itgt in targets]


class KeypointSetCriterion(paddle.nn.Layer):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        if args.MATCHER.fix_match_train:
            self.matcher = FixedMatcher(args.MATCHER)
        else:
            self.matcher = build_matcher(args.MATCHER)
        self.losses = args.losses
        self.focal_alpha = args.focal_alpha
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bce': args.
            cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou':
            args.giou_loss_coef, 'loss_kps_l1': args.keypoint_l1_loss_coef,
            'loss_oks': args.keypoint_oks_loss_coef}
        self.weight_dict = weight_dict
        self.class_normalization = args.class_normalization
        self.keypoint_normalization = args.keypoint_normalization
        self.bce_negative_weight = args.bce_negative_weight
        self.keypoint_reference = args.keypoint_reference
        assert self.keypoint_reference in ['absolute', 'relative']

    def loss_labels(self, outputs, targets, indices, num_boxes, num_pts):
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
            num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if False:
            losses['class_error'] = 100
        return losses

    def loss_bce(self, outputs, targets, indices, num_boxes, num_pts):
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

    def loss_boxes(self, outputs, targets, indices, num_boxes, num_pts):
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
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - paddle.diag(x=box_ops.generalized_box_iou(box_ops.
            box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(
            target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_oks(self, outputs, targets, indices, num_boxes, num_pts,
        with_center=True, eps=1e-15):
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
        loss_oks = loss_oks.sum() / (num_boxes + eps)
        losses = {'loss_oks': loss_oks}
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes, num_pts):
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_kps = outputs['pred_keypoints'][idx]
        target_kps = paddle.concat(x=[t['keypoints'][i] for t, (_, i) in
            zip(targets, indices)], axis=0)
        tgt_kps = target_kps[(...), :2]
        tgt_visible = (target_kps[..., 2] > 0) * (tgt_kps >= 0).all(axis=-1
            ) * (tgt_kps <= 1).all(axis=-1)
        if self.keypoint_reference == 'relative':
            target_boxes = paddle.concat(x=[t['boxes'][i] for t, (_, i) in
                zip(targets, indices)], axis=0)
            bbox_wh = target_boxes[(...), 2:].unsqueeze(axis=1)
            src_kps, tgt_kps = src_kps / bbox_wh, tgt_kps / bbox_wh
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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, num_pts):
        loss_map = {'labels': self.loss_labels, 'bce': self.loss_bce,
            'boxes': self.loss_boxes, 'keypoints': self.loss_keypoints,
            'oks': self.loss_oks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, num_pts)

    def forward(self, outputs, targets, save_print=False):
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
        num_boxes = sum(t['keypoints'].shape[0] for t in targets)
        num_boxes = paddle.to_tensor(data=[num_boxes], place=next(iter(
            outputs.values())).place).astype('float32')
        if is_dist_avail_and_initialized():
>>>            torch.distributed.all_reduce(num_boxes)
        num_boxes = paddle.clip(x=num_boxes / get_world_size(), min=1).item()
        kps = paddle.concat(x=[t['keypoints'] for t in targets], axis=0)
        kps = (kps[..., 2] > 0) * (kps[(...), :2] >= 0).all(axis=-1) * (kps
            [(...), :2] <= 1).all(axis=-1)
        num_pts = kps.sum()
        num_pts = paddle.to_tensor(data=num_pts, place=next(iter(outputs.
            values())).place).astype('float32')
        if is_dist_avail_and_initialized():
>>>            torch.distributed.all_reduce(num_pts)
        num_pts = paddle.clip(x=num_pts / get_world_size(), min=1).item()
        self.loss_normalization = {'num_box': num_boxes, 'num_pts': num_pts,
            'mean': outputs['pred_logits'].shape[1], 'none': 1}
        outputs_without_aux = {k: v for k, v in outputs.items() if k !=
            'aux_outputs' and k != 'enc_outputs'}
        unavail_mask = self.build_unavail_mask(outputs, targets
            ) if 'match_mask' in outputs else None
        indices = self.matcher(outputs_without_aux, targets, num_boxes,
            num_pts, save_print=save_print and self.training)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                num_boxes, num_pts))
        return self.rescale_loss(losses)

    def rescale_loss(self, loss_dict):
        return {k: (loss_dict[k] * self.weight_dict[k]) for k in loss_dict.
            keys() if k in self.weight_dict}

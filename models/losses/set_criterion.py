import paddle
from .matcher import build_matcher
from .losses import sigmoid_focal_loss
from util import box_ops
from util.misc import nested_tensor_from_tensor_list, interpolate, get_world_size, is_dist_avail_and_initialized
import numpy as np


class SetCriterion(paddle.nn.Layer):
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
        self.num_classes = 80
        self.matcher = build_matcher(args.MATCHER)
        self.weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox':
            args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
        self.losses = ['labels', 'boxes']
        self.focal_alpha = args.focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(x=[t['labels'][J] for t, (_, J) in
            zip(targets, indices)])
        target_classes = paddle.full(shape=src_logits.shape[:2], fill_value
            =self.num_classes).astype('int64')
        target_classes[idx] = target_classes_o
        target_classes_onehot = paddle.zeros(shape=[src_logits.shape[0],
            src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits
            .dtype)
        """Class Method: *.scatter_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        target_classes_onehot.scatter_(2, target_classes.unsqueeze(axis=-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,
            num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                target_classes_o)[0]
        return losses

    @paddle.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.place
        tgt_lengths = paddle.to_tensor(data=[len(v['labels']) for v in
            targets], place=device)
        card_pred = (pred_logits.argmax(axis=-1) != pred_logits.shape[-1] - 1
            ).sum(axis=1)
        """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.float, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        card_err = torch.nn.functional.l1_loss(card_pred.float(),
            tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
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

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        target_masks, valid = nested_tensor_from_tensor_list([t['masks'] for
            t in targets]).decompose()
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        target_masks = target_masks.to(src_masks)
        src_masks = src_masks[src_idx]
        src_masks = interpolate(src_masks[:, (None)], size=target_masks.
            shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, (0)].flatten(start_axis=1)
        target_masks = target_masks[tgt_idx].flatten(start_axis=1)
        losses = {'loss_mask': sigmoid_focal_loss(src_masks, target_masks,
            num_boxes), 'loss_dice': dice_loss(src_masks, target_masks,
            num_boxes)}
        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.
            loss_cardinality, 'boxes': self.loss_boxes, 'masks': self.
            loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k !=
            'aux_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = paddle.to_tensor(data=[num_boxes], place=next(iter(
            outputs.values())).place).astype('float32')
        if is_dist_avail_and_initialized():
>>>            torch.distributed.all_reduce(num_boxes)
        num_boxes = paddle.clip(x=num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices,
                num_boxes, **kwargs))
        return self.rescale_loss(losses, self.weight_dict)

    def rescale_loss(self, loss_dict, weight_dict):
        return {k: (loss_dict[k] * weight_dict[k]) for k in loss_dict.keys(
            ) if k in weight_dict}

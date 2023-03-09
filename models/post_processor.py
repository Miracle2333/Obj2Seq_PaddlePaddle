import paddle
"""
PostProcessor for Obj2Seq
"""
from util import box_ops


class PostProcess(paddle.nn.Layer):
    """ This module converts the model's output into the format expected by the coco api"""

    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        topk_values, topk_indexes = paddle.topk(x=prob.view(out_logits.
            shape[0], -1), k=100, axis=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = paddle.take_along_axis(arr=boxes, axis=1, indices=
            topk_boxes.unsqueeze(axis=-1).tile(repeat_times=[1, 1, 4]))
        img_h, img_w = target_sizes.unbind(axis=1)
        scale_fct = paddle.stack(x=[img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_fct[:, (None), :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in
            zip(scores, labels, boxes)]
        return results


class MutiClassPostProcess(paddle.nn.Layer):

    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        img_h, img_w = target_sizes.unbind(axis=1)
        scale_fct = paddle.stack(x=[img_w, img_h, img_w, img_h], axis=1)
        if 'detection' in outputs:
            output = outputs['detection']
            bs_idx, cls_idx = output['batch_index'], output['class_index']
            box_scale = scale_fct[bs_idx]
            all_scores = output['pred_logits'].sigmoid()
            nobj = all_scores.shape[-1]
            all_boxes = box_ops.box_cxcywh_to_xyxy(output['pred_boxes']
                ) * box_scale[:, (None), :]
            results_det = []
            """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            for id_b in bs_idx.unique():
                out_scores = all_scores[bs_idx == id_b].flatten()
                out_bbox = all_boxes[bs_idx == id_b].flatten(start_axis=0,
                    stop_axis=1)
                """Class Method: *.expand, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>                out_labels = output['class_index'][bs_idx == id_b].unsqueeze(
                    axis=-1).expand(-1, nobj).flatten()
                s, indices = out_scores.sort(descending=True)
                s, indices = s[:100], indices[:100]
                results_det.append({'scores': s, 'labels': out_labels[
                    indices], 'boxes': out_bbox[(indices), :]})
            return results_det
        if 'pose' in outputs:
            output = outputs['pose']
            bs_idx, cls_idx = output['batch_index'], output['class_index']
            box_scale = scale_fct[bs_idx]
            all_scores = output['pred_logits'].sigmoid()
            nobj = all_scores.shape[-1]
            all_keypoints = output['pred_keypoints'] * box_scale[:, (None),
                (None), :2]
            all_keypoints = paddle.concat(x=[all_keypoints, paddle.
                ones_like(x=all_keypoints)[(...), :1]], axis=-1)
            all_boxes = box_ops.box_cxcywh_to_xyxy(output['pred_boxes']
                ) * box_scale[:, (None), :]
            results_det = []
            """Class Method: *.unique, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            for id_b in bs_idx.unique():
                out_scores = all_scores[bs_idx == id_b].flatten()
                out_bbox = all_boxes[bs_idx == id_b].flatten(start_axis=0,
                    stop_axis=1)
                out_keypoints = all_keypoints[bs_idx == id_b].flatten(
                    start_axis=0, stop_axis=1)
                out_labels = paddle.zeros_like(x=out_scores).astype('int64')
                s, indices = out_scores.sort(descending=True)
                s, indices = s[:100], indices[:100]
                results_det.append({'scores': s, 'labels': out_labels[
                    indices], 'boxes': out_bbox[indices], 'keypoints':
                    out_keypoints[(indices), :]})
            return results_det


class KeypointPostProcess(paddle.nn.Layer):

    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs(Dict):
                pred_logits: Tensor [bs, nobj] (Currently support [bs, 1, nobj] too, may deprecated Later)
                pred_keypoints: Tensor [bs, nobj, 17, 2] (Currently support "keypoint_offsets" too)
                pred_boxes: Tensor [bs, nobj, 4] (Currently support [bs, 1, nobj, 4] too, may deprecated Later)
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if 'pred_keypoints' in outputs:
            out_logits, out_keypoints = outputs['pred_logits'].squeeze(axis=1
                ), outputs['pred_keypoints']
        else:
            out_logits, out_keypoints = outputs['pred_logits'].squeeze(axis=1
                ), outputs['keypoint_offsets']
        bs, num_obj = out_logits.shape
        scores = out_logits.sigmoid()
        labels = paddle.zeros_like(x=scores).astype('int64')
        img_h, img_w = target_sizes.unbind(axis=1)
        scale_fct = paddle.stack(x=[img_w, img_h], axis=1)
        out_keypoints = out_keypoints * scale_fct[:, (None), (None), :]
        ones = paddle.ones_like(x=out_keypoints)[(...), :1]
        keypoints = paddle.concat(x=[out_keypoints, ones], axis=-1)
        if 'pred_boxes' in outputs:
            out_bbox = outputs['pred_boxes'].squeeze(axis=1)
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            scale_fct = paddle.concat(x=[scale_fct, scale_fct], axis=1)
            boxes = boxes * scale_fct[:, (None), :]
        else:
            boxes = None
        results = []
        for idb, (s, l, k) in enumerate(zip(scores, labels, keypoints)):
            s, indices = s.sort(descending=True)
            s, indices = s[:100], indices[:100]
            results.append({'scores': s, 'labels': l[indices], 'keypoints':
                k[(indices), :]})
            if boxes is not None:
                results[-1]['boxes'] = boxes[(idb), (indices), :]
        return results


def build_postprocessor(args):
    if args.EVAL.postprocessor == 'MultiClass':
        postprocessor = MutiClassPostProcess()
    elif args.EVAL.postprocessor == 'Detr':
        postprocessor = PostProcess()
    elif args.EVAL.postprocessor == 'Keypoint':
        postprocessor = KeypointPostProcess()
    else:
        raise KeyError
    return postprocessor

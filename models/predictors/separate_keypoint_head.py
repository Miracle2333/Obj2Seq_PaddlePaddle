import paddle
import numpy as np
from util.misc import inverse_sigmoid
from .classifiers import build_label_classifier
from models.losses.e2e_keypoint_criterion import KeypointSetCriterion


class SeparateKeypointHead(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        d_model = args.CLASSIFIER.hidden_dim
        self.points_per_query = args.CLASSIFIER.num_points
        assert self.points_per_query == 1
        self.class_embed = build_label_classifier(args.CLASSIFIER)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.output_layer = MLP(d_model, d_model, 36, 3)
        self.keypoint_output = args.keypoint_output
        self.keypoint_relative_ratio = args.LOSS.keypoint_relative_ratio
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
>>>        torch.nn.init.constant_(self.output_layer.layers[-1].weight.data, 0)
>>>        torch.nn.init.constant_(self.output_layer.layers[-1].bias.data, 0)
        self.criterion = KeypointSetCriterion(args.LOSS)

    def forward(self, feat, query_pos, reference_points, srcs,
        src_padding_masks, **kwargs):
        if feat.dim() == 3:
            bs, nobj, c = feat.shape
        elif feat.dim() == 4:
            bs, cs, nobj, c = feat.shape
            if reference_points.dim() == 3:
                reference_points = reference_points.unsqueeze(axis=1)
        else:
            raise KeyError
        class_vector, cls_idx = kwargs.pop('class_vector', None), kwargs.pop(
            'cls_idx', None)
        outputs_class = self.class_embed(feat, class_vector=class_vector,
            cls_idx=cls_idx)
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        feat = feat.view(bs, nobj, c)
        tmp = self.bbox_embed(feat)
        reference_points = reference_points.squeeze(axis=1)
        """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        tmp_center, tmp_offsets = self.output_layer(feat).split([2, 34], dim=-1
            )
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        tmp_center = tmp_center.reshape(bs, nobj, 1, 2)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        tmp_offsets = tmp_offsets.reshape(bs, nobj, 17, 2)
        reference = inverse_sigmoid(reference_points)
        if reference.shape[-1] == 4:
            tmp = tmp + reference
        else:
            assert reference.shape[-1] == 2
            tmp = tmp + paddle.concat(x=[reference, paddle.zeros_like(x=
                reference)], axis=-1)
        outputs_coord = tmp.sigmoid()
        keypoint_coords = self.generate_keypoint_outputs(tmp_center,
            tmp_offsets, outputs_coord, reference_points)
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        outputs = {'pred_logits': outputs_class.view(bs, nobj),
            'pred_boxes': outputs_coord.view(bs, nobj, 4), 'pred_keypoints':
            keypoint_coords.view(bs, nobj, 17, 2)}
        targets = kwargs.pop('targets', None)
        targets = [({'image_id': tgt['image_id'].item(), 'boxes': tgt[0][
            'boxes'], 'keypoints': tgt[0]['keypoints']} if 0 in tgt else {
            'image_id': tgt['image_id'].item(), 'boxes': paddle.zeros(shape
            =(0, 4)), 'keypoints': paddle.zeros(shape=(0, 17, 3))}) for tgt in
            targets]
        loss_dict = self.criterion(outputs, targets)
        return outputs, loss_dict

    def generate_keypoint_outputs(self, center_logits, offset_logits, boxes,
        reference_points):
        boxes = boxes.unsqueeze(axis=-2)
        if self.keypoint_output == 'nd_box_relative':
            return boxes[(...), :2] + offset_logits * boxes[(...), 2:]
        return coords


class MLP(paddle.nn.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
        count_per_query=1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.LayerList(sublayers=(paddle.nn.Linear(
            in_features=n, out_features=k) for n, k in zip([input_dim] + h,
            h + [output_dim])))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = paddle.nn.functional.relu(x=layer(x)
                ) if i < self.num_layers - 1 else layer(x)
        return x

import paddle
import numpy as np
from util.misc import inverse_sigmoid
from .classifiers import build_label_classifier
from models.losses.classwise_criterion import ClasswiseCriterion
from models.losses.set_criterion import SetCriterion


class SeparateDetectHead(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        d_model = args.CLASSIFIER.hidden_dim
        self.points_per_query = args.CLASSIFIER.num_points
        assert self.points_per_query == 1
        self.class_embed = build_label_classifier(args.CLASSIFIER)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.reset_parameters_as_first_head()
        self.criterion = SetCriterion(args.LOSS
            ) if 'multi' in args.CLASSIFIER.type else ClasswiseCriterion(args
            .LOSS)

    def reset_parameters_as_first_head(self):
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

    def reset_parameters_as_refine_head(self):
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
>>>        torch.nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, feat, query_pos, reference_points, srcs,
        src_padding_masks, **kwargs):
        class_vector, cls_idx = kwargs.pop('class_vector', None), kwargs.pop(
            'cls_idx', None)
        reference = inverse_sigmoid(reference_points)
        outputs_class = self.class_embed(feat, class_vector=class_vector,
            cls_idx=cls_idx)
        tmp = self.bbox_embed(feat)
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[(...), :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs = {'pred_logits': outputs_class.unsqueeze(axis=-1) if feat.
            dim() == 4 else outputs_class, 'pred_boxes': outputs_coord,
            'class_index': cls_idx if cls_idx is not None else paddle.zeros
            (shape=(bs, 1), dtype='int64')}
        targets = kwargs.pop('targets', None)
        if targets is not None:
            detection_loss_dict = self.criterion(outputs, targets)
        else:
            assert not self.training, 'Targets are required for training mode (separate_detect_head.py)'
            detection_loss_dict = {}
        return outputs, detection_loss_dict


class MLP(paddle.nn.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
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

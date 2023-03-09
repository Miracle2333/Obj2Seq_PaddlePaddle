import paddle
import math


class DetrClassifier(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        if args.num_layers > 0:
            self.feature_linear = paddle.nn.LayerList(sublayers=[paddle.nn.
                Linear(in_features=args.hidden_dim, out_features=args.
                hidden_dim) for i in range(args.num_layers)])
        else:
            self.feature_linear = None
        self.classifier = paddle.nn.Linear(in_features=args.hidden_dim,
            out_features=80)
        self.reset_parameters(args.init_prob)

    def reset_parameters(self, init_prob):
        prior_prob = init_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
>>>        torch.nn.init.constant_(self.classifier.bias.data, bias_value)

    def forward(self, x, class_vector=None, cls_idx=None):
        if self.feature_linear is not None:
            for i in range(self.num_layers):
                x = paddle.nn.functional.relu(x=self.feature_linear[i](x)
                    ) if i < self.num_layers - 1 else self.feature_linear[i](x)
        return self.classifier(x)

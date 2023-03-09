import paddle
import math


class AbstractClassifier(paddle.nn.Layer):

    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        if args.num_layers > 0:
            self.feature_linear = paddle.nn.LayerList(sublayers=[paddle.nn.
                Linear(in_features=args.hidden_dim, out_features=args.
                hidden_dim) for i in range(args.num_layers)])
            self.skip_and_init = args.skip_and_init
            if args.skip_and_init:
>>>                torch.nn.init.constant_(self.feature_linear[-1].weight, 0.0)
>>>                torch.nn.init.constant_(self.feature_linear[-1].bias, 0.0)
        else:
            self.feature_linear = None
        if True:
            self.bias = True
>>>            self.b = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters(args.init_prob)

    def reset_parameters(self, init_prob):
        if True:
            prior_prob = init_prob
            bias_value = -math.log((1 - prior_prob) / prior_prob)
>>>            torch.nn.init.constant_(self.b.data, bias_value)

    def forward(self, x, class_vector=None, cls_idx=None):
        if self.feature_linear is not None:
            skip = x
            for i in range(self.num_layers):
                x = paddle.nn.functional.relu(x=self.feature_linear[i](x)
                    ) if i < self.num_layers - 1 else self.feature_linear[i](x)
            if self.skip_and_init:
                x = skip + x
        new_feat = x
        assert x.dim() == 3
        W = self.getClassifierWeight(class_vector, cls_idx)
        sim = (x * W).sum(axis=-1)
        if True:
            sim = sim + self.b
        return sim


class LinearClassifier(AbstractClassifier):

    def __init__(self, args):
        super().__init__(args)
        self.hidden_dim = args.hidden_dim
>>>        self.W = torch.nn.Parameter(torch.Tensor(self.hidden_dim))
        stdv = 1.0 / math.sqrt(self.W.shape[0])
        self.W.data.uniform_(min=-stdv, max=stdv)

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        return self.W


class DictClassifier(AbstractClassifier):

    def __init__(self, args):
        super().__init__(args)
        self.scale = args.hidden_dim ** -0.5

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        W = class_vector * self.scale
        return W


def build_label_classifier(args):
    if args.type == 'linear':
        return LinearClassifier(args)
    elif args.type == 'dict':
        return DictClassifier(args)
    else:
        raise KeyError

import paddle


class AsymmetricLoss(paddle.nn.Layer):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08,
        disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, weights=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = paddle.nn.functional.sigmoid(x=x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clip(max=1)
        los_pos = y * paddle.log(x=xs_pos.clip(min=self.eps))
        los_neg = (1 - y) * paddle.log(x=xs_neg.clip(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                paddle.set_grad_enabled(mode=False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = paddle.pow(x=1 - pt, y=one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                paddle.set_grad_enabled(mode=True)
            loss *= one_sided_w
        if weights is not None:
            self.loss *= weights
        return -loss.sum() / x.shape[0] * (x.size() / weights.sum())


class AsymmetricLossOptimized(paddle.nn.Layer):
    """ Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-05,
        disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        (self.targets) = (self.anti_targets) = (self.xs_pos) = (self.xs_neg
            ) = (self.asymmetric_w) = (self.loss) = None

    def forward(self, x, y, weights=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = paddle.nn.functional.sigmoid(x=x)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>            self.xs_neg.add_(self.clip).clip_(max=1)
        self.loss = self.targets * paddle.log(x=self.xs_pos.clip(min=self.eps))
        """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        self.loss.add_(self.anti_targets * paddle.log(x=self.xs_neg.clip(
            min=self.eps)))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with paddle.no_grad():
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = paddle.pow(x=1 - self.xs_pos - self
                        .xs_neg, y=self.gamma_pos * self.targets + self.
                        gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = paddle.pow(x=1 - self.xs_pos - self.
                    xs_neg, y=self.gamma_pos * self.targets + self.
                    gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
        if weights is not None:
            self.loss *= weights
        _loss = -self.loss.sum() / x.shape[0] * (x.size() / weights.sum())
        return _loss


class ASLSingleLabel(paddle.nn.Layer):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float=0.1, reduction=
        'mean'):
        super(ASLSingleLabel, self).__init__()
        self.eps = eps
        self.logsoftmax = paddle.nn.LogSoftmax(axis=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        """
        num_classes = inputs.shape[-1]
        log_preds = self.logsoftmax(inputs)
        """Class Method: *.scatter_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        self.targets_classes = paddle.zeros_like(x=inputs).scatter_(1,
            target.astype(dtype='int64').unsqueeze(axis=1), 1)
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = paddle.exp(x=log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = paddle.pow(x=1 - xs_pos - xs_neg, y=self.gamma_pos *
            targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w
        if self.eps > 0:
            self.targets_classes = paddle.Tensor.add(y=self.eps / num_classes)
        loss = -self.targets_classes.multiply(y=log_preds)
        loss = loss.sum(axis=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

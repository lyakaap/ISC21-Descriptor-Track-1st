import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from timm.data.loader import create_loader


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        # self.crit = nn.CrossEntropyLoss()
        self.s = s
        self.margins = margins

    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels_ = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (labels_ * phi) + ((1.0 - labels_) * cosine)
        output *= self.s
        loss = self.crit(output, labels_)

        return loss


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features, labels=None):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ArcFaceCutMix(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceCutMix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        label_a, label_b = label
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label_a.view(-1, 1).long(), 1)
        one_hot.scatter_(1, label_b.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class SubCenterArcFace(nn.Module):
    """Implementation of
    `Sub-center ArcFace: Boosting Face Recognition
    by Large-scale Noisy Web Faces`_.
    .. _Sub-center ArcFace\: Boosting Face Recognition \
        by Large-scale Noisy Web Faces:
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature,
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        k: number of possible class centroids.
            Default: ``3``.
        eps (float, optional): operation accuracy.
            Default: ``1e-6``.
    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.
    Example:
        >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()
    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        k: int = 2,
        eps: float = 1e-6,
    ):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.k = k
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        self.threshold = math.pi - self.m

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "SubCenterArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"k={self.k},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.
        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        feats = F.normalize(input).unsqueeze(0).expand(self.k, *input.shape)  # k*b*f
        wght = F.normalize(self.weight, dim=1)  # k*f*c
        cos_theta = torch.bmm(feats, wght)  # k*b*f
        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*f
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        selected = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(selected.bool(), theta + self.m, theta))
        logits *= self.s

        return logits


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, reduction='mean'):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss


class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(
            0, input.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - \
            sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        # return output, origin_cos * self.s
        return output


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, theta_zero=math.pi/4, *args, **kwargs):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(
                self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(
                B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s
        return output


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, num_classes, s=30.0, m=0.5, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.s = s
        self.cos_m = math.cos(m)  # 0.87758
        self.sin_m = math.sin(m)  # 0.47943
        self.th = math.cos(math.pi - m)      # -0.87758
        self.mm = math.sin(math.pi - m) * m  # 0.23971
        self.num_classes = num_classes

    def forward(self, logits, labels):
        labels = F.one_hot(labels, self.num_classes).float()
        logits = logits.float()  # float16 to float32 (if used float16)
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # equals to **2
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = DenseCrossEntropy()(output, labels, self.reduction)
        return loss / 2


class CosFace(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', s=' + str(self.s) \
            + ', m=' + str(self.m) + ')'


def pc_softmax_func(logits, lb_proportion):
    assert logits.size(1) == len(lb_proportion)
    shape = [1, -1] + [1 for _ in range(len(logits.size()) - 2)]
    W = torch.tensor(lb_proportion).view(*shape).to(logits.device).detach()
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    exp = torch.exp(logits)
    pc_softmax = exp.div_((W * exp).sum(dim=1, keepdim=True))
    return pc_softmax


class PCSoftmax(nn.Module):

    def __init__(self, lb_proportion):
        super(PCSoftmax, self).__init__()
        self.weight = lb_proportion

    def forward(self, logits):
        return pc_softmax_func(logits, self.weight)


class PCSoftmaxCrossEntropyV1(nn.Module):
    def __init__(self, lb_proportion, ignore_index=255, reduction='mean'):
        super(PCSoftmaxCrossEntropyV1, self).__init__()
        self.weight = torch.tensor(lb_proportion).cuda().detach()
        self.nll = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits, label):
        shape = [1, -1] + [1 for _ in range(len(logits.size()) - 2)]
        W = self.weight.view(*shape).to(logits.device).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        wexp_sum = torch.exp(logits).mul(W).sum(dim=1, keepdim=True)
        log_wsoftmax = logits - torch.log(wexp_sum)
        loss = self.nll(log_wsoftmax, label)
        return loss


class PCSoftmaxCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, label, lb_proportion, reduction, ignore_index):
        # prepare label
        label = label.clone().detach()
        ignore = label == ignore_index
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(
            1, label.unsqueeze(1), 1).detach()

        shape = [1, -1] + [1 for _ in range(len(logits.size()) - 2)]
        W = torch.tensor(lb_proportion).view(*shape).to(logits.device).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_wsum = torch.exp(logits).mul_(W).sum(dim=1, keepdim=True)

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(lb_one_hot.size(1)), *b]
        lb_one_hot[mask] = 0

        ctx.mask = mask
        ctx.W = W
        ctx.lb_one_hot = lb_one_hot
        ctx.logits = logits
        ctx.exp_wsum = exp_wsum
        ctx.reduction = reduction
        ctx.n_valid = n_valid

        log_wsoftmax = logits - torch.log(exp_wsum)
        loss = -log_wsoftmax.mul_(lb_one_hot).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.mask
        W = ctx.W
        lb_one_hot = ctx.lb_one_hot
        logits = ctx.logits
        exp_wsum = ctx.exp_wsum
        reduction = ctx.reduction
        n_valid = ctx.n_valid

        wlabel = torch.sum(W * lb_one_hot, dim=1, keepdim=True)
        wscores = torch.exp(logits).div_(exp_wsum).mul_(wlabel)
        wscores[mask] = 0
        grad = wscores.sub_(lb_one_hot)

        if reduction == 'none':
            grad.mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad.mul_(grad_output)
        elif reduction == 'mean':
            grad.div_(n_valid).mul_(grad_output)
        return grad, None, None, None, None, None


class PCSoftmaxCrossEntropyV2(nn.Module):

    def __init__(self, lb_proportion, reduction='mean', ignore_index=-100):
        super(PCSoftmaxCrossEntropyV2, self).__init__()
        self.lb_proportion = lb_proportion
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        return PCSoftmaxCrossEntropyFunction.apply(
            logits, label, self.lb_proportion, self.reduction, self.ignore_index)

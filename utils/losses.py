import torch as t
from torch import nn
import torch.nn.functional as F
import math

from torch.nn.modules.loss import KLDivLoss

from .metrics import Metric_for_Loss
# from metrics import Metric_for_Loss


class BiCutLoss(nn.Module):
    """bicut对应的loss
    
    """

    def __init__(self, alpha: float = 0.65, r: float = 0.0971134020, metric: str = 'nci'):
        super(BiCutLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.r = r

    def slice_index(self, out_tensor):
        """
        0对应truncation，1对应continue
        """
        temp = t.argmax(out_tensor, dim=1)
        ones_tensor = t.ones_like(temp)
        if temp.equal(ones_tensor): return temp.shape[0]
        length = temp.shape[0] - 1
        return length - t.argmin(t.flip(temp, dims=[0]))

    def forward(self, output: t.Tensor, labels: t.Tensor):
        mask = t.ones_like(output)
        for i in range(mask.shape[0]):
            mask[i][self.slice_index(output[i]) + 1:] = 0
        r = t.ones_like(output)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if self.metric == 'nci':
                    r[i][j] = t.tensor([0, -1 / math.log2(j + 2)]) if labels[i][j] == 1 else t.tensor(
                        [0, (j + 1) / self.alpha])
                else:
                    r[i][j] = t.tensor([(1 - self.alpha) / self.r, 0]) if labels[i][j] == 1 else t.tensor(
                        [0, self.alpha / (1 - self.r)])

        mask_output = output.mul(mask)
        loss_matrix = mask_output.mul(r)
        return t.sum(loss_matrix).div(output.shape[0])


class ChoopyLoss(nn.Module):
    """Choopy对应的loss

    """

    def __init__(self, metric: str = 'f1'):
        super(ChoopyLoss, self).__init__()
        self.metric = metric

    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j + 1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j + 1)

        loss_matrix = output.squeeze().mul(r)
        return -t.sum(loss_matrix).div(output.shape[0])

def sigmoid_focal_loss(
    inputs: t.Tensor,
    targets: t.Tensor,
    alpha: float = 0.25,  # 0.25, -1
    gamma: float = 2,
    reduction: str = "mean",
) -> t.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()  # (B, C)
    targets = targets.float()  # (B, C)
    p = t.sigmoid(inputs)  # (B, C)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (B, C)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)  # (B, C)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # # (B, C)
        loss = alpha_t * loss # (B, C)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class TruncationLoss(nn.Module):
    def __init__(self, metric: str = 'f1', tau: float = 0.95):
        super(TruncationLoss, self).__init__()
        self.metric = metric
        self.tau = tau

    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j + 1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j + 1)
        q = F.softmax(r.div(self.tau))

        soft_max = t.log(output.squeeze(2).div(self.tau))

        loss_matrix = soft_max.mul(q)
        return -t.sum(loss_matrix).div(output.shape[0])

class MultiViewLoss(nn.Module):
    def __init__(self):
        super(MultiViewLoss, self).__init__()
        self.loss_function = nn.BCELoss()

    def forward(self, output: t.Tensor, labels: t.Tensor):
        p = output.squeeze(1)
        return self.loss_function(p, labels).div(output.shape[0])

class MileCutLoss(nn.Module):
    def __init__(self, coefficient=0.5, metric: str = 'f1', tau: float = 0.95, view_num=3, view_no=None, rm_view_no=None):
        super(MileCutLoss, self).__init__()
        self.coefficient = coefficient
        self.view_num = view_num
        self.view_no = view_no
        self.rm_view_no = rm_view_no
        self.truncation_loss = TruncationLoss(metric=metric, tau=tau)
        self.multi_view_loss = MultiViewLoss()

    def forward(self, truncation_output, view_1_output, view_2_output, view_3_output, labels):
        view_1_loss = self.multi_view_loss(view_1_output.squeeze(2), labels)
        view_2_loss = self.multi_view_loss(view_2_output.squeeze(2), labels)
        view_3_loss = self.multi_view_loss(view_3_output.squeeze(2), labels)
        truncation_loss = self.truncation_loss(truncation_output, labels)
        loss = self.coefficient * truncation_loss + (1 - self.coefficient) * (view_1_loss + view_2_loss + view_3_loss)
        return loss

class AttnCutLoss(nn.Module):
    """AttnCut对应的loss
    """

    def __init__(self, metric: str = 'f1', tau: float = 0.95):
        super(AttnCutLoss, self).__init__()
        self.metric = metric
        self.tau = tau

    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j + 1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j + 1)
        q = t.exp(r.div(self.tau))
        norm_factor = t.sum(q, axis=1).unsqueeze(dim=1)
        q = q.div(norm_factor)
        # q = F.softmax(r.div(self.tau))

        output = t.log(output.squeeze())
        loss_matrix = output.mul(q)
        return -t.sum(loss_matrix).div(output.shape[0])


class RerankLoss(nn.Module):
    """
    Creates a criterion that measures rank hinge loss.
    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`labels` (containing 1 or -1).
    If :math:`labels = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`labels = -1`.
    The loss function for each sample in the mini-batch is:
    .. math::
        loss_{output, labels} = max(0, -labels * (x1 - x2) + margin)
    """

    def __init__(self, margin: float = 5e-4, reduction: str = 'mean'):
        """
        :class:`RankHingeLoss` constructor.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super(RerankLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output: t.Tensor, labels: t.Tensor):
        """
        Calculate rank hinge loss.
        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.
        """
        y_rele = labels == 1.
        y_irre = labels == 0.
        total_rele = y_rele.sum().item()
        total_irre = y_irre.sum().item()
        if total_rele == 0 or total_irre == 0: return t.tensor(0, requires_grad=True)
        y_pos_mean = y_rele.mul(output.squeeze()).sum().div(total_rele)
        y_neg_mean = y_irre.mul(output.squeeze()).sum().div(total_irre)
        return max(t.tensor(0., requires_grad=True), y_neg_mean - y_pos_mean + self.margin)
        # y_pos, y_neg = [], []
        # n_pos, n_neg = 0, 0
        # for sample_pred, sample_label in zip(output, labels):
        #     for i, label in enumerate(sample_label):
        #         if label: 
        #             y_pos.append(sample_pred[i])
        #             n_pos += 1
        #         else: 
        #             y_neg.append(sample_pred[i])
        #             n_neg += 1
        # total_rele = sum(y_pos)
        # if total_rele == 0 or total_rele == len(y_pos): return t.tensor(0., requires_grad=True)
        # y_pos_1D = t.tensor(y_pos, requires_grad=True).unsqueeze(-1).expand(-1, n_neg).flatten()
        # y_neg_1D = t.tensor(y_neg, requires_grad=True).repeat(n_pos)
        # y_true = t.ones_like(y_pos_1D)
        # return F.margin_ranking_loss(
        #     y_pos_1D, y_neg_1D, y_true,
        #     margin=self.margin,
        #     reduction=self.reduction
        # )
class LeCutLoss(nn.Module):
    """LeCut对应的loss

    """
    def __init__(self, metric: str='f1', tau: float=0.95):
        super(LeCutLoss, self).__init__()
        self.metric = metric
        self.tau = tau
    
    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j+1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j+1)
        
        # attncut loss
        q = t.exp(r.div(self.tau))
        norm_factor = t.sum(q, axis=1).unsqueeze(dim=1)
        q = q.div(norm_factor)

        # cross entropy
        # q = (r == r.max(dim=1, keepdim=True)[0]).to(dtype=t.int32)

        # choppy
        # loss_matrix = output.squeeze().mul(r)
        
        output = t.log(output.squeeze())
        loss_matrix = output.mul(q)
        return -t.sum(loss_matrix).div(output.shape[0])
        # return (-t.sum(loss_matrix).div(output.shape[0])-4.5)*100

class MtCutLoss(nn.Module):
    """MtCut的loss，尝试加入分类loss

    Args:
        nn ([type]): [description]
    """

    def __init__(self, metric: str = 'f1', rerank_weight: float = 0.5, classi_weight: float = 0.5,
                 num_tasks: float = 3):
        super(MtCutLoss, self).__init__()
        self.rerank_weight, self.classi_weight = rerank_weight, classi_weight
        self.weights = nn.Parameter(t.randn(int(num_tasks)), requires_grad=True)
        # self.cutloss = AttnCutLoss(metric=metric)
        self.cutloss = DivLoss(metric=metric, div_type='js', augmented=True)
        self.rerankloss = RerankLoss()
        self.classiloss = nn.BCELoss()
        self.num_tasks = num_tasks

    def forward(self, output: t.Tensor, labels: t.Tensor):
        if self.num_tasks == 3:
            pred_y, rerank_y, cut_y = output
        elif self.num_tasks == 2.1:
            pred_y, cut_y = output
        else:
            rerank_y, cut_y = output
        class_label = rerank_label = cut_label = labels
        cutloss = self.cutloss(cut_y, cut_label)
        if self.num_tasks == 3 or self.num_tasks == 2.2: rerankloss = self.rerankloss(rerank_y, rerank_label).mul(
            self.rerank_weight)
        if self.num_tasks == 3 or self.num_tasks == 2.1: classiloss = self.classiloss(pred_y.squeeze(),
                                                                                      class_label).mul(
            self.classi_weight)
        # print('cutloss: {} | rerankloss: {} | classify_loss: {}'.format(cutloss, rerankloss, classiloss))
        if self.num_tasks == 3:
            return cutloss.add(rerankloss).add(classiloss)
        elif self.num_tasks == 2.1:
            return cutloss.add(classiloss)
        else:
            return cutloss.add(rerankloss)


class DivLoss(nn.Module):
    """基于奖励分布KL散度的loss

    Args:
        nn ([type]): [description]
    """

    def __init__(self, metric: str = 'f1', tau: float = 0.85, div_type: str = 'kl', augmented: bool = True):
        """初始化分布损失函数

        Args:
            metric (str, optional): 目标奖励的metric. Defaults to 'f1'.
            tau (float, optional): 控制目标奖励缩放的超参数. Defaults to 0.95.
            div_type (str, optional): 散度计算方式. Defaults to 'kl'.
        """
        super(DivLoss, self).__init__()
        self.metric = metric
        self.KLDiv = t.nn.KLDivLoss(reduction='batchmean')
        self.div_type = div_type
        self.augmented = augmented
        if self.augmented:
            self.tau = tau
        else:
            self.tau = 1.

    def forward(self, output: t.Tensor, labels: t.Tensor):
        r = t.ones_like(output.squeeze(2))
        if self.metric == 'f1':
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.f1(labels[i], j + 1)
        else:
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    r[i][j] = Metric_for_Loss.dcg(labels[i], j + 1)
        q = t.exp(r.div(self.tau))
        norm_factor = t.sum(q, axis=1).unsqueeze(dim=1)
        q = q.div(norm_factor)

        if self.div_type == 'kl':
            return self.KLDiv(output.squeeze().log(), q)
        else:
            log_mean = output.squeeze().add(q).div(2).log()
            return self.KLDiv(log_mean, q).add(self.KLDiv(log_mean, output.squeeze())).div(2)


class WassDistLoss(nn.Module):
    """基于Wasserstein距离的损失函数

    Args:
        nn ([type]): [description]
    """

    def __init__(self, eps: float = 1e-3, max_iter: int = 100, metric: str = 'f1', tau: float = 0.95, reduction='mean'):
        """wasserstein损失函数初始化

        Args:
            eps ([type]): regularization coefficient
            max_iter ([type]): maximum number of Sinkhorn iterations
            metric (str, optional): [description]. Defaults to 'f1'.
            tau (float, optional): [description]. Defaults to 0.95.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to 'mean'.
        """
        super(WassDistLoss, self).__init__()
        self.metric = metric
        self.tau = tau
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, output: t.Tensor, labels: t.Tensor):
        # The Sinkhorn algorithm takes as input three variables :
        output = output.squeeze()
        C = self._cost_matrix(output, labels)  # Wasserstein cost function
        x_points = output.shape[-2]
        y_points = labels.shape[-2]
        batch_size = 1 if output.dim() == 2 else output.shape[0]

        # both marginals are fixed with equal weights
        mu = t.empty(batch_size, x_points, dtype=t.float, requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = t.empty(batch_size, y_points, dtype=t.float, requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = t.zeros_like(mu)
        v = t.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached, Stopping criterion
        actual_nits, thresh = 0, 1e-1

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (t.log(mu + 1e-8) - t.logsumexp(self.M(C, u, v), dim=-1)).mul(self.eps).add(u)
            v = (t.log(nu + 1e-8) - t.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)).mul(self.eps).add(v)
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh: break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = t.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = t.sum(pi * C, dim=(-2, -1))

        return cost.mean() if self.reduction == 'mean' else cost.sum()

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(output, labels, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = output.unsqueeze(-2)
        y_lin = labels.unsqueeze(-3)
        C = t.sum((t.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


if __name__ == '__main__':
    a = t.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], requires_grad=True).unsqueeze(dim=2)
    # a_1 = t.tensor([[[0.1, 0.9], [0.2, 0.8], [0.4, 0.6]], [[0.12, 0.88], [0.8, 0.2], [0.4, 0.6]]])
    b = t.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    detection_output = t.Tensor([[[0.5082],
             [0.4337],
             [0.4446],
             [0.6253],
             [0.6411],
             [0.2607],
             [0.7079],
             [0.5271],
             [0.5799],
             [0.7577]],
            [[0.5652],
             [0.4020],
             [0.4157],
             [0.6619],
             [0.6143],
             [0.3620],
             [0.7144],
             [0.4409],
             [0.5693],
             [0.6969]]])
    truncation_output = t.Tensor([[[0.0691],
         [0.0585],
         [0.0842],
         [0.0891],
         [0.1167],
         [0.0663],
         [0.0911],
         [0.1200],
         [0.1207],
         [0.1842]],
        [[0.0986],
         [0.1102],
         [0.0699],
         [0.0955],
         [0.1706],
         [0.0660],
         [0.1007],
         [0.0839],
         [0.0892],
         [0.1154]]])
    labels = t.Tensor([[1., 0., 1., 1., 1., 0., 0., 0., 0., 0.], [1., 0., 1., 1., 1., 0., 0., 0., 0., 0.]])
    # bi_loss = BiCutLoss()
    # ch_loss = ChoopyLoss()
    at_loss = AttnCutLoss()
    # div_loss = DivLoss()
    loss = at_loss(detection_output, truncation_output, labels)
    print(loss)
    # l1 = bi_loss(a_1, b)
    # l2 = ch_loss(a, b)
    # print(div_loss)
    # print(l1, l2, l3)
    # loss_f = RerankLoss()
    # y_pred = t.tensor([1., 1.2, 3.1, 0.1, 1.1, 4.2, 2.1, 5.1]).unsqueeze(dim=-1)
    # y_true = t.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    # loss = loss_f(a, b)
    # print(loss)

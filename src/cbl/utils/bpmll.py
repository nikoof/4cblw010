import torch
from torch import nn, Tensor

from typing import override

def bpmll(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    r"""Vanilla BP-MLL Loss for multilabel classification [1].

    .. math::
        \sum_{p=1}^{m} \frac{
            \sum_{(r,s) \in Y_p \times \overline{Y_p}} e^{-(c_r^p - c_s^p)}
        }{
            |Y_p| |\overline{Y_p}|
        }

    Note. Undefined for Y_p = \emptyset or \overline{Y_p} = \emptyset.


    References
    ----------
    .. [1] Min-Ling Zhang and Zhi-Hua Zhou, "Multilabel Neural Networks
           with Applications to Functional Genomics and Text Categorization,"
           in IEEE Transactions on Knowledge and Data Engineering, vol. 18,
           no. 10, pp. 1338-1351, Oct. 2006, doi: 10.1109/TKDE.2006.162.
    """
    Yp_mask, coYp_mask = targets.bool(), ~targets.bool()
    preds_r, preds_s = logits.unsqueeze(2), logits.unsqueeze(1)

    mat_mask = Yp_mask.unsqueeze(2) & coYp_mask.unsqueeze(1)

    numerator = torch.exp(-(preds_r - preds_s) * mat_mask).sum(dim=(1,2))
    denominator = Yp_mask.sum(dim=1) * coYp_mask.sum(dim=1)

    loss = numerator / (denominator + 1e-8)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Reduction must be one of 'mean', 'sum', 'none'")

def bpmll_with_global_threshold(logits: Tensor, threshold: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    r"""BP-MLL Loss with global threshold. This objective behaves like Vanilla BP-MLL,
    but it also optimizes a global threshold instead of having it fixed a priori [1].

    .. math::
        \sum_{p=1}^{m} \frac{
            \sum_{(r,s) \in Y_p \times \overline{Y_p}} e^{-(c_r^p - c_s^p)}
            + \sum_{r \in           Y_p } e^{-(c_r^p - c_Q^p)}
            + \sum_{s \in \overline{Y_p}} e^{-(c_Q^p - c_s^p)}
        }{
            |Y_p| |\overline{Y_p}| + |Y_p| + |\overline{Y_p}|
        }

    Note. Unlike Vanilla BP-MLL, this is always well-defined.

    References
    ----------
    .. [1] Grodzicki, R., Mańdziuk, J., Wang, L. (2008). Improved Multilabel Classification
           with Neural Networks. In: Rudolph, G., Jansen, T., Beume, N., Lucas, S., Poloni,
           C. (eds) Parallel Problem Solving from Nature – PPSN X. PPSN 2008. Lecture Notes
           in Computer Science, vol 5199. Springer, Berlin, Heidelberg.
    """
    cQ = threshold.unsqueeze(1)

    Yp_mask, coYp_mask = targets.bool(), ~targets.bool()
    preds_r, preds_s = logits.unsqueeze(2), logits.unsqueeze(1)

    mat_mask = Yp_mask.unsqueeze(2) & coYp_mask.unsqueeze(1)

    num1 = torch.exp(-(preds_r - preds_s) * mat_mask).sum(dim=(1,2))
    num2 = torch.exp(-(logits - cQ)[Yp_mask]).sum(dim=-1)
    num3 = torch.exp(-(cQ - logits)[coYp_mask]).sum(dim=-1)
    numerator = num1 + num2 + num3

    n_Yp = Yp_mask.sum(dim=1)
    n_coYp = coYp_mask.sum(dim=1)
    denominator = n_Yp * n_coYp + n_Yp + n_coYp

    loss = numerator / denominator

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Reduction must be one of 'mean', 'sum', 'none'")

class BPMLL(nn.Module):
    reduction: str

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return bpmll(predictions, targets, self.reduction)


class BPMLLWithGlobalThreshold(nn.Module):
    reduction: str

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    @override
    def forward(self, predictions: Tensor, threshold: Tensor, targets: Tensor) -> Tensor:
        return bpmll_with_global_threshold(predictions, threshold, targets, self.reduction)

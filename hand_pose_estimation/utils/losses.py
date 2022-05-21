import torch
from torch import nn


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = pred - gt
        l = (l * l) / 2 / 32  # TODO: 32 refers to batch size
        return l.sum()


class IoULoss(nn.Module):
    """
    Intersection over Union Loss.
    IoU = Area of Overlap / Area of Union
    IoU loss is modified to use for heatmaps.
    """

    def __init__(self):
        super(IoULoss, self).__init__()
        self.EPSILON = 1e-6

    def _op_sum(self, x):
        return x.sum(-1).sum(-1)

    def forward(self, y_pred, y_true):
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2) + self._op_sum(y_pred ** 2) - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.EPSILON) / (union + self.EPSILON)
        iou = torch.mean(iou)
        return 1 - iou


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        """

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)))
            * (self.alpha - y2)
            * (torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)
        )
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class JointLoss(torch.nn.Module):
    """
    loss for joints
    """

    def __init__(self):
        super(JointLoss, self).__init__()

    def forward(self, pred, gt):
        l = (pred - gt) ** 2
        l = l.sum(-1).sum(-1).mean()
        return l  ## l of dim bsize

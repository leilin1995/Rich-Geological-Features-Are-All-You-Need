"""
__author__: Lei Lin
__project__: losses.py
__time__: 2024/5/9 
__email__: leilin1117@outlook.com
"""
import torch
import torch.nn.functional as F
from torch import nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.size()[1] != 1:
            pred = pred[:, -1, :, :]  # 选择最后一个通道
        else:
            pred = pred.squeeze(1)
        # 因为 target 可能有一个通道维度，需要压缩这个维度
        if target.size()[1] == 1:
            target = target.squeeze(1)

            # 确保 pred 和 target 为 (N, H, W)
        pred = pred.contiguous()
        target = target.contiguous()

        # 计算交集，需要在三个空间维度上进行求和
        intersection = (pred * target).sum(dim=[1, 2, 3])

        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (
                pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) + self.smooth)

        # 计算 Dice 损失
        dice_loss = 1 - dice
        return dice_loss.mean()


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-6):
        super().__init__()
        self.bbce_loss = BBCELoss(epsilon=epsilon)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, label):
        bbce = self.bbce_loss(pred, label)
        dice = self.dice_loss(pred, label)

        # 混合损失：权重结合二者
        loss = self.alpha * bbce + self.beta * dice
        return loss


class BBCELoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, label):
        if pred.size()[1] != 1:
            pred = pred[:, -1, :]
        pred = pred.view(pred.shape[0], -1)
        label = label.view(label.shape[0], -1)

        # 使用clamp函数和epsilon避免数值稳定性问题
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        log_pred = torch.log(pred)
        log_pred_neg = torch.log(1 - pred)

        # 计算权重beta
        beta = 1 - torch.sum(label, dim=1, keepdim=True) / label.shape[1]
        beta = torch.clamp(beta, self.epsilon, 1 - self.epsilon)  # 确保beta在合理范围内
        # 计算平衡的交叉熵损失
        loss = -beta * label * log_pred - (1 - beta) * (1 - label) * log_pred_neg
        return loss.mean()


# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2, size_average=True):
        """
        Args:
            alpha:Control positive and negative unbalanced samples weights
            gamma:Control the learning of samples with low execution
            size_average:average or not
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, labels):
        if preds.size[1] != 1:
            preds = F.softmax(preds, dim=1)
            preds = preds[:, 1, ]
        preds = preds.view(preds.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)
        # focal loss
        loss = - (self.alpha * ((1 - preds) ** self.gamma) * labels * torch.log(preds + 10e-8) + (1 - self.alpha) * (
                preds ** self.gamma) * (1 - labels) * torch.log(1 - preds + 10e-8))
        if self.size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss

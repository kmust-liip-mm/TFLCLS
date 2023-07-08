# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class AutomaticWeightedOneLoss(nn.Module):
    def __init__(self, ratio=1.5):
        super(AutomaticWeightedOneLoss, self).__init__()
        params = torch.ones(1, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.radio = ratio

    def forward(self, *x):
        loss_sum = (0.5 / (self.params[0] ** 2) * x[0] + torch.log(1 + self.params[0] ** 2)) + \
                   (0.5 / ((self.params[0]*self.radio) ** 2) * x[1] + torch.log(1 + (self.params[0]*self.radio) ** 2))
        return loss_sum

class AutomaticWeightedTwoLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedTwoLoss, self).__init__()
        params = torch.tensor([0.5, 0.5], requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        loss_sum += ((self.params[0]/self.params[1]) - 2) ** 2
        return loss_sum
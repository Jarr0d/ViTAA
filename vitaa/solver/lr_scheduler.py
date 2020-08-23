# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch
from math import pi, cos


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class LRSchedulerWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        mode="step",
        warmup_factor=1.0 / 3,
        warmup_epochs=10,
        warmup_method="linear",
        last_epoch=-1,
        target_lr=0,
        power=0.9
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if mode not in ("step", "exp", "poly", "cosine"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode=mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.target_lr = target_lr
        self.power = power
        super(LRSchedulerWithWarmup, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]

        if self.mode == "step":
            return [
                base_lr
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        elif self.mode == "exp":
            factor = (self.last_epoch - self.warmup_epochs) / (self.milestones[0] - self.warmup_epochs)
            return [
                base_lr * self.power ** factor
                for base_lr in self.base_lrs
            ]
        elif self.mode == "poly":
            factor = 1 - (self.last_epoch - self.warmup_epochs) / (self.milestones[0] - self.warmup_epochs)
            return [
                self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                for base_lr in self.base_lrs
            ]
        elif self.mode == "cosine":
            factor = (1 + cos(pi * (self.last_epoch - self.warmup_epochs) / (self.milestones[0] - self.warmup_epochs))) / 2
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
        else:
            raise NotImplementedError

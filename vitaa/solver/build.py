# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import LRSchedulerWithWarmup


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    optimizer = torch.optim.Adam(params, lr, betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA), eps=1e-8)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        mode=cfg.SOLVER.LRSCHEDULER,
        target_lr=cfg.SOLVER.TARGET_LR,
        power=cfg.SOLVER.POWER
    )

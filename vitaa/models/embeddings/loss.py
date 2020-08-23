import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class LossComputation(nn.Module):
    def __init__(self, num_classes, feature_size, dropout_prob):
        super(LossComputation, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dropout_prob = dropout_prob
        self.scale = 28
        self.margin = 0.3

        self.W = Parameter(torch.randn(self.feature_size, self.num_classes), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def instance_loss(self, visual_embed, textual_embed, labels):
        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=0)

        visual_logits = self.scale * torch.matmul(visual_norm, W_norm)
        textual_logits = self.scale * torch.matmul(textual_norm, W_norm)

        criterion = nn.CrossEntropyLoss(reduction='mean')
        v_loss = criterion(input=visual_logits, target=labels)
        t_loss = criterion(input=textual_logits, target=labels)
        loss = v_loss + t_loss

        return loss

    def global_align_loss(self, visual_embed, textual_embed, labels):
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40

        batch_size = visual_embed.size(0)
        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)
        similarity = torch.matmul(visual_norm, textual_norm.t())
        labels_ = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t())

        loss = 0
        for i in range(batch_size):
            pred = similarity[i]
            label = labels_[i].float()
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
            loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
            loss += loss_pos.sum() + loss_neg.sum()

            pred = similarity[:, i]
            label = labels_[:, i].float()
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
            loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
            loss += loss_pos.sum() + loss_neg.sum()

        loss /= batch_size
        return loss

    def forward(self, visual_embed, textual_embed, labels):
        global_align_loss = self.global_align_loss(visual_embed, textual_embed, labels)
        instance_loss = self.instance_loss(visual_embed, textual_embed, labels)

        losses = {
            "instance_loss": instance_loss,
            "global_align_loss": global_align_loss,
        }
        return losses


def make_loss_evaluator(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES
    feature_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
    dropout_prob = cfg.MODEL.EMBEDDING.DROPOUT_PROB
    return LossComputation(num_classes, feature_size, dropout_prob)

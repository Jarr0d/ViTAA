import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super(LossComputation, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.feature_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.dropout_prob = cfg.MODEL.EMBEDDING.DROPOUT_PROB
        self.num_parts = cfg.MODEL.NUM_PARTS

        self.scale = 28
        self.margin = 0.2

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

    def mask_loss(self, seg_feat, masks):
        mask_logits = seg_feat
        masks = torch.stack(masks, dim=1)
        masks = masks.view(-1, masks.size(-2), masks.size(-1))

        mask_loss = F.cross_entropy(
            mask_logits, masks.long(), reduction='none'
        )
        mask_loss = self.num_parts * mask_loss.mean()
        return mask_loss

    def global_align_loss(self, visual_embed, textual_embed, labels):
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40

        batch_size = labels.size(0)
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

    def local_align_loss(self, part_embed, attribute_embed, labels, part_masks, attr_masks):
        alpha = 0.6
        beta = 0.4
        scale_pos = 10
        scale_neg = 40
        topK = 8

        batch_size = labels.size(0)
        part_embed = F.normalize(part_embed, p=2, dim=2)
        attribute_embed = F.normalize(attribute_embed, p=2, dim=2)
        labels_ = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t())

        losses = 0
        for i in range(self.num_parts):
            part_mask = part_masks[:, i]
            attr_mask = attr_masks[:, i]
            similarity = torch.matmul(part_embed[i], attribute_embed[i].t())
            rank1 = torch.argsort(similarity, dim=1, descending=True)
            rank2 = torch.argsort(similarity.t(), dim=1, descending=True)

            loss = 0
            for j in range(batch_size):
                if part_mask[j] == 0:
                    continue
                pred = similarity[j, attr_mask]
                # k-reciprocal sample
                label = labels_[j, :].float()
                forward_k_idx = rank1[i, :topK]
                backward_k_idx = rank2[forward_k_idx, :topK]
                sample_pos_idx = torch.nonzero(backward_k_idx == i)[:, 0]
                sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
                label[sample_pos_idx] = 1
                label = label[attr_mask]
                pos_inds = torch.nonzero(label == 1).squeeze(1)
                neg_inds = torch.nonzero(label == 0).squeeze(1)
                if pos_inds.numel() > 0:
                    loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
                    loss += loss_pos.sum()
                if neg_inds.numel() > 0:
                    loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                    loss += loss_neg.sum()

                if attr_mask[j] == 0:
                    continue
                pred = similarity[part_mask, j]
                # k-reciprocal sample
                label = labels_[j, :].float()
                forward_k_idx = rank2[i, :topK]
                backward_k_idx = rank1[forward_k_idx, :topK]
                sample_pos_idx = torch.nonzero(backward_k_idx == i)[:, 0]
                sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
                label[sample_pos_idx] = 1
                label = label[part_mask]
                pos_inds = torch.nonzero(label == 1).squeeze(1)
                neg_inds = torch.nonzero(label == 0).squeeze(1)
                if pos_inds.numel() > 0:
                    loss_pos = torch.log(1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha)))
                    loss += loss_pos.sum()
                if neg_inds.numel() > 0:
                    loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                    loss += loss_neg.sum()

            loss /= batch_size
            losses += loss
        losses /= self.num_parts
        return losses

    def forward(self, visual_embed, textual_embed,
                part_embed, attribute_embed,
                seg_feat, captions):
        labels = torch.stack([caption.get_field('id') for caption in captions]).long()
        masks = [caption.get_field('crops') for caption in captions]
        vmask = torch.stack([caption.get_field('mask') for caption in captions])
        attributes = [caption.get_field('attribute') for caption in captions]
        tmask = torch.stack([attribute.get_field('mask') for attribute in attributes])

        global_align_loss = self.global_align_loss(visual_embed, textual_embed, labels)
        local_align_loss = self.local_align_loss(part_embed, attribute_embed, labels, vmask, tmask)
        instance_loss = self.instance_loss(visual_embed, textual_embed, labels)
        mask_loss = self.mask_loss(seg_feat, masks)

        losses = {
            "instance_loss": instance_loss,
            "mask_loss": mask_loss,
            "global_align_loss": global_align_loss,
            "local_align_loss": local_align_loss
        }
        return losses


def make_loss_evaluator(cfg):
    return LossComputation(cfg)

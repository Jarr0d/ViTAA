import torch
import torch.nn as nn
import torch.nn.functional as F

from .seg_head.seg_head import build_seg_head
from .loss import make_loss_evaluator


class SimpleHead(nn.Module):
    def __init__(self,
                 cfg,
                 visual_size,
                 textual_size,
                 ):
        super(SimpleHead, self).__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_embed_layer = nn.Sequential(
            nn.Linear(visual_size, self.embed_size),
            # nn.BatchNorm1d(self.embed_size),
            # nn.ReLU(True)
        )
        self.textual_embed_layer = nn.Sequential(
            nn.Linear(textual_size, self.embed_size),
            # nn.BatchNorm1d(self.embed_size),
            # nn.ReLU(True)
        )

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                visual_feature,
                textual_feature,
                captions):
        batch_size = visual_feature.size(0)
        visual_feature = self.avgpool(visual_feature)

        visual_embed = visual_feature.view(batch_size, -1)
        textual_embed = textual_feature.view(batch_size, -1)
        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)

        if self.training:
            labels = torch.stack([caption.get_field('id') for caption in captions]).long()

            losses = self.loss_evaluator(
                visual_embed, textual_embed, labels
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None


def build_embed(cfg,
                visual_out_channels,
                textual_out_channels):
    if cfg.MODEL.EMBEDDING.EMBED_HEAD == 'seg':
        return build_seg_head(cfg,
                              visual_out_channels,
                              textual_out_channels)

    return SimpleHead(cfg,
                      visual_out_channels,
                      textual_out_channels)
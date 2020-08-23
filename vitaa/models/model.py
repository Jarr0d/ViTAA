import torch
from torch import nn

from .backbones.resnet import build_resnet
from .backbones.lstm import build_lstm
from .embeddings.embed import build_embed


class ViTAA(nn.Module):

    def __init__(self, cfg):
        super(ViTAA, self).__init__()
        self.visual_model = build_resnet(cfg)
        self.textual_model = build_lstm(cfg, bidirectional=True)
        self.embed_model = build_embed(
            cfg,
            self.visual_model.out_channels,
            self.textual_model.out_channels
        )

    def forward(self, images, captions, only_img=False):
        visual_feat = self.visual_model(images)
        textual_feat = self.textual_model(captions)
        attributes = [caption.get_field('attribute') for caption in captions]
        attribute_feat = self.textual_model(attributes)

        outputs_embed, losses_embed = self.embed_model(
            visual_feat, textual_feat, attribute_feat, captions)

        if self.training:
            losses = {}
            losses.update(losses_embed)
            return losses

        return outputs_embed


def build_model(cfg):
    return ViTAA(cfg)
"""Frozen encoders used as fixed feature extractors.

Keeping the encoders frozen is what makes the whole study tractable without a
GPU: every image is encoded exactly once, and the hundreds of cross-validated
head fits that the evaluation protocol demands then run on cached vectors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _TimmWrap(nn.Module):
    def __init__(self, name: str, **kw):
        super().__init__()
        import timm

        self.net = timm.create_model(name, pretrained=True, num_classes=0, **kw)
        cfg = self.net.default_cfg
        self.mean, self.std = cfg["mean"], cfg["std"]
        self.dim = self.net.num_features

    def forward(self, x):
        return self.net(x)


class _HFVit(nn.Module):
    """HuggingFace ViT encoder; uses the CLS token, the representation these
    pathology encoders were benchmarked with."""

    def __init__(self, repo: str):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(repo)
        self.net = AutoModel.from_pretrained(repo)
        self.mean = tuple(proc.image_mean)
        self.std = tuple(proc.image_std)
        self.dim = self.net.config.hidden_size

    def forward(self, x):
        return self.net(pixel_values=x).last_hidden_state[:, 0]


# name -> (constructor, short description for the paper's table)
REGISTRY: dict[str, tuple] = {
    # natural-image supervised pretraining: the original paper's backbone
    "resnet50_in1k": (lambda: _TimmWrap("resnet50.tv_in1k"), "ResNet-50, ImageNet-1k supervised"),
    # natural-image self-supervised
    "dinov2_vitb14": (lambda: _TimmWrap("vit_base_patch14_dinov2.lvd142m", img_size=224),
                      "ViT-B/14, DINOv2 self-supervised (LVD-142M)"),
    # pathology foundation model: iBOT ViT-B on 40M pan-cancer TCGA tiles
    "phikon_vitb16": (lambda: _HFVit("owkin/phikon"), "ViT-B/16, Phikon iBOT (TCGA pan-cancer)"),
    # efficient CNN, for the compute/accuracy trade-off panel
    "efficientnet_b0": (lambda: _TimmWrap("efficientnet_b0.ra_in1k"), "EfficientNet-B0, ImageNet-1k"),
}


def load(name: str) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; have {sorted(REGISTRY)}")
    model = REGISTRY[name][0]()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

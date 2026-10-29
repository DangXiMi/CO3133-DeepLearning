"""Pretrained/modern model for Assignment 2.

ImageNet ResNet-50 fine-tuned on Food-101 at 224px with 101 classes.  This is
the pretrained model required by the A2 technical requirements; the simple
baseline is the Assignment 1 CNN in `baseline.py`.
"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(num_classes=101, pretrained=True, freeze_backbone=False):
    """ImageNet ResNet-50 with a fresh 101-class head.

    ``freeze_backbone=True`` keeps the ImageNet weights fixed (linear probe)
    and only trains the head; BatchNorm statistics are also frozen by
    ``training_loop.set_bn_eval`` during training.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    return model

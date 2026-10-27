"""Pretrained/modern model for Assignment 2.

ImageNet ResNet-18 fine-tuned on Food-101 at 224px with 101 classes.  This is
the pretrained model required by the A2 technical requirements; the simple
baseline is the Assignment 1 CNN in `baseline.py`.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes=101, pretrained=True, freeze_backbone=False):
    """ImageNet ResNet-18 with a fresh 101-class head.

    ``freeze_backbone=True`` keeps the ImageNet weights fixed (linear probe)
    and only trains the head; BatchNorm statistics are also frozen by
    ``training_loop.set_bn_eval`` during training.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    return model

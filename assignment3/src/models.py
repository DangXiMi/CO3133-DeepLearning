"""Models for Assignment 3: unimodal encoders and the simple fusion head."""

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)
from transformers import AutoModel, AutoModelForSequenceClassification

TEXT_MODEL_NAME = "bert-base-uncased"


def build_image_model(num_labels=23, backbone="resnet50", freeze_backbone=False):
    if backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported image backbone: {backbone}")

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_labels),
    )

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False
    return model


def build_text_model(num_labels=23, model_name=TEXT_MODEL_NAME):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )


def build_image_encoder(backbone="resnet50"):
    if backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported image backbone: {backbone}")
    model.fc = nn.Identity()
    return model


def build_text_encoder(model_name=TEXT_MODEL_NAME):
    return AutoModel.from_pretrained(model_name)


def build_fusion_head(input_dim=2816, hidden_dim=512, num_labels=23):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, num_labels),
    )


class GatedMultimodalUnit(nn.Module):
    """GMU from Arevalo et al., with optional modality dropout."""

    def __init__(self, image_dim=2048, text_dim=768, hidden_dim=512,
                 num_labels=23, modality_dropout=0.0):
        super().__init__()
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.gate_fc = nn.Linear(image_dim + text_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.modality_dropout = modality_dropout

    def forward(self, image_features, text_features, drop_image=None, drop_text=None):
        batch_size = image_features.size(0)
        device = image_features.device

        if self.training and self.modality_dropout > 0:
            if drop_image is None:
                drop_image = (torch.rand(batch_size, device=device) < self.modality_dropout).float()
            if drop_text is None:
                drop_text = (torch.rand(batch_size, device=device) < self.modality_dropout).float()
        if drop_image is None:
            drop_image = torch.zeros(batch_size, device=device)
        if drop_text is None:
            drop_text = torch.zeros(batch_size, device=device)

        image_features = image_features * (1.0 - drop_image)[:, None]
        text_features = text_features * (1.0 - drop_text)[:, None]

        image_hidden = torch.tanh(self.image_fc(image_features))
        text_hidden = torch.tanh(self.text_fc(text_features))
        gate = torch.sigmoid(self.gate_fc(torch.cat([image_features, text_features], dim=1)))
        fused = image_hidden * gate + text_hidden * (1.0 - gate)
        return self.classifier(fused)


def build_gmu(num_labels=23, hidden_dim=512, modality_dropout=0.0):
    return GatedMultimodalUnit(
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        modality_dropout=modality_dropout,
    )


class MultimodalGMU(nn.Module):
    """End-to-end GMU with ResNet-50 and BERT encoders."""

    def __init__(self, num_labels=23, hidden_dim=512, modality_dropout=0.0):
        super().__init__()
        self.image_encoder = build_image_encoder()
        self.text_encoder = build_text_encoder()
        self.gmu = build_gmu(
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            modality_dropout=modality_dropout,
        )

    def forward(self, images, input_ids, attention_mask, drop_image=None, drop_text=None):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        return self.gmu(
            image_features,
            text_features,
            drop_image=drop_image,
            drop_text=drop_text,
        )


def build_multimodal_gmu(num_labels=23, hidden_dim=512, modality_dropout=0.0):
    return MultimodalGMU(
        num_labels=num_labels,
        hidden_dim=hidden_dim,
        modality_dropout=modality_dropout,
    )

# CNN models: convolutional feature extractor and classifier
import numpy as np
import torch
import torch.nn as nn

from .mlp import MLP


class CNN(nn.Module):
    """Convolutional feature extractor: stacked [Conv -> ReLU -> MaxPool] blocks.
        Each convolution learns local pattern detectors (edges, textures) with
        weights shared across all image positions; the activation keeps only
        active detections; max pooling takes the maximum over small patches,
        which halves the spatial size, cuts computation, and adds small
        translation tolerance. The produced channels are feature maps: one
        response plane per learned detector.
        Spatial sizes follow
            out = ((in + 2*padding - kernel_size) // stride) + 1
        e.g. an input of spatial size (28, 32) with kernel_size=4, padding=3,
        stride=2 results in an output of size (16, 18). With the defaults
        (kernel_size=3, padding=1, stride=1) each conv preserves the spatial
        size and each pool halves it: 28 -> 14 -> 7.
        Input:
            - input_shape (tuple): image shape (C, H, W), no batch dimension
            - channels (tuple[int]): output channel count of each conv block
            - kernel_size (int), padding (int), stride (int): conv hyperparameters
            - activation: activation class, instantiated once per block
        Output:
            - feature maps of shape (B, channels[-1], H/2^len(channels), W/2^len(channels))
    """

    def __init__(self,
                 input_shape=(1, 28, 28),
                 channels=(32, 64),
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 activation=nn.ReLU):
        super().__init__()

        in_channels = input_shape[0]
        blocks = []
        for out_channels in channels:
            # conv: out = ((in + 2*padding - kernel_size) // stride) + 1, presets kept
            # pool : out = in // 2
            blocks += [
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding),
                activation(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

    def forward(self, x):
        return self.features(x)

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNClassifier(nn.Module):
    """CNN classifier: CNN feature extractor + MLP classification head.
        The conv stack turns the image into feature maps; the MLP head
        flattens them (its own nn.Flatten) and maps them to class logits.
        The head input size is measured with one dummy forward pass on a batch
        of one image, so any conv configuration works without editing this
        class: the conv/pool layout decides the feature size, not this code.
        Input:
            - input_shape (tuple): image shape (C, H, W), no batch dimension
            - channels (tuple[int]): output channel count of each conv block
            - kernel_size (int), padding (int), stride (int): conv hyperparameters
            - activation: activation class, forwarded to CNN and MLP
            - num_classes (int): number of output logits
            - hidden_dims (tuple[int]): hidden widths of the MLP head
        Output:
            - logits of shape (B, num_classes), no softmax
    """

    def __init__(self,
                 input_shape=(1, 28, 28),
                 channels=(32, 64),
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 activation=nn.ReLU,
                 num_classes=10,
                 hidden_dims=(128, 256)):
        super().__init__()

        self.cnn = CNN(input_shape=input_shape, channels=channels,
                       kernel_size=kernel_size, padding=padding,
                       stride=stride, activation=activation)

        with torch.no_grad():
            features = self.cnn(torch.zeros(1, *input_shape))  # batch size of 1
            cnn_output_shape = features.shape[1:]              # drop the batch dim
        input_dim = int(np.prod(cnn_output_shape))

        self.mlp = MLP(input_dim=input_dim,
                       hidden_dims=hidden_dims,
                       output_dim=num_classes,
                       activation=activation)

    def forward(self, x):
        return self.mlp(self.cnn(x))  # the MLP head flattens the feature maps itself

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

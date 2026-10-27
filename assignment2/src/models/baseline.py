"""Simple baseline for Assignment 2.

This is the Assignment 1 CNN, retrained from scratch on Food-101 at 128px with
101 classes.  It is the self-implemented baseline that the pretrained ResNet-18
is compared against.
"""

from assignment1.src.models.cnn import CNNClassifier


def build_baseline_cnn(num_classes=101, image_size=128,
                       channels=(32, 64, 128, 256),
                       hidden_dims=(512,)):
    return CNNClassifier(
        input_shape=(3, image_size, image_size),
        channels=channels,
        kernel_size=3,
        padding=1,
        stride=1,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
    )

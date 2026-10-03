# Linear (softmax) classifier model
import numpy as np
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear (softmax) classifier: a single linear layer on the flattened image.
        Flattening turns each image into one feature vector; the linear layer
        produces one score (logit) per class as the dot product of the features
        with a learned class template. There is no hidden layer and no
        nonlinearity, so the decision boundaries are hyperplanes in pixel
        space; this is the baseline the stronger models are compared against.
        The model outputs raw logits on purpose: CrossEntropyLoss applies
        log-softmax internally, so adding a softmax here would apply it twice.
        Input:
            - input_shape (tuple): image shape (C, H, W), no batch dimension
            - num_classes (int): number of output logits
        Output:
            - logits of shape (B, num_classes), no softmax
    """

    def __init__(self,
                 input_shape=(1, 28, 28),
                 num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(np.prod(input_shape)), num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

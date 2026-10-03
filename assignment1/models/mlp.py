# Multilayer perceptron model
import torch.nn as nn


class MLP(nn.Module):
    """Multilayer perceptron: stacked Linear -> activation blocks plus a classifier head.
        Each layer performs a learned linear projection; the activation (ReLU by
        default) adds the nonlinearity that lets stacked layers model non-linear
        functions. A dropout layer can be inserted after each activation as
        regularization; with the default dropout=0.0 no regularization is used.
        The input is flattened inside, so this class works both as a standalone
        image classifier (input (B, C, H, W)) and as a classification head on
        pre-flattened features (input (B, input_dim)).
        Input:
            - input_dim (int): number of input features after flattening
            - hidden_dims (tuple[int]): width of each hidden layer
            - output_dim (int): number of classes
            - activation: activation class, instantiated once per hidden layer
            - dropout (float): dropout probability after each activation, 0.0 disables it
        Output:
            - logits of shape (B, output_dim), no softmax (CrossEntropyLoss expects raw logits)
    """

    def __init__(self,
                 input_dim=784,
                 hidden_dims=(256, 128),
                 output_dim=10,
                 activation=nn.ReLU,
                 dropout=0.0):
        super().__init__()

        self.flatten = nn.Flatten()

        layers, in_dim = [], input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(self.flatten(x))

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

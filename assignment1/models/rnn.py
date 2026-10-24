# Recurrent models: plain RNN, GRU and LSTM through one shared scaffold
from functools import partial

import torch.nn as nn


class RNNClassifier(nn.Module):
    """Recurrent image classifier: reads the image as a sequence and scores the final hidden state.
        Sequence definition:
            Each image is read one row at a time, top to bottom. For an image
            (C, H, W), timestep t is row t, so the sequence length is T = H and
            each step's input is that row's C*W pixel values. After
            permute(0, 2, 1, 3).flatten(2), the batch has shape (B, H, C*W).
            For Fashion-MNIST this is 28 timesteps of 28 features each.
        Input size:
            input_size = C*W, the number of values in one image row.
        Hidden representation:
            The recurrent cell updates h_t = f(x_t, h_{t-1}), where h_t has
            hidden_size units. The final hidden state of the last layer, h_T,
            summarizes the whole image and is mapped to class logits by a linear
            head. LSTM also carries a cell state c_t, but only h_T is used for
            classification.
        Why GRU/LSTM:
            A plain RNN applies the same weights at every step, so gradients can
            vanish or explode over 28 steps. GRU/LSTM use gates to control what
            is written, forgotten, and read. GRU has 3 gates and fewer
            parameters; LSTM has 4 gates and a separate cell state.
        Input:
            - input_shape (tuple): image shape (C, H, W), no batch dimension
            - hidden_size (int): width of the recurrent hidden state
            - num_layers (int): number of stacked recurrent layers
            - num_classes (int): number of output logits
            - cell: the recurrent layer class to use: nn.RNN, nn.GRU or nn.LSTM
        Output:
            - logits of shape (B, num_classes), no softmax
    """

    def __init__(self,
                 input_shape=(1, 28, 28),
                 hidden_size=128,
                 num_layers=1,
                 num_classes=10,
                 cell=nn.GRU):
        super().__init__()

        # All three cells share the exact same constructor:
        #   cell(input_size=..., hidden_size=..., num_layers=..., batch_first=True)
        # input_size is the per-step feature vector: C*W (one row, all channels).
        # batch_first=True is required: input layout (B, seq, features);
        # without it the layout is (seq, B, features) and your shapes silently misread.

        self.rnn = cell(input_size=input_shape[0] * input_shape[2],
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
        
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> sequence of image rows: (B, H, C*W) = (batch, seq, features)
        seq = x.permute(0, 2, 1, 3).flatten(2)

        # output: (B, seq, hidden), the last layer's hidden state at every step.
        # output[:, -1] is the final step, i.e. the memory after the whole image
        # has been read. This equals h_n[-1] for a uni-directional RNN/GRU/LSTM,
        # and also works for LSTM without unpacking its (h_n, c_n) state tuple.
        output, _ = self.rnn(seq)
        return self.head(output[:, -1])

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


GRUClassifier = partial(RNNClassifier, cell=nn.GRU)
LSTMClassifier = partial(RNNClassifier, cell=nn.LSTM)
PlainRNNClassifier = partial(RNNClassifier, cell=nn.RNN)

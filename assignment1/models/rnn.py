# Recurrent models: plain RNN, GRU and LSTM through one shared scaffold
from functools import partial

import torch.nn as nn


class RNNClassifier(nn.Module):
    """Recurrent image classifier: reads the image as a sequence and scores the final hidden state.
        [TODO (you): sequence definition — handbook section 11.1.4 asks you to explain
        the timestep definition, the input size, and the hidden representation.
        Fill this in after reading d2l's GRU/LSTM sections.]
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

        # The call returns two things:
        #   output: (B, seq, hidden)  per-step hidden states, unused here
        #   state:  h_n for RNN/GRU; (h_n, c_n) for LSTM, hence the tuple check
        output, state = self.rnn(seq)
        h_n = state[0] if isinstance(state, tuple) else state

        # h_n: (num_layers, B, hidden); [-1] takes the LAST LAYER's final state,
        # i.e. the memory after the whole image has been read
        return self.head(h_n[-1])

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


GRUClassifier = partial(RNNClassifier, cell=nn.GRU)
LSTMClassifier = partial(RNNClassifier, cell=nn.LSTM)
PlainRNNClassifier = partial(RNNClassifier, cell=nn.RNN)

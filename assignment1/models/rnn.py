# Recurrent models: plain RNN, GRU and LSTM through one shared scaffold
from functools import partial

import torch
import torch.nn as nn

# Map the family name to its single-step cell class.
_CELLS = {
    "RNN": nn.RNNCell,
    "GRU": nn.GRUCell,
    "LSTM": nn.LSTMCell,
}


class RNNClassifier(nn.Module):
    """Recurrent image classifier over a sequence of image rows.

        Sequence and timesteps:
            A timestep is one iteration of the recurrence: the cell reads one
            input vector and updates its hidden state. Here the sequence is
            defined along the image rows, top to bottom. Timestep t consumes
            row t, so the sequence length is T = H and each step's input vector
            has C*W values (28 for Fashion-MNIST). T, C*W, and hidden_size are
            different quantities.

        Recurrence and hidden representation:
            h_0 = 0
            h_t = cell(x_t, h_{t-1})
            h_t has hidden_size values and summarizes the rows read so far. The
            final hidden state h_T is mapped to class logits by a linear head.
            LSTM also carries a cell state c_t and returns (h_t, c_t); only h_T
            is classified.

        Implementation:
            Uses single-step RNNCell/GRUCell/LSTMCell in an explicit loop over T.
            num_layers stacks cells; only the last layer's h_T is classified.

        Args:
            input_shape (tuple): image shape (C, H, W), no batch dimension.
            hidden_size (int): width of the hidden state h_t.
            num_layers (int): number of stacked recurrent layers.
            num_classes (int): number of output logits.
            cell (str): "RNN", "GRU" or "LSTM"; mapped internally to the
                corresponding single-step cell class.

        Output:
            logits of shape (B, num_classes), no softmax.
    """

    def __init__(self,
                 input_shape=(1, 28, 28),
                 hidden_size=128,
                 num_layers=1,
                 num_classes=10,
                 cell="GRU"):
        super().__init__()

        if not isinstance(cell, str) or cell.upper() not in _CELLS:
            raise ValueError(
                f"Unsupported cell {cell!r}; use 'RNN', 'GRU' or 'LSTM'."
            )
        cell = cell.upper()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}.")

        self.input_size = input_shape[0] * input_shape[2]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_name = cell
        self.is_lstm = cell == "LSTM"

        # Layer 0 consumes rows of size C*W; each later layer consumes the
        # previous layer's hidden vectors of size hidden_size.
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = self.input_size if layer == 0 else hidden_size
            self.cells.append(
                _CELLS[cell](
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                )
            )

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # (B, C, H, W) -> (B, T=H, C*W): one image row per timestep
        seq = x.permute(0, 2, 1, 3).flatten(2)
        batch_size, seq_len, _ = seq.shape

        layer_input = seq
        for cell in self.cells:
            # Initial state: h_0 = 0; LSTM also has c_0 = 0.
            h = layer_input.new_zeros(batch_size, self.hidden_size)
            c = layer_input.new_zeros(batch_size, self.hidden_size) if self.is_lstm else None

            outputs = []
            for t in range(seq_len):
                x_t = layer_input[:, t, :]  # row t, shape (B, C*W)

                # LSTM updates both states; RNN/GRU update only h.
                if self.is_lstm:
                    h, c = cell(x_t, (h, c))
                else:
                    h = cell(x_t, h)

                outputs.append(h)

            # Stack per-timestep states: (B, T, hidden_size); feeds the next layer.
            layer_input = torch.stack(outputs, dim=1)

        # Classify the last layer's final hidden state h_T.
        return self.head(h)

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


GRUClassifier = partial(RNNClassifier, cell="GRU")
LSTMClassifier = partial(RNNClassifier, cell="LSTM")
PlainRNNClassifier = partial(RNNClassifier, cell="RNN")

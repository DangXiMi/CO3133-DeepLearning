# Transformer model: row-token image classifier with self-attention
import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(seq_len, d_model):
    """Build the fixed sinusoidal positional encoding from Vaswani et al. (2017).
        Each position p gets a d_model-dimensional vector whose even dimensions
        are sin(p / 10000^(2i/d_model)) and whose odd dimensions are the matching
        cosine. The pattern is deterministic, so it adds no trainable parameters
        and can be sliced to any sequence length up to seq_len.
        Input:
            - seq_len (int): number of positions (rows) to encode
            - d_model (int): embedding width
        Output:
            - pe of shape (1, seq_len, d_model); the leading 1 broadcasts over B
    """
    if d_model % 2 != 0:
        raise ValueError(
            f"d_model={d_model} must be even for sinusoidal positional encoding."
        )

    # For d_model = 6, the rows look like:
    # token 1: [ sin(0),      cos(0),      sin(0),      cos(0),      sin(0),      cos(0)      ]
    # token 2: [ sin(f0),     cos(f0),     sin(f1),     cos(f1),     sin(f2),     cos(f2)     ]
    # token 3: [ sin(2*f0),   cos(2*f0),   sin(2*f1),   cos(2*f1),   sin(2*f2),   cos(2*f2)   ]
    # ...
    # where fi = 1 / 10000^(2i/d_model)

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

    # 1 / 10000^(2i/d_model) = exp((2i/d_model) * -log(10000))
    #                        = exp(2i * -log(10000) / d_model)
    freq = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    # slicing is start:stop:step so 0::2 means start from 0, to the end of
    # array, with step 2. similar for 1::2
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)

    return pe.unsqueeze(0)  # (1, seq_len, d_model)
                            # unsqueeze(0) helps broadcast over (B,)


class TransformerClassifier(nn.Module):
    """Transformer image classifier over a sequence of image rows.

        Each image row is one token: T = H tokens of C*W values. A linear layer
        projects each token to d_model, fixed sinusoidal positions are added so the
        model can distinguish row order, and a Transformer encoder applies
        multi-head self-attention. For each head, attention computes Q, K, V and
        softmax(QK^T / sqrt(d_head)) V, so every token can attend to every other
        token; there is no causal mask. Mean pooling over the T token outputs gives
        one (B, d_model) vector per image, which the linear head maps to class
        logits.

        Args:
            input_shape (tuple): image shape (C, H, W), no batch dimension.
            d_model (int): token embedding width; must be divisible by nhead.
            nhead (int): number of attention heads.
            num_layers (int): number of stacked encoder layers.
            dim_feedforward (int): hidden width of each feed-forward network.
            dropout (float): dropout probability inside the encoder.
            num_classes (int): number of output logits.

        Output:
            logits of shape (B, num_classes), no softmax.
    """

    def __init__(self,
                 input_shape=(1, 28, 28),
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 num_classes=10):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}."
            )

        C, H, W = input_shape
        self.token_dim = C * W
        self.seq_len = H

        # Each row token has C*W values; project it to the attention width.
        self.proj = nn.Linear(self.token_dim, d_model)

        # Fixed sinusoidal positions, moved to the device by .to(device).
        # It is a buffer, not a parameter: it is saved in the state_dict but
        # does not receive gradients.
        self.register_buffer(
            "pos",
            sinusoidal_positional_encoding(self.seq_len, d_model),
        )

        # batch_first=True is required: tokens are (B, T, d_model).
        # The default for transformer layers is False, i.e. (T, B, d_model).
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> sequence of image rows: (B, H, C*W)
        tokens = x.permute(0, 2, 1, 3).flatten(2)
        seq_len = tokens.size(1)

        if seq_len > self.pos.size(1):
            raise ValueError(
                f"Input has {seq_len} tokens but the positional encoding was "
                f"built for at most {self.pos.size(1)}."
            )

        # Raw row token (C*W) -> attention embedding (d_model).
        tokens = self.proj(tokens)

        # Add positions; (1, T, d_model) broadcasts over the batch dimension.
        tokens = tokens + self.pos[:, :seq_len]

        # Self-attention over all rows, for every image independently.
        tokens = self.encoder(tokens)  # (B, T, d_model)

        # Aggregate the token outputs into one vector per image.
        pooled = tokens.mean(dim=1)  # (B, d_model)
        return self.head(pooled)     # (B, num_classes)

    def num_para(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

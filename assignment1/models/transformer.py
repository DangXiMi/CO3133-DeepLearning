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

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class TransformerClassifier(nn.Module):
    """Transformer image classifier: reads the image as a sequence of rows and attends over them.
        Tokenization:
            Each image row is one token, exactly as in RNNClassifier. For an
            image (C, H, W), timestep t is row t, so the sequence length is
            T = H and each token has C*W values. After
            permute(0, 2, 1, 3).flatten(2), the batch is (B, H, C*W).
            For Fashion-MNIST this is 28 tokens of 28 features each.
        Token projection:
            A linear layer maps each raw row token from C*W to d_model, the
            width attention operates on.
        Positional encoding:
            Self-attention is permutation-invariant: without position
            information, shuffling the rows would only shuffle the outputs.
            A fixed sinusoidal encoding of shape (1, T, d_model) is added to
            the projected tokens, so the model knows which row each token came
            from. The leading 1 broadcasts over the batch.
        Attention (inside each encoder layer):
            The layer projects the tokens to queries Q, keys K and values V.
            With h heads each head has width d_model/h. For one head:
                scores = Q K^T / sqrt(d_head)          (T, T)
                A      = softmax(scores, dim=-1)
                head   = A V
            Each token therefore attends to every other token, and the heads
            are concatenated and projected back to d_model. A residual
            connection and LayerNorm follow the attention sublayer, then a
            position-wise feed-forward network with another residual and
            LayerNorm. There is no causal mask here: this is classification,
            not language modelling, so every row may see the whole image.
        Pooling and output:
            The encoder returns (B, T, d_model). Mean pooling over the T token
            outputs gives one (B, d_model) vector per image, and a linear head
            maps it to class logits.
        Input:
            - input_shape (tuple): image shape (C, H, W), no batch dimension
            - d_model (int): token embedding width; must be divisible by nhead
            - nhead (int): number of attention heads
            - num_layers (int): number of stacked encoder layers
            - dim_feedforward (int): hidden width of each feed-forward network
            - dropout (float): dropout probability inside the encoder
            - num_classes (int): number of output logits
        Output:
            - logits of shape (B, num_classes), no softmax
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

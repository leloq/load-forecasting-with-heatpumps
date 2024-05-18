import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    without dropout
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1), :]
        return x


class EncoderOnlyTransformer(nn.Module):
    def __init__(
            self,
            input_length,
            input_dim,
            output_length,
            num_layers,
            d_model,
            n_heads,
            dropout: float = 0.1
    ):
        super(EncoderOnlyTransformer, self).__init__()

        self.dropout = dropout

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=input_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, layer_norm)
        self.output_layer = nn.Linear(input_length * d_model, output_length)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output_layer(x)
        return x


class EncoderDecoderTransformer(nn.Module):
    def __init__(
            self,
            input_length,
            input_dim,
            output_length,
            num_layers,
            d_model,
            n_heads,
            dropout: float = 0.1
    ):
        super(EncoderDecoderTransformer, self).__init__()

        self.dropout = dropout

        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=input_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, layer_norm)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, layer_norm)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_dec):
        x_enc = self.encoder_embedding(x_enc)
        x_enc = self.positional_encoding(x_enc)
        x_dec = self.decoder_embedding(x_dec)
        x_dec = self.positional_encoding(x_dec)
        x_enc = self.encoder(x_enc)
        x_dec = self.decoder(x_dec, memory=x_enc)
        x_dec = self.output_layer(x_dec)[:, :, 0]
        return x_dec

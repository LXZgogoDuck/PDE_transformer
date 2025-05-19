import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_dim=784, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence length dimension
        x = self.pos_encoding(x)
        attn_weights = []
        for layer in self.transformer.layers:
            x, attn = layer.self_attn(x, x, x, need_weights=True)
            attn_weights.append(attn)
        logits = self.fc(x.mean(dim=1))
        return logits, attn_weights

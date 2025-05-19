import torch
import torch.nn as nn
from models.transformer_with_attention import PositionalEncoding

class ImprovedPDEModel(nn.Module):
    def __init__(self, d_model=64, num_steps=5, num_heads=4):
        super().__init__()
        self.diffusion = nn.ParameterList([nn.Parameter(torch.rand(d_model)) for _ in range(num_steps)])
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.nonlinear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_steps)
        ])
        self.pos_encoding = PositionalEncoding(d_model)
        self.dynamic_weight = nn.Linear(d_model, 3)
        self.num_steps = num_steps

    def forward(self, initial_state):
        states = [initial_state]
        attn_weights = []
        state = self.pos_encoding(initial_state)

        for i in range(self.num_steps):
            laplacian = torch.roll(state, 1, dims=1) + torch.roll(state, -1, dims=1) - 2 * state
            diffusion = self.diffusion[i].unsqueeze(0).unsqueeze(0) * laplacian
            attn_output, attn = self.attention(state, state, state, need_weights=True)
            nonlinear = self.nonlinear[i](state)

            weights = torch.softmax(self.dynamic_weight(state.mean(dim=1)), dim=-1)
            state = (weights[:, 0].unsqueeze(1).unsqueeze(2) * state +
                     weights[:, 1].unsqueeze(1).unsqueeze(2) * (diffusion + attn_output) +
                     weights[:, 2].unsqueeze(1).unsqueeze(2) * nonlinear)

            states.append(state)
            attn_weights.append(attn)

        return states, attn_weights

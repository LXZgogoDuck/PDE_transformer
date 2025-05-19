# File: models/transformer.py
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    '''
    input x: (batch_size, input_dim) input_dim: number of features per input sample (28*28 for minist)
    after embedding -> (batch_size, d_model) -> unsqueeze: add a new dimension (b, squence_length, d)

    '''
    def __init__(self, input_dim=784, d_model=64, nhead=4, num_layers=9):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.embedding(x).unsqueeze(1)  # Shape: (batch_size, seq_len=1, d_model)
        hidden_states = [x.squeeze(1)]  # Capture initial embedding state

        for layer in self.transformer.layers:
            x = layer(x)
            hidden_states.append(x.squeeze(1))

        logits = self.fc(x.squeeze(1))  # Final output logits
        return logits, hidden_states


class PDEModel(nn.Module):
    def __init__(self, d_model=64, num_steps=9, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps
        self.num_heads = num_heads

        # Learnable diffusion coefficients for each step
        self.diffusion_coeff = nn.Parameter(torch.rand(num_steps))

        # Non-linear transformation (Residuals)
        self.nonlinear = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        ) for _ in range(num_steps)])

        # Multihead attention for the attention mechanism
        self.attention = nn.MultiheadAttention(d_model, num_heads)

        # Time step parameters for dynamic influence
        self.time_steps = nn.Parameter(torch.rand(num_steps))

    def forward(self, initial_state):
        states = [initial_state]
        state = initial_state

        for i in range(self.num_steps):
            # Diffusion term D∇²u
            laplacian = torch.roll(state, 1, dims=1) + torch.roll(state, -1, dims=1) - 2 * state
            diffusion = self.diffusion_coeff[i] * laplacian

            # Non-linear transformation term R(u, θ)
            nonlinear = self.nonlinear[i](state)

            # Attention term A(u)
            attn_output, attn = self.attention(state, state, state, need_weights=True)

            # Combine all the terms (Diffusion + Attention + Residual)
            state = state + self.time_steps[i] * (diffusion + nonlinear + attn_output)
            states.append(state)

        return states

class PDEsimpleModel(nn.Module):
    def __init__(self, d_model, num_steps):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

    def forward(self, initial_state):
        states = [initial_state]
        state = initial_state
        for _ in range(self.num_steps):
            state = state + 0.1 * (torch.roll(state, 1, dims=1) +
                                   torch.roll(state, -1, dims=1) - 2 * state)
            states.append(state)
        return states

import torch
import torch.nn as nn

class ImprovedTransformer(nn.Module):
    def __init__(self, input_dim=784, d_model=64, nhead=4, num_layers=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (B, 1, D)
        hidden_states = [x.squeeze(1)]
        for layer in self.transformer.layers:
            x = layer(x)
            hidden_states.append(x.squeeze(1))
        logits = self.fc(x.squeeze(1))
        return logits, hidden_states



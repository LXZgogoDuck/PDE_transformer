import torch
import torch.nn as nn

class ImprovedPDEModel(nn.Module):
    def __init__(self, d_model=64, num_steps=5, num_heads=4):
        super().__init__()
        self.diffusion_coeff = nn.Parameter(torch.rand(num_steps))
        self.nonlinear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_steps)
        ])
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.time_steps = nn.Parameter(torch.rand(num_steps))

    def forward(self, initial_state):
        states = [initial_state]
        state = initial_state
        for i in range(self.time_steps.size(0)):
            laplacian = torch.roll(state, 1, dims=1) + torch.roll(state, -1, dims=1) - 2 * state
            diffusion = self.diffusion_coeff[i] * laplacian
            nonlinear = self.nonlinear[i](state)
            attn_output, _ = self.attention(state, state, state)
            state = state + self.time_steps[i] * (diffusion + nonlinear + attn_output)
            states.append(state)
        return states

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.visualization import visualize_flow

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# Models
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

class PDEModel:
    def __init__(self, d_model, num_steps):
        self.d_model = d_model
        self.num_steps = num_steps

    def simulate(self, initial_state):
        states = [initial_state]
        state = initial_state
        for _ in range(self.num_steps):
            state = state + 0.1 * (torch.roll(state, 1, dims=1) + torch.roll(state, -1, dims=1) - 2 * state)
            states.append(state)
        return states

# Experiment Settings
input_dim = 28 * 28
d_model = 64
nhead = 4
num_layers = 5

model = SimpleTransformer(input_dim, d_model, nhead, num_layers)
pde_model = PDEModel(d_model, num_steps=10)

# Get One Batch of Data
images, _ = next(iter(train_loader))
inputs = images.view(-1, 28 * 28)

# Transformer Information Flow
transformer_states = []
with torch.no_grad():
    x = model.embedding(inputs)
    transformer_states.append(x)
    for layer in model.transformer.layers:
        x = layer(x)
        transformer_states.append(x)

# PDE Information Flow
initial_state = transformer_states[0][0].unsqueeze(0)  # Select first sample for visualization
pde_states = pde_model.simulate(initial_state)

# Visualization
visualize_flow(transformer_states, "Transformer Information Flow",
                fixed_scale=True, save_path="results/transformer_info_flow.png")

visualize_flow(pde_states, "PDE Model Information Flow",
                fixed_scale=True, save_path="results/pde_info_flow.png")

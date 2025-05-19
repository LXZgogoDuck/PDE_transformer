# File: experiments/information_flow_experiment.py

import torch
import matplotlib.pyplot as plt
from data.dataset import get_dataloader
from models.transformer import SimpleTransformer, PDEsimpleModel, PDEModel
from utils.visualization import plot_heatmaps, plot_similarity
from utils.similarity import compute_cosine_similarity


def run_comparison_experiment(dataset_name="MNIST", batch_size=1, num_steps=10):
    # Step 1: Load the Data
    dataloader, input_dim = get_dataloader(name=dataset_name, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Initialize Models
    d_model = 64
    transformer_model = SimpleTransformer(input_dim=input_dim, d_model=d_model).to(device)
    simple_pde_model = PDEsimpleModel(d_model=d_model, num_steps=num_steps).to(device)
    advanced_pde_model = PDEModel(d_model=d_model, num_steps=num_steps, num_heads=4).to(device)

    # Step 3: Get one batch of data
    images, _ = next(iter(dataloader))
    inputs = images.view(images.size(0), -1).to(device)

    # Step 4: Run the Transformer model
    with torch.no_grad():
        _, transformer_states = transformer_model(inputs)

    # Step 5: Run the Simple PDE model (Linear Diffusion)
    simple_pde_states = simple_pde_model(transformer_states[0])

    # Step 6: Run the Advanced PDE model (with Learnable Diffusion and Attention)
    advanced_pde_states = advanced_pde_model(transformer_states[0])

    # Step 7: Visualize Information Flow
    plot_heatmaps(transformer_states, "Transformer Hidden States", fixed_scale=True, save_path="results/info_flow/transformer_states.png")
    plot_heatmaps(simple_pde_states, "Simple PDE Model States", fixed_scale=True, save_path="results/info_flow/simple_pde_states.png")
    plot_heatmaps(advanced_pde_states, "Advanced PDE Model States", fixed_scale=True, save_path="results/info_flow/advanced_pde_states.png")

    # Step 8: Compute Similarity between Models (Simple PDE vs. Advanced PDE)
    simple_pde_similarities = [
        compute_cosine_similarity(simple_state.cpu().numpy(), transformer_state.cpu().numpy())
        for simple_state, transformer_state in zip(simple_pde_states[1:], transformer_states[1:])
    ]

    advanced_pde_similarities = [
        compute_cosine_similarity(advanced_state.cpu().detach().numpy(), transformer_state.cpu().detach().numpy())
        for advanced_state, transformer_state in zip(advanced_pde_states[1:], transformer_states[1:])
    ]

    # Step 9: Plot Similarities
    plot_similarity(simple_pde_similarities, title="Simple PDE vs Transformer Similarity", save_path="results/info_flow/simple_pde_vs_transformer.png")
    plot_similarity(advanced_pde_similarities, title="Advanced PDE vs Transformer Similarity", save_path="results/info_flow/advanced_pde_vs_transformer.png")


if __name__ == "__main__":
    for dataset in ["MNIST"]:
        print(f"\nRunning Information Flow Comparison on {dataset}")
        run_comparison_experiment(dataset_name=dataset)
#
# def run_information_flow_experiment(dataset_name="MNIST", batch_size=1, num_steps=9):
#     # Step 1: Load Data
#     dataloader, input_dim = get_dataloader(name=dataset_name, batch_size=batch_size)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Step 2: Initialize Models
#     d_model = 64
#     transformer_model = SimpleTransformer(input_dim=input_dim, d_model=d_model).to(device)
#     #pde_model = PDEModel(d_model=d_model, num_steps=num_steps).to(device)
#     pde_model = PDEModel(d_model=d_model, num_steps=num_steps)
#
#     # Step 3: Run Experiment on One Batch
#     images, _ = next(iter(dataloader))
#     inputs = images.view(images.size(0), -1).to(device)
#
#     with torch.no_grad():
#         # Transformer Information Flow
#         _, transformer_states = transformer_model(inputs)
#
#         # PDE Simulation starting from the initial Transformer state
#         pde_states = pde_model(transformer_states[0])
#
#     # Step 4: Visualization
#     plot_heatmaps(transformer_states, "Transformer Hidden States", fixed_scale=True,
#                   save_path="results/info_flow/transformer_states.png")
#     plot_heatmaps(pde_states, "PDE information flow", fixed_scale=True,
#                   save_path="results/info_flow/pde_flow.png")
#
#     # Step 5: Compute and Plot Similarity
#     similarities = [compute_cosine_similarity(p.cpu().numpy(), t.cpu().numpy())
#                     for p, t in zip(pde_states[1:], transformer_states[1:])]
#     plot_similarity(similarities, title="Cosine Similarity between PDE and Transformer", save_path="results/info_flow/similarity.png")
#
#     return similarities
#
# if __name__ == "__main__":
#     # for dataset in ["MNIST", "FashionMNIST", "CIFAR10"]:
#     for dataset in ["MNIST"]:
#         print(f"\nRunning Information Flow Experiment on {dataset}")
#         sims = run_information_flow_experiment(dataset_name=dataset)
#         print(f"Similarity Scores for {dataset}: {sims}")

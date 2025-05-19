import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg', 'Qt5Agg', depending on your system
import os

def plot_heatmaps(states, title, fixed_scale=False, save_path=None):
    fig, axes = plt.subplots(1, len(states), figsize=(20, 4))

    # Compute global vmin and vmax if fixed_scale is True
    if fixed_scale:
        all_values = np.concatenate([state[0].detach().cpu().numpy().flatten() for state in states])
        vmin, vmax = np.min(all_values), np.max(all_values)
    else:
        vmin = vmax = None  # Let seaborn decide per plot

    for idx, state in enumerate(states):
        vec = state[0].detach().cpu().numpy().flatten()
        side = int(np.ceil(np.sqrt(vec.shape[0])))
        padded_vec = np.zeros((side * side,))
        padded_vec[:vec.shape[0]] = vec
        mat = padded_vec.reshape(side, side)

        sns.heatmap(mat, ax=axes[idx], cmap='viridis', vmin=vmin, vmax=vmax, cbar=(idx == len(states) - 1))
        axes[idx].set_title(f'{title} {idx}')
        axes[idx].axis('off')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_similarity(similarities, title="Cosine Similarity", save_path=None):
    plt.plot(similarities, marker='o')
    plt.title(title)
    plt.xlabel('Layer/Step')
    plt.ylabel('Similarity')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_flow(states, title, fixed_scale=True, save_path=None):
    # Prepare figure layout
    num_states = min(len(states), 10)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Compute global vmin and vmax if fixed scale is requested
    if fixed_scale:
        all_vals = np.concatenate([state.detach().cpu().numpy().flatten() for state in states[:num_states]])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
    else:
        vmin = vmax = None

    for i, state in enumerate(states[:num_states]):
        state_np = state.detach().cpu().numpy()
        # If state is 2D (batch_size, features), pick first batch
        if state_np.ndim == 2:
            state_np = state_np[0]

        side = int(np.ceil(np.sqrt(state_np.shape[0])))
        padded = np.zeros((side * side,))
        padded[:state_np.shape[0]] = state_np
        mat = padded.reshape(side, side)

        sns.heatmap(mat, ax=axes[i], cmap='viridis', vmin=vmin, vmax=vmax, cbar=(i == num_states - 1))
        axes[i].set_title(f'{title} Step {i}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_states(states, title, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None

    fig, axes = plt.subplots(1, min(len(states), 5), figsize=(20, 4))
    vmin, vmax = np.min([s.detach().cpu().numpy() for s in states]), np.max([s.detach().cpu().numpy() for s in states])

    for idx, state in enumerate(states[:5]):
        mat = state[0].detach().cpu().numpy().flatten()
        side = int(np.ceil(np.sqrt(len(mat))))
        padded = np.zeros(side * side)
        padded[:len(mat)] = mat
        mat_reshaped = padded.reshape(side, side)
        sns.heatmap(mat_reshaped, ax=axes[idx], cmap='viridis', vmin=vmin, vmax=vmax, cbar=False)
        axes[idx].set_title(f"{title} Step {idx}")
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


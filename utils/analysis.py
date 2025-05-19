import numpy as np
import os
import matplotlib.pyplot as plt

def compute_feature_correlation(states):
    correlations = []
    for i in range(1, len(states)):
        a = states[i-1][0].detach().cpu().numpy().flatten()
        b = states[i][0].detach().cpu().numpy().flatten()
        corr = np.corrcoef(a, b)[0, 1]
        correlations.append(corr)
    return correlations

def plot_correlation(transformer_corr, pde_corr, digit, exp_num, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None
    plt.plot(transformer_corr, 'b-o', label='Transformer')
    plt.plot(pde_corr, 'r-o', label='PDE')
    plt.title(f'Correlation Comparison (Digit: {digit})')
    plt.xlabel('Layer')
    plt.ylabel('Correlation')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

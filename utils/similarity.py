import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

def compute_cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return cosine_similarity(a_flat.reshape(1, -1), b_flat.reshape(1, -1))[0, 0]

def compute_kl_divergence(a, b):
    a_flat = np.clip(a.flatten(), 1e-10, 1.0)
    b_flat = np.clip(b.flatten(), 1e-10, 1.0)
    return entropy(a_flat, b_flat)

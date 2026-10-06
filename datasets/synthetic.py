"""
Synthetic Dataset Generators with configurable controls
"""

from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
import numpy as np

def get_synthetic_data(dataset_type:str, n_samples:int=300, noise:float=0.1, random_seed:int=42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D synthetic classification dataset.

    Parameters:
        dataset_type: One of 'moons', 'circles', 'blobs', 'linear', 'xor'
        n_samples: Total number of samples
        noise: Noise level (std dev of Gaussian noise added to data)
        random_seed: Random seed for reproducibility
    
    Returns:
        X: Feature matrix of shape (n_samples, 2)
        y: Label array of shape (n_samples,)
    """
    rng = np.random.RandomState(random_seed)

    if dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_seed)

    elif dataset_type == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_seed)

    elif dataset_type == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise * 3 + 0.5, random_state=random_seed)

    elif dataset_type == "linear":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=noise, random_state=random_seed)
    
    elif dataset_type == "xor":
        # XOR: 4 quadrants, label = (x1 > 0) XOR (x2 > 0)
        X = rng.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
        # adding noise
        X += rng.normal(scale=noise, size=X.shape)
    else:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'"
            "Choose from Moons, Circles, Blobs, Linear, XOR"
        )

    return X, y

# Dataset metadata for UI rendering
SYNTHETIC_DATASETS = {
    "moons": {
        "label": "Two Moons",
        "description": "Two interleaving half-circles(crescent moons). Tests non-linear boundaries.",
        "n_classes": 2
    },
    "circles": {
        "label": "Concentric Circles",
        "description": "Inner and outer rings. Requires radial decision boundary.",
        "n_classes": 2
    },
    "blobs": {
        "label": "Gaussian Blobs",
        "description": "Three isotropic Gaussian clusters. Classic multi-class task.",
        "n_classes": 3
    },
    "linear": {
        "label": "Linearly Separable Data",
        "description": "Two classes separable by a linear decision boundary.",
        "n_classes": 2
    },
    "xor": {
        "label": "XOR Pattern",
        "description": "XOR logic gate layout. No linear separator exists.",
        "n_classes": 2
    }
}
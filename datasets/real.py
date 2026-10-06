"""
Real dataset loaders from sklearn.datasets.
Supports: Iris, Wine, Breast Cancer
Includes feature selection for 2D visualization.
"""

import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

REAL_DATASETS = {
    "iris": {
        "label": "Iris",
        "description": "3-class flower classification. 150 samples, 4 features",
        "loader": load_iris,
        "n_classes": 3
    },
    "wine": {
        "label": "Wine",
        "description": "3-class wine origin recognition. 178 samples, 13 features",
        "loader": load_wine,
        "n_classes": 3
    },
    "breast_cancer": {
        "label": "Wisconsin Breast Cancer",
        "description": "Binary tumor classification. 569 samples, 30 features",
        "loader": load_breast_cancer,
        "n_classes": 2
    },
}

def get_real_dataset(dataset_name:str, feature_indices:tuple[int, int] | None):
    """
    Load a real sklearn dataset, selecting 2 features for 2D visualization.
 
    Args:
        dataset_name: One of 'iris', 'wine', 'breast_cancer'
        feature_indices: Tuple (i, j) selecting which 2 features to keep for visualization.
                 If None, the first 2 features are used for visualization.
 
    Returns:
        X:             Full feature matrix
        X_vis:         2D feature slice used for visualization
        y:             Integer class labels
        feature_names: List of feature name strings
        class_names:   List of class name strings
    """
    if dataset_name not in REAL_DATASETS:
        raise ValueError(
            f"Unknown Dataset '{dataset_name}'"
            f"Choose from {list(REAL_DATASETS.keys())}"
        )

    loader = REAL_DATASETS[dataset_name]["loader"]
    load_data = loader()

    X = load_data.data
    y = load_data.target
    feature_names = load_data.feature_names
    class_names = load_data.target_names

    # Pick 2 features for visualization
    if feature_indices is not None:
        i, j = feature_indices
        X_vis = X[:, [i, j]]
        return X, X_vis, y, feature_names, class_names
    else:
        X_vis = X[:, :2] if X.shape[1] >= 2 else X
        return X, X_vis, y, feature_names, class_names

def get_feature_names(dataset_name:str) -> list[str]:
    """
    Return all feature names for a given real dataset (for the dropdown UI).
    """
    loader = REAL_DATASETS[dataset_name]["loader"]
    load_data = loader()
    return load_data.feature_names

def dataset_as_df(dataset_name:str) -> pd.DataFrame:
    """
    Return the full dataset as a pandas DataFrame (all features + target column).
    Useful for preview tables and CSV export.
    """
    loader = REAL_DATASETS[dataset_name]["loader"]
    load_data = loader()
    df = load_data.frame
    return df
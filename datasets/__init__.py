from .synthetic import get_synthetic_data, SYNTHETIC_DATASETS
from .real import get_real_dataset, get_feature_names, dataset_as_df, REAL_DATASETS

__all__ = [
    "get_synthetic_data",
    "SYNTHETIC_DATASETS",
    "get_real_dataset",
    "get_feature_names",
    "dataset_as_df",
    "REAL_DATASETS",
]
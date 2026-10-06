import numpy as np
from sklearn.model_selection import learning_curve, validation_curve

def compute_learning_curve(clf, X_train, y_train, cv=5, scoring="accuracy", n_points=8):
    """"
    Returns train_sizes, train_scores_mean, train_scores_std,
    val_scores_mean, val_scores_std.
    """
    n_max = X_train.shape[0]
    train_sizes_abs = np.linspace(0.1, 1.0, n_points)

    train_sizes, train_scores, val_scores = learning_curve(
        clf,
        X_train, y_train,
        train_sizes=train_sizes_abs,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    return (
        train_sizes,
        train_scores.mean(axis=1),
        train_scores.std(axis=1),
        val_scores.mean(axis=1),
        val_scores.std(axis=1)
    )

def compute_validation_curve(clf, X_train, y_train, param_name, param_range, cv=5, scoring="accuracy"):
    """
    Returns param_range, train_scores_mean, train_scores_std,
    val_scores_mean, val_scores_std.
    """
    train_scores, val_scores = validation_curve(
        clf,
        X_train, y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return (
        param_range,
        train_scores.mean(axis=1),
        train_scores.std(axis=1),
        val_scores.mean(axis=1),
        val_scores.std(axis=1)
    )
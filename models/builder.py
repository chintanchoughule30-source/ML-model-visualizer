"""
Model builder — instantiates sklearn models from registry param dicts.
 
Also wraps models in a Pipeline with StandardScaler so every model
receives scaled features (critical for SVM, KNN, LR, LDA).
 
Usage:
    from models.builder import build_model
    clf = build_model("svm", {"kernel": "rbf", "C": 2.0, "gamma": "scale", "degree": 3})
    clf.fit(X_train, y_train)
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from models.registry import MODEL_REGISTRY

# Models that benefit from or require scaling
_NEEDS_SCALING = {
    "logistic_regression",
    "knn",
    "svm",
    "LDA",
    "naive_bayes",
}

def build_model(model_key:str, user_params:dict):
    """
    Build a pipeline wrapped sklearn-based model.

    Args:
        model_key: Key into MODEL_REGISTRY (e.g. 'svm', 'random_forest').
        user_params: Dict of hyperparameter values from the sidebar widgets.
                     Keys match the 'params' keys in the registry entry.
    Returns:
        A fitted-ready sklearn estimator or pipeline.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown Model key: '{model_key}'")
    
    entry = MODEL_REGISTRY[model_key]
    sklearn_cls = entry["sklearn_cls"]
    fixed_params = entry.get("fixed_params", {})

    # Naive Bayes
    if model_key == "naive_bayes":
        user_params = dict(user_params)
        # var_smoothing is stored as log10
        user_params["var_smoothing"] = 10 ** user_params["var_smoothing"]
    # LDA
    if model_key == "LDA":
        user_params = dict(user_params)
        #LDA shrinkage 'none' → None
        if user_params.get("shrinkage") == "none":
            user_params["shrinkage"] = None
        # shrinkage only valid with lsqr or eigen solvers
        if user_params.get("solver") == "svd":
            user_params["shrinkage"] = None
    # Random Forest
    if model_key == "random_forest":
        user_params = dict(user_params)
        # Random Forest max_features 'none' → None
        if user_params.get("max_features") == "none":
            user_params["max_features"] = None
    # Logistic Regression
    if model_key == "logistic_regression":
        user_params = dict(user_params)
        # penalty 'none' string → None
        l1_ratio = user_params.get("l1_ratio", 0.0)
        solver = user_params.get("solver", "lbfgs")
        if l1_ratio == "none":
            user_params["penalty"] = None
        # Fix incompatible solver/penalty combos silently
        if l1_ratio == 1.0 and solver not in ("liblinear", "saga"):
            user_params["solver"] = "saga"
        elif l1_ratio < 1.0 and l1_ratio > 0.0 and solver != "saga":
            user_params["solver"] = "saga"
            user_params["l1_ratio"] = 0.5   # required for elasticnet
    # Voting Classifier
    if model_key == "voting_classifier":
        return _build_voting_classifier(user_params)
    
    # Filter user_params to only keys accepted by this model's constructor
    # (avoids passing toggle-only UI params like 'bootstrap' accidentally)
    import inspect
    valid_keys = set(inspect.signature(sklearn_cls.__init__).parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in user_params.items() if k in valid_keys}
    all_params = {**filtered_params, **fixed_params}
    estimator = sklearn_cls(**all_params)

    if model_key in _NEEDS_SCALING:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])
    return estimator

def _build_voting_classifier(user_params:dict) -> Pipeline:
    """Builds a Voting Classifier from the toggle selections."""
    voting_strategy = user_params.get("voting", "soft")
    estimators = []

    if user_params.get("include_lr", True):
        estimators.append(("lr", LogisticRegression(max_iter=200, random_state=42)))
    if user_params.get("include_knn", True):
        estimators.append(("knn", KNeighborsClassifier(n_neighbors=5)))
    if user_params.get("include_tree", True):
        estimators.append(("tree", DecisionTreeClassifier(max_depth=5, random_state=42)))

    if len(estimators) < 2:
        # Always include at least LR+KNN
        estimators = [
            ("lr", LogisticRegression(max_iter=200, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ]
    clf = VotingClassifier(estimators=estimators, voting=voting_strategy)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

def get_default_params(model_key:str) -> dict:
    """Return the default hyperparameter values for a model (used by 'Reset' button)."""
    entry = MODEL_REGISTRY[model_key]
    return {key: spec["default"] for key, spec in entry["params"].items()}
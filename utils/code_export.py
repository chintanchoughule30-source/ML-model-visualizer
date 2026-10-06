# ── Per-model import map ────────────────────────────────────────────────────
_MODEL_IMPORT = {
    "logistic_regression": "from sklearn.linear_model import LogisticRegression",
    "knn":                 "from sklearn.neighbors import KNeighborsClassifier",
    "decision_tree":       "from sklearn.tree import DecisionTreeClassifier",
    "random_forest":       "from sklearn.ensemble import RandomForestClassifier",
    "gradient_boosting":   "from sklearn.ensemble import GradientBoostingClassifier",
    "svm":                 "from sklearn.svm import SVC",
    "naive_bayes":         "from sklearn.naive_bayes import GaussianNB",
    "LDA":                 "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
    "adaboost":            "from sklearn.ensemble import AdaBoostClassifier",
    "voting_classifier": (
        "from sklearn.ensemble import VotingClassifier\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.tree import DecisionTreeClassifier"
    )
}

# Models that are wrapped in a StandardScaler Pipeline
_NEEDS_SCALER = {
    "logistic_regression",
    "knn",
    "svm",
    "LDA",
    "naive_bayes",
    "voting_classifier"
}

_FIXED_PARAMS = {
    "logistic_regression": {"random_state": 42},
    "decision_tree": {"random_state": 42},
    "random_forest": {"random_state": 42},
    "gradient_boosting": {"random_state": 42},
    "svm": {"probability": True},
    "adaboost": {"random_state": 42},
}

_ALLOWED_PARAMS = {
    "logistic_regression": {"C", "l1_ratio", "solver", "max_iter", "penalty"},
    "knn": {"n_neighbors", "weights", "metric"},
    "decision_tree": {"max_depth", "criterion", "min_samples_split"},
    "random_forest": {"n_estimators", "max_depth", "max_features", "bootstrap"},
    "gradient_boosting": {"n_estimators", "learning_rate", "max_depth", "subsample"},
    "svm": {"kernel", "C", "gamma", "degree"},
    "naive_bayes": {"var_smoothing"},
    "adaboost": {"n_estimators", "learning_rate"},
    "LDA": {"solver", "shrinkage"}
}


def _fmt(v):
    """Format parameter values as valid Python literals."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return repr(round(v, 8))
    return repr(v)


def _normalize_user_params(model_key: str, user_params: dict) -> dict:
    """Mirror model.builder adjustments so exported scripts match app behavior."""
    params = dict(user_params)

    if model_key == "naive_bayes" and "var_smoothing" in params:
        params["var_smoothing"] = 10 ** params["var_smoothing"]

    if model_key == "random_forest" and params.get("max_features") == "none":
        params["max_features"] = None

    if model_key == "logistic_regression":
        l1_ratio = params.get("l1_ratio", 0.0)
        solver = params.get("solver", "lbfgs")
        if l1_ratio == "none":
            params["penalty"] = None
        if l1_ratio == 1.0 and solver not in ("liblinear", "saga"):
            params["solver"] = "saga"
        elif 0.0 < l1_ratio < 1.0 and solver != "saga":
            params["solver"] = "saga"
            params["l1_ratio"] = 0.5

    if model_key in {"LDA", "lda"}:
        if params.get("shrinkage") == "none":
            params["shrinkage"] = None
        if params.get("solver") == "svd":
            params["shrinkage"] = None

    return params


def _build_voting_constructor(user_params: dict) -> str:
    """Build the VotingClassifier constructor string from toggle selections."""
    voting_strategy = user_params.get("voting", "soft")
    estimators = []

    if user_params.get("include_lr", True):
        estimators.append("('lr', LogisticRegression(max_iter=200, random_state=42))")
    if user_params.get("include_knn", True):
        estimators.append("('knn', KNeighborsClassifier(n_neighbors=5))")
    if user_params.get("include_tree", True):
        estimators.append("('tree', DecisionTreeClassifier(max_depth=5, random_state=42))")

    if len(estimators) < 2:
        estimators = [
            "('lr', LogisticRegression(max_iter=200, random_state=42))",
            "('knn', KNeighborsClassifier(n_neighbors=5))",
        ]

    return f"VotingClassifier(estimators=[{', '.join(estimators)}], voting={_fmt(voting_strategy)})"

# ── Per-model constructor map ───────────────────────────────────────────────
def _build_constructor(model_key: str, user_params: dict) -> str:
    """Return the model constructor call as a string, e.g. 'SVC(C=1.0, kernel="rbf")'"""
    normalized_params = _normalize_user_params(model_key, user_params)

    if model_key in {"voting_classifier", "voting"}:
        return _build_voting_constructor(normalized_params)

    class_map = {
        "logistic_regression": "LogisticRegression",
        "knn":                 "KNeighborsClassifier",
        "decision_tree":       "DecisionTreeClassifier",
        "random_forest":       "RandomForestClassifier",
        "gradient_boosting":   "GradientBoostingClassifier",
        "svm":                 "SVC",
        "naive_bayes":         "GaussianNB",
        "LDA":                 "LinearDiscriminantAnalysis",
        "lda":                 "LinearDiscriminantAnalysis",
        "adaboost":            "AdaBoostClassifier",
        "voting_classifier":   "VotingClassifier",
        "voting":              "VotingClassifier",
    }
    cls = class_map.get(model_key, "UnknownModel")

    valid_keys = _ALLOWED_PARAMS.get(model_key, set(normalized_params.keys()))
    filtered_params = {k: v for k, v in normalized_params.items() if k in valid_keys}
    all_params = {**filtered_params, **_FIXED_PARAMS.get(model_key, {})}

    params_str = ", ".join(f"{k}={_fmt(v)}" for k, v in all_params.items())
    return f"{cls}({params_str})"


#---Synthetic dataset loader snippet---#
_SYNTHETIC_LOADER = {
    "moons": (
        "from sklearn.datasets import make_moons\n"
        "X, y = make_moons(n_samples={n_samples}, noise={noise}, random_state={seed})"
    ),
    "circles": (
        "from sklearn.datasets import make_circles\n"
        "X, y = make_circles(n_samples={n_samples}, noise={noise}, factor=0.5, random_state={seed})"
    ),
    "blobs": (
        "from sklearn.datasets import make_blobs\n"
        "X, y = make_blobs(n_samples={n_samples}, centers=3, cluster_std={noise} * 3 + 0.5, random_state={seed})"
    ),
    "linear": (
        "from sklearn.datasets import make_classification\n"
        "X, y = make_classification(n_samples={n_samples}, n_features=2, n_informative=2, "
        "n_redundant=0, n_clusters_per_class=1, flip_y={noise}, random_state={seed})"
    ),
    "xor": (
        "import numpy as np\n"
        "rng = np.random.RandomState({seed})\n"
        "X = rng.randn({n_samples}, 2)\n"
        "y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)\n"
        "X += rng.normal(scale={noise}, size=X.shape)"
    ),
}

_REAL_LOADER = {
    "iris":          "from sklearn.datasets import load_iris\ndata = load_iris()\nX, y = data.data, data.target",
    "wine":          "from sklearn.datasets import load_wine\ndata = load_wine()\nX, y = data.data, data.target",
    "breast_cancer": "from sklearn.datasets import load_breast_cancer\ndata = load_breast_cancer()\nX, y = data.data, data.target",
}


def generate_export_code(
    model_key: str,
    user_params: dict,
    dataset_source: str,       # "Synthetic" | "Real (sklearn)"
    dataset_key: str,
    n_samples: int = 300,
    noise: float = 0.1,
    random_seed: int = 42,
    test_split: float = 0.2,
) -> str:
    """
    Build and return a complete, runnable Python script string.
    """
    needs_scaler = model_key in _NEEDS_SCALER
    constructor  = _build_constructor(model_key, user_params)
    model_import = _MODEL_IMPORT.get(model_key, "# unknown model")

    # ── 1. Header
    lines = [
        "# ════════════════════════════════════════════════════════════",
        "# Auto-generated by ML Playground",
        "# Reproduces the exact dataset + model + hyperparameters",
        "# you configured in the app.",
        "# ════════════════════════════════════════════════════════════",
        "",
    ]

    # ── 2. Imports
    lines += [
        "# ── Imports",
        "import numpy as np",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import accuracy_score, f1_score, classification_report",
    ]
    if needs_scaler:
        lines += [
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.pipeline import Pipeline",
        ]
    lines += [model_import, ""]

    # ── 3. Dataset
    lines += ["# ── Dataset"]
    if dataset_source == "Synthetic":
        template = _SYNTHETIC_LOADER.get(
            dataset_key,
            "from sklearn.datasets import make_classification\nX, y = make_classification(n_samples={n_samples}, random_state={seed})"
        )
        lines.append(
            template.format(n_samples=n_samples, noise=noise, seed=random_seed)
        )
    else:
        loader = _REAL_LOADER.get(
            dataset_key,
            "from sklearn.datasets import load_iris\ndata = load_iris()\nX, y = data.data, data.target  # fallback"
        )
        lines.append(loader)
    lines.append("")

    # ── 4. Train / test split
    lines += [
        "# ── Train / test split",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y,",
        f"    test_size={test_split},",
        f"    random_state={random_seed},",
        f"    stratify=y,",
        f")",
        "",
    ]

    # ── 5. Model
    lines += ["# ── Model"]
    if needs_scaler:
        lines += [
            f"# {constructor} is wrapped in a Pipeline with StandardScaler",
            f"# because this model is sensitive to feature scale.",
            f"clf = Pipeline([",
            f'    ("scaler", StandardScaler()),',
            f'    ("clf", {constructor}),',
            f"])",
        ]
    else:
        lines += [
            f"clf = {constructor}",
        ]
    lines.append("")

    # ── 6. Training
    lines += [
        "# ── Training",
        "clf.fit(X_train, y_train)",
        "",
    ]

    # ── 7. Evaluation
    lines += [
        "# ── Evaluation",
        "y_pred = clf.predict(X_test)",
        "",
        "print(f'Train accuracy : {clf.score(X_train, y_train):.4f}')",
        "print(f'Test  accuracy : {accuracy_score(y_test, y_pred):.4f}')",
        "print(f'Test  F1 score : {f1_score(y_test, y_pred, average=\"weighted\"):.4f}')",
        "print()",
        "print('Classification Report:')",
        "print(classification_report(y_test, y_pred))",
        "",
    ]

    # ── 8. Optional: predict_proba note
    proba_models = {
        "logistic_regression",
        "knn",
        "random_forest",
        "gradient_boosting",
        "svm",
        "naive_bayes",
        "adaboost",
        "voting_classifier",
        "voting",
        "LDA",
        "lda",
    }
    if model_key in proba_models:
        lines += [
            "# ── Probability scores (optional)",
            "# y_proba = clf.predict_proba(X_test)",
            "# Each column corresponds to a class in clf.classes_",
            "",
        ]

    return "\n".join(lines)
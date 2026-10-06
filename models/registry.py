"""
Model registry — single source of truth for every model in the playground.
 
Each entry defines:
  - label:       Display name in the UI
  - description: Short text about the model shown below the selector
  - sklearn_cls: The actual sklearn class to instantiate
  - docs_url:    Link to sklearn docs page
  - params:      Dict of hyperparameter specs used to auto-render sidebar widgets
                 and to build the model with current user settings.
 
Hyperparameter specifications:
  type        : 'slider_float' | 'slider_int' | 'selectbox' | 'toggle'
  label       : Widget label
  default     : Default value
  help        : Tooltip text shown in sidebar
  -- for sliders --
  min, max, step
  -- for selectbox --
  options     : list of values
  format_func : optional lambda str for display (not used in registry, handled in sidebar)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


MODEL_REGISTRY = {
    "logistic_regression":{
        "label": "Logistic Regression",
        "description": (
            "A linear classifier that models class probabilities via the sigmoid/softmax "
            "function. Fast, interpretable, and a strong baseline for linearly separable data. "
            "Regularization (C) prevents overfitting."
        ),
        "sklearn_cls": LogisticRegression,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "supports_predict_proba": True,
        "supports_feature_importance": False,
        "supports_coef": True,
        "params":{
            "C":{
                "type": "slider_float",
                "label": "C (Regularization Strength)",
                "default": 1.0,
                "min": 0.01,
                "max": 10.0,
                "step": 0.01,
                "help": "Inverse of regularization strength. Smaller C -> stronger regularization."
            },
            "l1_ratio":{
                "type": "slider_float",
                "label": "L1 Ratio",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.001,
                "help": "Setting l1_ratio=1 gives a pure L1-penalty, setting l1_ratio=0 a pure L2-penalty. Any value between 0 and 1 gives an Elastic-Net penalty of the form l1_ratio * L1 + (1 - l1_ratio) * L2."
            },
            "solver":{
                "type": "selectbox",
                "label": "Solver",
                "default": "lbfgs",
                "options": ["lbfgs", "liblinear", "saga", "newton-cg"],
                "help": "Optimization algorithm. Use 'saga' for L1/ElasticNet; 'lbfgs' for L2."
            },
            "max_iter":{
                "type": "slider_int",
                "label": "Max Iterations",
                "default": 200,
                "min": 50,
                "max": 1000,
                "step": 50,
                "help": "Maximum iterations for the solver to converge."
            }
        },
        "fixed_params": {"random_state": 42},
        "tips":{
            "best_practices": [
                "Standardize features — LR is sensitive to scale.",
                "Use L2 penalty by default; switch to L1 for sparse features.",
                "Increase max_iter if the solver warns about non-convergence.",
                "Try 'lbfgs' for multiclass and 'liblinear' for small datasets.",
            ],
            "pitfalls": [
                "Assumes a linear decision boundary — won't capture complex patterns.",
                "Low C (strong regularization) can underfit; high C can overfit.",
                "ElasticNet requires solver='saga' — mismatches cause errors.",
            ]
        },
        "val_curve_param": {
            "name": "clf__C", # use clf__ prefix because it's inside a Pipeline
            "label": "C (Regularization)",
            "range": np.logspace(-3, 3, 10).tolist(),
            "log_scale": True
        }
    },
    "svm":{
        "label": "Support Vector Machine",
        "description": (
            "Finds the maximum-margin hyperplane separating classes, optionally using the "
            "kernel trick to map to higher dimensions. Powerful for small/medium datasets. "
            "Sensitive to feature scaling — always standardize inputs."
        ),
        "sklearn_cls": SVC,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
        "supports_predict_proba": True,
        "supports_feature_importance": False,
        "supports_coef": False,
        "params":{
            "kernel": {
                "type": "selectbox",
                "label": "Kernel",
                "default": "rbf",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "help": "Kernel function. 'rbf' handles non-linear; 'linear' is fastest.",
            },
            "C":{
                "type": "slider_float",
                "label": "C (Regularization)",
                "default": 1.0,
                "min": 0.01,
                "max": 20.0,
                "step": 0.1,
                "help": "Penalty for misclassifications. High C -> smaller margin, less regularization."
            },
            "gamma":{
                "type": "selectbox",
                "label": "Gamma",
                "default": "scale",
                "options": ["scale", "auto"],
                "help": "Kernel coefficient for rbf/poly/sigmoid. 'scale' = 1/(n_features * X.var())."
            },
            "degree":{
                "type": "slider_int",
                "label": "Poly Degree",
                "default": 3,
                "min": 2,
                "max": 6,
                "step": 1,
                "help": "Degree for polynomial kernel only. Ignored for other kernels."
            }
        },
        "fixed_params": {"probability": True},
        "tips":{
            "best_practices": [
                "Always standardize features — SVM is scale-sensitive.",
                "Start with RBF kernel; try linear if data is high-dimensional.",
                "Grid-search C and gamma together on a log scale.",
                "Use probability=True only when you need predict_proba (slower).",
            ],
            "pitfalls": [
                "Scales poorly with n_samples (O(n²–n³) training time).",
                "gamma='scale' is usually safer than 'auto'.",
                "Polynomial kernel can be very slow with high degree.",
            ]
        },
        "val_curve_param": {
            "name": "clf__C",
            "label": "C (Regularization)",
            "range": np.logspace(-3, 3, 10).tolist(),
            "log_scale": True
        }
    },
    "knn":{
        "label": "K-Nearest Neighbors",
        "description": (
            "A non-parametric instance-based learner. Classifies a point by majority vote "
            "among its K nearest neighbors. No training phase(it does not learn a discriminative function) — all computation happens at "
            "prediction time. Sensitive to feature scaling."
        ),
        "sklearn_cls": KNeighborsClassifier,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
        "supports_predict_proba": True,
        "supports_feature_importance": False,
        "supports_coef": False,
        "params": {
            "n_neighbors":{
                "type": "slider_int",
                "label": "K (Neighbours)",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "help": "Number of nearest neighbors to consider. Low K = complex boundary; High K = smoother."
            },
            "weights":{
                "type": "selectbox",
                "label": "Weight Function",
                "default": "uniform",
                "options": ["uniform", "distance"],
                "help": "'uniform': all neighbors vote equally. 'distance': closer neighbors vote more."
            },
            "metric":{
                "type": "selectbox",
                "label": "Distance Metric",
                "default": "euclidean",
                "options": ["euclidean", "manhattan", "minkowski", "chebyshev"],
                "help": "Distance function used to find nearest neighbours."
            }
        },
        "fixed_params": {},
        "tips":{
            "best_practices": [
                "Always normalize features — KNN is distance-based.",
                "Start with k=5 and tune via cross-validation.",
                "Use 'distance' weights when data is noisy.",
                "Manhattan distance can outperform Euclidean on high-dim data.",
            ],
            "pitfalls": [
                "Very slow at prediction time on large datasets (no training phase).",
                "Suffers badly from the curse of dimensionality.",
                "k=1 almost always overfits; avoid it unless data is huge and clean.",
            ]
        },
        "val_curve_param": {
            "name": "clf__n_neighbors",
            "label": "n_neighbors",
            "range": list(range(1, 21)),
            "log_scale": False
        }
    },
    "decision_tree":{
        "label": "Decision Tree",
        "description": (
            "A hierarchical model that splits data on feature thresholds to form a tree of "
            "if-else rules. Highly interpretable and captures non-linear patterns. Prone to "
            "overfitting — control with max_depth."
        ),
        "sklearn_cls": DecisionTreeClassifier,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
        "supports_predict_proba": True,
        "supports_feature_importance": True,
        "supports_coef": False,
        "params":{
            "max_depth":{
                "type": "slider_int",
                "label": "Max Depth",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "Maximum depth of the tree. Deeper = more complex. None = grow until pure."
            },
            "criterion":{
                "type": "selectbox",
                "label": "Split Criterion",
                "default": "gini",
                "options": ["gini", "entropy", "log_loss"],
                "help": "'gini': Gini impurity. 'entropy': information gain. Both work similarly in practice."
            },
            "min_samples_split":{
                "type": "slider_int",
                "label": "Min Samples Split",
                "default": 2,
                "min": 2,
                "max": 30,
                "step": 1,
                "help": "Minimum samples required to split an internal node. Higher -> more regularization"
            },
        },
        "fixed_params": {"random_state": 42},
        "tips":{
            "best_practices": [
                "Limit max_depth (3-6) to prevent overfitting.",
                "Use min_samples_split > 2 on small/noisy datasets.",
                "Entropy and Gini produce similar results — Gini is faster.",
                "Visualize the tree to interpret decisions.",
            ],
            "pitfalls": [
                "Unconstrained trees memorize training data (overfit easily).",
                "Highly sensitive to small changes in data (high variance).",
                "Not great at extrapolating beyond training range.",
            ]
        },
        "val_curve_param": {
            "name": "max_depth",
            "label": "max_depth",
            "range": list(range(1, 21)),
            "log_scale": False
        }
    },
    "random_forest":{
        "label": "Random Forest",
        "description": (
            "An ensemble of decorrelated decision trees trained on bootstrap samples. "
            "Each tree votes on the final class. More robust than a single tree('Wisdom of the crowd'), handles "
            "high-dimensional data well, and provides feature importances."
        ),
        "sklearn_cls": RandomForestClassifier,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
        "supports_predict_proba": True,
        "supports_feature_importance": True,
        "supports_coef": False,
        "params":{
            "n_estimators":{
                "type": "slider_int",
                "label": "Number of Trees",
                "default": 100,
                "min": 10,
                "max": 500,
                "step": 10,
                "help": "More trees = better performance but slower. Diminishing returns beyond ~200."
            },
            "max_depth":{
                "type": "slider_int",
                "label": "Max Depth",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "Max depth per tree. None = fully grown trees."
            },
            "max_features":{
                "type": "selectbox",
                "label": "Max Features per split",
                "default": "sqrt",
                "options": ["sqrt", "log2", "none"],
                "help": "'sqrt': √n_features (default, good for classification). 'log2': log₂(n_features)."
            },
            "bootstrap":{
                "type": "toggle",
                "label": "Bootstrap Sampling",
                "default": True,
                "help": "If True, each tree trains on a bootstrap sample. Disabling removes bagging."
            },
        },
        "fixed_params": {"random_state": 42},
        "tips":{
            "best_practices": [
                "Start with n_estimators=100; more trees rarely hurt but slow training.",
                "max_features='sqrt' is a solid default for classification.",
                "Use feature importances to identify irrelevant features.",
                "Enable bootstrap=True (default) for variance reduction.",
            ],
            "pitfalls": [
                "Memory-intensive for large n_estimators + deep trees.",
                "Feature importances can be biased toward high-cardinality features.",
                "Not easily interpretable compared to a single Decision Tree.",
            ]
        },
        "val_curve_param": {
            "name": "n_estimators",
            "label": "n_estimators",
            "range": [10, 25, 50, 75, 100, 150, 200],
            "log_scale": False
        }
    },
    "gradient_boosting":{
        "label": "Gradient Boosting",
        "description": (
            "Builds trees sequentially — each tree corrects the errors of the previous. "
            "Often the strongest out-of-box performer on tabular data. "
            "More sensitive to hyperparameters than Random Forest."
        ),
        "sklearn_cls": GradientBoostingClassifier,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
        "supports_predict_proba": True,
        "supports_feature_importance": True,
        "supports_coef": False,
        "params":{
            "n_estimators":{
                "type": "slider_int",
                "label": "Number of Estimators",
                "default": 100,
                "min": 10,
                "max": 300,
                "step": 10,
                "help": "Number of boosting rounds. More = better but slower; use with low learning_rate."
            },
            "learning_rate":{
                "type": "slider_float",
                "label": "Learning Rate",
                "default": 0.1,
                "min": 0.001,
                "max": 1.0,
                "step": 0.005,
                "help": "Shrinks each tree's contribution. Lower rate + more trees = better generalization."
            },
            "max_depth":{
                "type": "slider_int",
                "label": "Max Depth",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "help": "Depth of each individual tree. Keep low (3-5) for GB."
            },
            "subsample":{
                "type": "slider_float",
                "label": "Subsample Ratio",
                "default": 1.0,
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "help": "Fraction of samples used per tree. < 1.0 adds stochasticity (Stochastic GB)."
            },
        },
        "fixed_params": {"random_state": 42},
        "tips":{
            "best_practices": [
                "Keep learning_rate low (0.05-0.1) and n_estimators high.",
                "Use subsample < 1.0 (stochastic GB) to reduce overfitting.",
                "Monitor train vs val score — stop early if gap widens.",
                "Tune max_depth (3-5) before tuning other params.",
            ],
            "pitfalls": [
                "Prone to overfitting with high n_estimators + high learning_rate.",
                "Much slower to train than Random Forest.",
                "Sensitive to outliers in the target variable.",
            ]
        },
        "val_curve_param": {
            "name": "n_estimators",
            "label": "n_estimators",
            "range": [10, 25, 50, 75, 100, 150, 200],
            "log_scale": False
        }
    },
    "naive_bayes": {
        "label": "Naive Bayes",
        "description": (
            "Applies Bayes' theorem with the 'naive' assumption that features are conditionally "
            "independent. Extremely fast and works well with small data. "
            "GaussianNB assumes features follow a normal distribution."
        ),
        "sklearn_cls": GaussianNB,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html",
        "supports_predict_proba": True,
        "supports_feature_importance": False,
        "supports_coef": False,
        "params": {
            "var_smoothing": {
                "type": "slider_float",
                "label": "Var Smoothing (log₁₀)",
                "default": -9.0,
                "min": -12.0,
                "max": -1.0,
                "step": 0.5,
                "help": "log₁₀ of variance smoothing (added to variance for numerical stability). "
                        "Actual value = 10^x. Default 1e-9.",
            },
        },
        "fixed_params": {},
        "tips":{
            "best_practices": [
                "Works surprisingly well on text and small datasets.",
                "Increase var_smoothing if features have very small variance.",
                "Great as a fast baseline before trying complex models.",
                "Gaussian NB assumes normally distributed continuous features.",
            ],
            "pitfalls": [
                "The 'naive' independence assumption is rarely true in practice.",
                "Poor calibrated probabilities — use with caution for probability outputs.",
                "Sensitive to highly correlated features.",
            ]
        },
        "val_curve_param": {
            "name": "clf__var_smoothing",
            "label": "var_smoothing",
            "range": np.logspace(-10, -1, 10).tolist(),
            "log_scale": True
        }
    },
    "adaboost":{
        "label": "Ada Boost",
        "description": (
            "Sequentially trains weak learners (stumps by default), each focusing on the "
            "previously misclassified samples by upweighting them. "
            "Robust to overfitting in practice; sensitive to noisy data."
        ),
        "sklearn_cls": AdaBoostClassifier,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
        "supports_predict_proba": True,
        "supports_feature_importance": True,
        "supports_coef": False,
        "params":{
            "n_estimators":{
                "type": "slider_int",
                "label": "Number of estimators",
                "default": 50,
                "min": 10,
                "max": 300,
                "step": 10,
                "help": "Number of weak learners to chain together"
            },
            "learning_rate":{
                "type": "slider_float",
                "label": "Learning Rate",
                "default": 1.0,
                "min": 0.01,
                "max": 2.0,
                "step": 0.05,
                "help": "Shrinks each estimator's contribution. Trade-off with n_estimators."
            },
        },
        "fixed_params": {"random_state": 42},
        "tips":{
            "best_practices": [
                "Use shallow Decision Trees (max_depth=1–2) as base estimators.",
                "Lower learning_rate requires more n_estimators to compensate.",
                "Monitor training error — AdaBoost should drive it to near zero.",
                "Works well on binary tasks; multiclass support is limited.",
            ],
            "pitfalls": [
                "Very sensitive to noisy data and outliers (hard labels).",
                "Can overfit if n_estimators is too high relative to dataset size.",
                "Slower than Random Forest at inference due to sequential nature.",
            ]
        },
        "val_curve_param":{
            "name": "n_estimators",
            "label": "n_estimators",
            "range": [10, 25, 50, 75, 100, 150, 200],
            "log_scale": False
        }
    },
    "voting_classifier":{
        "label": "Voting Classifier",
        "description": (
            "Combines multiple diverse models and aggregates their predictions. "
            "'Hard' voting uses majority class labels; 'soft' voting averages predicted "
            "probabilities (often better when models are well-calibrated)."
        ),
        "sklearn_cls":  VotingClassifier,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html",
        "supports_predict_proba": True, # only when voting='Soft'
        "supports_feature_importance": False,
        "supports_coef": False,
        "params":{
            "voting": {
                "type": "selectbox",
                "label": "Voting Strategy",
                "default": "soft",
                "options": ["soft", "hard"],
                "help": "'soft': average probabilities (needs predict_proba). 'hard': majority class vote."
            },
            "include_lr": {
                "type": "toggle",
                "label": "Include Logistic Regression",
                "default": True,
                "help": "Add Logistic Regression as a base estimator."
            },
            "include_knn": {
                "type": "toggle",
                "label": "Include KNN",
                "default": True,
                "help": "Add K-Nearest Neghbours (k=5) as a base estimator"
            },
            "include_tree": {
                "type": "toggle",
                "label": "Include Descision Tree",
                "default": True,
                "help": "Add a Descion Tree (depth=5) as a base estimator."
            },
        },
        "fixed_params": {},
        "tips":{
            "best_practices": [
                "Combine diverse, well-tuned base models for best results.",
                "Use soft voting when all base models support predict_proba.",
                "Hard voting is more robust when model probability estimates are poor.",
                "Ensure base models are trained on the same feature space.",
            ],
            "pitfalls": [
                "Correlated base models add little diversity — pick varied algorithms.",
                "Soft voting can be dominated by overconfident models.",
                "Not easily interpretable — treat as a final ensemble, not for exploration.",
            ]
        },
        "val_curve_param": None
    },
    "LDA":{
        "label": "Linear Discriminant Analysis",
        "description": (
            "Projects data onto axes that maximize class separability. "
            "Assumes Gaussian class-conditional distributions with equal covariance. "
            "Also useful as a dimensionality reduction technique."
        ),
        "sklearn_cls": LinearDiscriminantAnalysis,
        "docs_url": "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html",
        "supports_predict_proba": True,
        "supports_feature_importance": False,
        "supports_coef": True,
        "params":{
            "solver": {
                "type": "selectbox",
                "label": "Solver",
                "default": "svd",
                "options": ["svd", "lsqr", "eigen"],
                "help": "'svd': no covariance matrix, best for many features. 'lsqr' and 'eigen': support shrinkage."
            },
            "shrinkage": {
                "type": "selectbox",
                "label": "Shrinkage",
                "default": "none",
                "options": ["none", "auto"],
                "help": "Covariance shrinkage. 'auto' uses Ledoit-Wolf lemma. Only valid with lsqr/eigen solver."
            }
        },
        "fixed_params": {},
        "tips":{
            "best_practices": [
                "Excellent when classes are roughly Gaussian with equal covariance.",
                "Use shrinkage='auto' with solver='lsqr'/'eigen' for small n.",
                "Also useful as a dimensionality reduction step before other models.",
                "Fast and interpretable — good baseline for linearly separable data.",
            ],
            "pitfalls": [
                "Assumes equal class covariance matrices — violated in many real datasets.",
                "Performs poorly on highly non-linear decision boundaries.",
                "Sensitive to outliers that distort the class means.",
            ]
        },
        "val_curve_param": {
            "name": "clf__solver",
            "label": "solver",
            "range": ["svd", "lsqr", "eigen"],
            "log_scale": False
        }
    }
}
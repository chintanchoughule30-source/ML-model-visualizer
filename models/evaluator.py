"""
Training & evaluation engine.
 
Handles:
  - Model fitting with timing
  - Train/test metrics: accuracy, F1, precision, recall
  - Confusion matrix
  - ROC-AUC (binary) and per-class OvR (multiclass)
  - Predict-proba for probabilistic models
 
Returns a structured EvalResult dict so Streamlit pages just render, not compute.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score, 
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
import streamlit as st


@dataclass
class EvalResult:
    train_time_ms:float = 0.0
    # Predictions
    y_train_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_test_pred:  np.ndarray = field(default_factory=lambda: np.array([]))
    y_train_proba: np.ndarray | None = None
    y_test_proba:  np.ndarray | None = None
 
    # Train metrics
    train_accuracy:  float = 0.0
    train_f1:        float = 0.0
    train_precision: float = 0.0
    train_recall:    float = 0.0
 
    # Test metrics
    test_accuracy:  float = 0.0
    test_f1:        float = 0.0
    test_precision: float = 0.0
    test_recall:    float = 0.0
 
    # Confusion matrix
    conf_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
 
    # ROC
    roc_auc: float | None = None
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
 
    # Error message if fitting failed
    error: str | None = None

def train_and_evaluate(model, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, class_names: list[str]|None = None) -> EvalResult:
    """
    Fit the model on X_train, evaluate on both splits.
 
    Args:
        model:       A fitted-ready sklearn estimator or Pipeline from builder.build_model()
        X_train/test: Feature arrays
        y_train/test: Label arrays
        class_names:  For multi-class averaging labels (optional)
 
    Returns:
        EvalResult dataclass with all computed metrics.
    """
    result = EvalResult()
    n_classes = len(np.unique(y_train))
    avg = "binary" if n_classes == 2 else "weighted"

    #---Fit---#
    try:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        result.train_time_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        result.error = f"Model fitting failed: {e}"
        return result
    #---Predictions---#
    result.y_train_pred = model.predict(X_train)
    result.y_test_pred  = model.predict(X_test)
 
    # Probabilities (not all models support this)
    try:
        result.y_train_proba = model.predict_proba(X_train)
        result.y_test_proba  = model.predict_proba(X_test)
    except (AttributeError, NotImplementedError) as e:
        pass  # model doesn't support predict_proba
    except Exception as e:
        st.warning(f"predict_proba failed: {e}")  # surface the real error

    #---Metrics---#
    zero_div = 0  # avoid warnings on edge cases
 
    result.train_accuracy  = accuracy_score(y_train, result.y_train_pred)
    result.train_f1        = f1_score(y_train, result.y_train_pred, average=avg, zero_division=zero_div)
    result.train_precision = precision_score(y_train, result.y_train_pred, average=avg, zero_division=zero_div)
    result.train_recall    = recall_score(y_train, result.y_train_pred, average=avg, zero_division=zero_div)
 
    result.test_accuracy  = accuracy_score(y_test, result.y_test_pred)
    result.test_f1        = f1_score(y_test, result.y_test_pred, average=avg, zero_division=zero_div)
    result.test_precision = precision_score(y_test, result.y_test_pred, average=avg, zero_division=zero_div)
    result.test_recall    = recall_score(y_test, result.y_test_pred, average=avg, zero_division=zero_div)

    result.conf_matrix = confusion_matrix(y_test, result.y_test_pred)

    #---ROC-AUC---#
    if result.y_test_proba is not None:
        try:
            if n_classes == 2:
                result.fpr, result.tpr, _ = roc_curve(y_test, result.y_test_proba[:, 1])
                result.roc_auc = roc_auc_score(y_test, result.y_test_proba[:, 1])
            else:
                # One-vs-Rest for multiclass
                classes = np.unique(y_train)
                y_bin = label_binarize(y_test, classes=classes)
                result.roc_auc = roc_auc_score(
                    y_bin, result.y_test_proba, average="weighted", multi_class="ovr"
                )
                # Store per-class curves as a dict for plotting
                fpr_dict, tpr_dict = {}, {}
                for i, cls in enumerate(classes):
                    fpr_dict[cls], tpr_dict[cls], _ = roc_curve(y_bin[:, i], result.y_test_proba[:, i])
                result.fpr = fpr_dict
                result.tpr = tpr_dict
        except Exception:
            pass   # ROC not critical; skip silently
 
    return result
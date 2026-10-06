"""
Phase 3 + 4 + 5 — Model Training Page
Sidebar: model selector + dynamic hyperparameter widgets
Main:    metrics, confusion matrix, ROC curve
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from models.registry import MODEL_REGISTRY
from models.builder import build_model
from models.evaluator import train_and_evaluate, EvalResult
from utils.boundary_plot import build_boundary_figure
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from utils.insights import compute_learning_curve, compute_validation_curve
from utils.code_export import generate_export_code

st.title("⚙ Model Training")
st.caption("Select a model, tune its hyperparameters, and evaluate performance.")

if not st.session_state.get("dataset_ready"):
    st.warning("⚠️ No dataset loaded. Please visit the **Dataset** page first.")
    st.stop()

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
feature_names = st.session_state["feature_names"]
class_names = st.session_state.get("class_names")
# --- Detect dataset change and clear stale results ---
current_dataset = st.session_state.get("dataset_key_export")  # or whatever key you use

if "last_model_dataset" not in st.session_state:
    st.session_state.last_model_dataset = current_dataset

if st.session_state.last_model_dataset != current_dataset:
    st.session_state.last_model_dataset = current_dataset
    # Clear stale training artifacts
    for key in ["eval_result", "trained_model", "class_names"]:  # adjust key names to match yours
        st.session_state.pop(key, None)

#------Sidebar------#
with st.sidebar:
    st.header("Model Controls")

    all_model_keys = [key for key in MODEL_REGISTRY.keys()]
    def model_format(key):
        label = MODEL_REGISTRY[key]["label"]
        return label
    
    def on_model_change():
        # Wipe trained model so boundary section shows the "train first" message
        st.session_state.pop("trained_model", None)
        st.session_state.pop("training_done", None)
        st.session_state.pop("eval_result", None)
        st.session_state.pop("train_metrics", None)
        st.session_state.pop("test_metrics", None)
        st.session_state.pop("confusion_matrix", None)
        st.session_state.pop("roc_data", None)
        st.session_state.pop("pr_data", None)

    model_key = st.selectbox(
        "Model",
        options=all_model_keys,
        format_func=model_format,
        on_change=on_model_change,
        help="Select a classification algorithm to train."
    )

    entry = MODEL_REGISTRY[model_key]
    st.caption(entry["description"])
    st.markdown(f"[[sklearn docs]]({entry['docs_url']})", unsafe_allow_html=False)
    st.divider()

    #------Dynamic hyperparameter widgets------#
    st.subheader("⚙️ Hyperparameters")
    if st.button("Reset to defaults", width="stretch"):
        st.session_state[f"_reset_{model_key}"] = True
        st.rerun()
    # If reset flag is set, overwrite widget keys with defaults BEFORE widgets render
    if st.session_state.pop(f"_reset_{model_key}", False):
        for pk, spec in entry["params"].items():
            wkey  = f"{model_key}_{pk}"
            ptype = spec["type"]
            if ptype == "slider_float":
                st.session_state[wkey] = float(spec["default"])
            elif ptype == "slider_int":
                st.session_state[wkey] = int(spec["default"])
            else:
                st.session_state[wkey] = spec["default"]

    user_params = {}

    for param_key, spec in entry["params"].items():
        ptype = spec["type"]
        label = spec["label"]
        help = spec.get("help", "")
        wkey = f"{model_key}_{param_key}"
        if wkey in st.session_state:
            default = st.session_state[wkey]
        else:
            default = spec["default"]

        if ptype == "slider_float":
            val = st.slider(
                label,
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(default),
                step=float(spec["step"]),
                help=help,
                key=wkey,
            )
        elif ptype == "slider_int":
            val = st.slider(
                label,
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                value=int(default),
                step=int(spec["step"]),
                help=help,
                key=wkey,
            )
        elif ptype == "selectbox":
            options = spec["options"]
            idx = options.index(default) if default in options else 0
            val = st.selectbox(
                label,
                options=options,
                index=idx,
                help=help,
                key=wkey,
            )
        elif ptype == "toggle":
            val = st.toggle(
                label, 
                value=bool(default),
                help=help,
                key=wkey
            )
        else: val = default
        user_params[param_key] = val

#------Build and Train------#
try:
    clf = build_model(model_key, user_params)
except Exception as e:
    st.error(f"Could not build model: {e}")
    st.stop()

#---Train button---#
col_btn, col_status = st.columns([1, 3])
with col_btn:
    train_clicked = st.button("Train Model", type="primary", width="stretch")

params_fingerprint = str(model_key) + str(user_params)
if train_clicked or st.session_state.get("last_fingerprint") != params_fingerprint:
    if train_clicked:
        st.session_state["last_fingerprint"] = params_fingerprint
        with st.spinner("Training..."):
            result:EvalResult = train_and_evaluate(clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, class_names=class_names)
        # Store trained model for visualization page
        st.session_state["trained_model"] = clf
        st.session_state["training_done"] = True
        st.session_state["eval_result"] = result
        st.session_state["model_key"] = model_key
    elif "eval_result" not in st.session_state:
        with col_status:
            st.info("Configure hyperparameters and click **Train Model** to begin.")
        st.stop()

result = st.session_state.get("eval_result")
if result is not None and result.error:
    st.error(f"Training Error: {result.error}")
    st.stop()


#------Metrics Row------#
st.subheader("📊 Evaluation Metrics")

col_train, col_test = st.columns(2)

if result is None:
    st.info("Train a model to see evaluation results.", icon="📊")
else:    
    with col_train:
        st.markdown("**Train Set**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result.train_accuracy:.3f}")
        m2.metric("F1 Score", f"{result.train_f1:.3f}")
        m3.metric("Precision", f"{result.train_precision:.3f}")
        m4.metric("Recall", f"{result.train_recall:.3f}")

    with col_test:
        st.markdown("**Test Set**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result.test_accuracy:.3f}", delta=f"{result.test_accuracy - result.train_accuracy:+.3f}")
        m2.metric("F1 Score", f"{result.test_f1:.3f}", delta=f"{result.test_f1 - result.train_f1:+.3f}")
        m3.metric("Precision", f"{result.test_precision:.3f}", delta=f"{result.test_precision - result.train_precision:+.3f}")
        m4.metric("Recall", f"{result.test_recall:.3f}", delta=f"{result.test_recall - result.train_recall:+.3f}")
    st.caption(f"⏱️ Training time: **{result.train_time_ms:.1f} ms**")
st.divider()

#------Confusion Matrix + ROC + PR curve------#
col_cm, col_roc = st.columns(2)

if result is not None:
    with col_cm:
        st.subheader("Confusion Matrix")
        if "eval_result" in st.session_state:
            cm = result.conf_matrix
            display_labels = class_names if class_names is not None else [f"Class{i}" for i in range(cm.shape[0])]
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=display_labels,
                y=display_labels,
                colorscale="Blues",
                showscale=True,
                reversescale=False,
            )
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                yaxis_autorange="reversed",
                height=360,
                margin=dict(l=20, r=20, t=30, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_cm, width="stretch")

    with col_roc:
        st.subheader("ROC Curve")
        if result.roc_auc is not None and result.fpr is not None:
            fig_roc = go.Figure()

            # Diagonal baseline
            fig_roc.add_trace(
                go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                name="Random (AUC = 0.50)",
                showlegend=True,
            ))

            colors = px.colors.qualitative.Set2

            if isinstance(result.fpr, dict):
                # Multiclass: one curve per class
                for i, cls in enumerate(result.fpr.keys()):
                    label = class_names[cls] if class_names is not None and cls < len(class_names) else f"Class {cls}"
                    fig_roc.add_trace(go.Scatter(
                        x=result.fpr[cls],
                        y=result.tpr[cls],
                        mode="lines",
                        name=label,
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            else:
                # Binary
                fig_roc.add_trace(
                    go.Scatter(
                    x=result.fpr,
                    y=result.tpr,
                    mode="lines",
                    name=f"ROC (AUC = {result.roc_auc:.3f})",
                    line=dict(color=colors[0], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(102,194,165,0.15)",
                ))

            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=360,
                margin=dict(l=20, r=20, t=30, b=60),
                template="plotly_white",
                legend=dict(x=0.55, y=0.1),
            )
            if result.roc_auc:
                st.plotly_chart(fig_roc, width="stretch")
                st.caption(f"Weighted ROC-AUC: **{result.roc_auc:.4f}**")
        else:
            st.info("ROC curve not available for this model/dataset combination.")
    
    st.subheader("Precision-Recall Curve")
    
    if result.y_test_proba is not None:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
    
        colors = px.colors.qualitative.Set2
        n_classes = len(np.unique(y_test))
    
        fig_pr = go.Figure()
    
        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y_test, result.y_test_proba[:, 1])
            ap = average_precision_score(y_test, result.y_test_proba[:, 1])
            baseline = y_test.sum() / len(y_test)   # positive class ratio
    
            # Baseline (random classifier)
            fig_pr.add_trace(go.Scatter(
                x=[0, 1], y=[baseline, baseline],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                name=f"Random (AP = {baseline:.2f})",
            ))
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=f"PR curve (AP = {ap:.3f})",
                line=dict(color=colors[0], width=2),
                fill="tozeroy",
                fillcolor="rgba(102,194,165,0.15)",
            ))
            st.caption(f"Average Precision: **{ap:.4f}**")
    
        else:
            # One-vs-Rest per class
            classes = np.unique(y_train)
            y_bin = label_binarize(y_test, classes=classes)
            ap_scores = []
    
            for i, cls in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], result.y_test_proba[:, i])
                ap = average_precision_score(y_bin[:, i], result.y_test_proba[:, i])
                ap_scores.append(ap)
                label = class_names[cls] if class_names is not None and cls < len(class_names) else f"Class {cls}"
    
                fig_pr.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode="lines",
                    name=f"{label} (AP = {ap:.3f})",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
    
            mean_ap = np.mean(ap_scores)
            st.caption(f"Mean Average Precision (mAP): **{mean_ap:.4f}**")
    
        fig_pr.update_layout(
            xaxis=dict(title="Recall", range=[0, 1]),
            yaxis=dict(title="Precision", range=[0, 1.05]),
            height=380,
            margin=dict(l=20, r=20, t=20, b=60),
            template="plotly_white",
            legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
        )
        st.plotly_chart(fig_pr, width="stretch")
    
    else:
        st.info("Precision-Recall curve requires `predict_proba` — not available for this model.")

st.divider()

#------Feature Importance------#
st.subheader("🔍 Model Insights")

tab_fi, tab_lc, tab_vc, tab_tips = st.tabs([
    "📊 Feature Importance / Coefficients",
    "📈 Learning Curve",
    "🎛️ Validation Curve",
    "💡 Tips & Pitfalls"
])

with tab_fi:
    # Extract the actual estimator from Pipeline if wrapped
    actual_clf = st.session_state.get("trained_model", clf)
    if hasattr(actual_clf, "named_steps"):
        actual_clf = actual_clf.named_steps.get("clf", actual_clf)
    
    shown_insight = False
    
    # Feature importance (tree-based models)
    if entry.get("supports_feature_importance") and hasattr(actual_clf, "feature_importances_"):
        importances = actual_clf.feature_importances_
        feat_labels = feature_names if len(feature_names) == len(importances) else [f"F{i}" for i in range(len(importances))]

        sorted_idx = np.argsort(importances)[::-1]
        fig_fi = go.Figure(go.Bar(
            x=[feat_labels[i] for i in sorted_idx],
            y=importances[sorted_idx],
            marker_color="steelblue",
        ))
        fig_fi.update_layout(
            title="Feature Importances",
            xaxis_title="Feature",
            yaxis_title="Importance",
            height=300,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_fi, width="stretch")
        shown_insight = True
    
    # Coefficients (LR, LDA)
    elif entry.get("supports_coef") and hasattr(actual_clf, "coef_"):
        coef = actual_clf.coef_
        feat_labels = feature_names if len(feature_names) == coef.shape[1] else [f"F{i}" for i in range(coef.shape[1])]
    
        if coef.shape[0] == 1:
            # Binary
            fig_coef = go.Figure(go.Bar(
                x=feat_labels,
                y=coef[0],
                marker_color=["steelblue" if v > 0 else "tomato" for v in coef[0]],
            ))
        else:
            # Multi-class: grouped bars
            import plotly.express as px
            colors = px.colors.qualitative.Set2
            fig_coef = go.Figure()
            for i, row in enumerate(coef):
                label = class_names[i] if class_names is not None and i < len(class_names) else f"Class {i}"
                fig_coef.add_trace(go.Bar(
                    x=feat_labels, y=row, name=label,
                    marker_color=colors[i % len(colors)],
                ))
            fig_coef.update_layout(barmode="group")
    
        fig_coef.update_layout(
            title="Model Coefficients",
            xaxis_title="Feature",
            yaxis_title="Coefficient value",
            height=300,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_coef, width="stretch")
        shown_insight = True
    
    if not shown_insight:
        st.info("No feature importance or coefficient visualization available for this model.")

with tab_lc:
    st.markdown("**Learning Curve** — how train and cross-validation accuracy change as more training data is added.")
    lc_cv = st.slider("Cross-validation folds", min_value=2, max_value=10, value=5, key="lc_cv")

    if st.button("Compute Learning Curve", key="btn_lc"):
        with st.spinner("Computing learning curve..."):
            try:
                train_sizes, tr_mean, tr_std, val_mean, val_std = compute_learning_curve(clf, X_train, y_train, cv=lc_cv)
                fig_lc = go.Figure()
                # Train band
                fig_lc.add_trace(go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate([tr_mean + tr_std, (tr_mean - tr_std)[::-1]]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                ))
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=tr_mean,
                    mode="lines+markers", name="Train score",
                    line=dict(color="rgba(99,110,250,1)", width=2),
                ))

                # Val band
                fig_lc.add_trace(go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                    fill="toself", fillcolor="rgba(239,85,59,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                ))
                fig_lc.add_trace(go.Scatter(
                    x=train_sizes, y=val_mean,
                    mode="lines+markers", name="Val score",
                    line=dict(color="rgba(239,85,59,1)", width=2),
                ))

                fig_lc.update_layout(
                    xaxis_title="Training samples",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1.05]),
                    height=360, template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=60),
                    legend=dict(x=0.75, y=0.05),
                )
                st.plotly_chart(fig_lc, width="stretch")

                # Bias-variance interpretation
                gap = float(tr_mean[-1] - val_mean[-1])
                train_score_final = float(tr_mean[-1])

                if train_score_final < 0.75:
                    verdict = "🔴 **High Bias (Underfitting)** — Both train and val scores are low. The model is too simple."
                    suggestion = "Try a more complex model, reduce regularization, or add more features."
                elif gap > 0.10:
                    verdict = "🟡 **High Variance (Overfitting)** — Train score is much higher than val score."
                    suggestion = "Increase regularization, reduce model complexity, or gather more training data."
                else:
                    verdict = "🟢 **Well Fitted** — Train and val scores are close and both high."
                    suggestion = "The model generalizes well to unseen data."

                st.info(f"{verdict}\n\n{suggestion}")

            except Exception as e:
                st.error(f"Learning curve failed: {e}")
    else:
        st.caption("Click **Compute Learning Curve** to generate the plot. This may take a few seconds.")

with tab_vc:
    st.markdown("**Validation Curve** — how train and val score change as one hyperparameter is swept.")

    vc_config = entry.get("val_curve_param")

    if vc_config is None:
        st.info("Validation curve not configured for this model.")
    else:
        st.caption(f"Sweeping **{vc_config['label']}** over {len(vc_config['range'])} values.")
        vc_cv = st.slider("CV folds", min_value=2, max_value=10, value=5, key="vc_cv")

        if st.button("Compute Validation Curve", key="btn_vc"):
            with st.spinner(f"Sweeping {vc_config['label']}..."):
                try:
                    param_range, tr_mean, tr_std, val_mean, val_std = compute_validation_curve(
                        clf,
                        X_train, y_train,
                        param_name=vc_config["name"],
                        param_range=vc_config["range"],
                        cv=vc_cv,
                    )

                    x_vals = [str(round(v, 6)) if isinstance(v, float) else str(v) for v in param_range]

                    fig_vc = go.Figure()

                    # Train band
                    fig_vc.add_trace(go.Scatter(
                        x=x_vals + x_vals[::-1],
                        y=list(tr_mean + tr_std) + list((tr_mean - tr_std)[::-1]),
                        fill="toself", fillcolor="rgba(99,110,250,0.15)",
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                    ))
                    fig_vc.add_trace(go.Scatter(
                        x=x_vals, y=tr_mean,
                        mode="lines+markers", name="Train score",
                        line=dict(color="rgba(99,110,250,1)", width=2),
                    ))

                    # Val band
                    fig_vc.add_trace(go.Scatter(
                        x=x_vals + x_vals[::-1],
                        y=list(val_mean + val_std) + list((val_mean - val_std)[::-1]),
                        fill="toself", fillcolor="rgba(239,85,59,0.15)",
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                    ))
                    fig_vc.add_trace(go.Scatter(
                        x=x_vals, y=val_mean,
                        mode="lines+markers", name="Val score",
                        line=dict(color="rgba(239,85,59,1)", width=2),
                    ))

                    fig_vc.update_layout(
                        xaxis_title=vc_config["label"],
                        yaxis_title="Accuracy",
                        yaxis=dict(range=[0, 1.05]),
                        height=360, template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=60),
                        legend=dict(x=0.75, y=0.05),
                    )

                    if vc_config.get("log_scale"):
                        fig_vc.update_xaxes(type="category")  # already stringified log values

                    st.plotly_chart(fig_vc, width="stretch")

                    # Optimal value
                    best_idx = int(np.argmax(val_mean))
                    st.success(f"Best val score **{val_mean[best_idx]:.4f}** at `{vc_config['label']} = {param_range[best_idx]}`")

                except Exception as e:
                    st.error(f"Validation curve failed: {e}")
        else:
            st.caption("Click **Compute Validation Curve** to generate the plot.")

with tab_tips:
    tips = entry.get("tips")
    if tips:
        col_bp, col_pf = st.columns(2)
        with col_bp:
            st.markdown("#### Best Practices")
            for tip in tips["best_practices"]:
                st.markdown(f"- {tip}")
        with col_pf:
            st.markdown("#### Common Pitfalls")
            for pitfall in tips["pitfalls"]:
                st.markdown(f"- {pitfall}")
    else:
        st.info("No tips available for this model yet.")

#------Decision Boundary------#
X_train      = st.session_state.get("X_train")
X_test       = st.session_state.get("X_test")
y_train      = st.session_state.get("y_train")
y_test       = st.session_state.get("y_test")
boundary_n_classes = len(np.unique(y_train)) if y_train is not None else 0
feature_names = st.session_state.get("feature_names", ["Feature 0", "Feature 1"])
class_names   = st.session_state.get("class_names", None)
trained_model = st.session_state.get("trained_model", None)
feat_x = st.session_state.get("viz_feat_idx_0", 0)
feat_y = st.session_state.get("viz_feat_idx_1", 1)
n_features = X_train.shape[1]
# Clamp to valid range — guards against stale indices from a previous dataset
feat_x = min(feat_x, n_features - 1)
feat_y = min(feat_y, n_features - 1)

# Edge case: if both clamped to same index, force feat_y to next one
if feat_x == feat_y:
    feat_y = min(feat_x + 1, n_features - 1)
    feat_x = max(0, feat_y - 1)  # ensure they're different

X_train_2d = X_train[:, [feat_x, feat_y]]
X_test_2d  = X_test[:, [feat_x, feat_y]]
viz_feature_names = [feature_names[feat_x], feature_names[feat_y]]

st.divider()
st.subheader("🗺️ Decision Boundary")

# Check if current trained model is an SVM
def _is_svm(model):
    if isinstance(model, Pipeline):
        return any(isinstance(step, SVC) for _, step in model.steps)
    return isinstance(model, SVC)

if trained_model is None:
    st.info("Train a model above to see the decision boundary.")

elif X_train is not None:
    has_proba = hasattr(trained_model, "predict_proba") and boundary_n_classes == 2
    use_proba = False
    show_support_vectors = False

    col_toggles = st.columns(2)
    with col_toggles[0]:
        if has_proba:
            use_proba = st.toggle(
                "Probability shading",
                value=False,
                help="Overlays soft probability gradient (binary tasks only)"
            )
    with col_toggles[1]:
        if _is_svm(trained_model):
            show_support_vectors = st.toggle(
                "Show support vectors",
                value=False,
                help="Highlights support vectors as open circle markers"
            )

    col1, col2 = st.columns(2)
    with col1:
        fig_train = build_boundary_figure(
            trained_model, X_train_2d, y_train,
            split_label="Train",
            feature_names=viz_feature_names,
            class_names=class_names,
            use_proba=use_proba,
            show_support_vectors=show_support_vectors,
            X_full=X_train,
            feat_idx_0=feat_x,
            feat_idx_1=feat_y
        )
        st.plotly_chart(fig_train, width="stretch")

    with col2:
        fig_test = build_boundary_figure(
            trained_model, X_test_2d, y_test,
            split_label="Test",
            feature_names=viz_feature_names,
            class_names=class_names,
            use_proba=use_proba,
            show_support_vectors=show_support_vectors,
            X_full=X_test,
            feat_idx_0=feat_x,
            feat_idx_1=feat_y,
        )
        st.plotly_chart(fig_test, width="stretch")

    if X_train.shape[1] > 2:
        st.caption(
            f"⚠️ Model trained on **{X_train.shape[1]} features**. This plot shows a 2D slice "
            f"through **{viz_feature_names[0]}** vs **{viz_feature_names[1]}** — all other features "
            f"are fixed at their **median**. Points may fall in the 'wrong' region because the "
            f"true decision boundary lives in {X_train.shape[1]}D space. "
            f"For an accurate boundary, use a synthetic dataset or select 2 features on a 2-feature dataset."
        )

st.divider()
st.subheader("Export Code")
st.caption("Get a self-contained Python script that reproduces your exact setup.")

if not st.session_state.get("training_done"):
    st.info("Train a model first to generate the export snippet.")
else:
    dataset_source  = st.session_state.get("dataset_source", "Synthetic")
    dataset_key_exp = st.session_state.get("dataset_key_export", "moons")
    n_samples_exp   = st.session_state.get("n_samples_export", 300)
    noise_exp       = st.session_state.get("noise_export", 0.1)
    seed_exp        = st.session_state.get("random_seed_export", 42)
    test_split_exp  = st.session_state.get("test_split_export", 0.2)

    # Generate the code
    code_str = generate_export_code(
        model_key=model_key,
        user_params=user_params,
        dataset_source=dataset_source,
        dataset_key=dataset_key_exp,
        n_samples=n_samples_exp,
        noise=noise_exp,
        random_seed=seed_exp,
        test_split=test_split_exp,
    )

    # Display with syntax highlighting
    st.code(code_str, language="python")
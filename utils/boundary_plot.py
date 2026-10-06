import numpy as np
import plotly.graph_objects as go
import plotly.subplots as make_subplots
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# background (decision regions)
# choose color scales accroding to the number of classes
COLOR_SCALES = {
    2: [[0, "#4361ee"], [1, "#f72585"]],
    3: [[0, "#4361ee"], [0.5, "#f72585"], [1, "#4cc9f0"]],
    4: [[0, "#4361ee"], [0.33, "#f72585"], [0.66, "#4cc9f0"], [1, "#f8961e"]],
}
# scatter points (actual data)
CLASS_COLORS = ["#4361ee", "#f72585", "#4cc9f0", "#f8961e", "#7209b7"]

def make_meshgrid(X, resolution=200, padding=0.5):
    """
    Creates a Mesh grid because instead of predicting only on data points, we predict on every pixel in space.
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    return xx, yy

def get_boundary_values(model, xx, yy):
    """Returns Z for contour — predicted class index."""
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    label_to_idx = {c: i for i, c in enumerate(model.classes_)}
    Z = np.array([label_to_idx[z] for z in Z])
    return Z.reshape(xx.shape)

def get_proba_values(model, xx, yy, class_idx=0):
    """Returns probability for one class — for soft shading."""
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid)[:, class_idx]
    return proba.reshape(xx.shape)

def get_boundary_values_2d_slice(model, xx, yy, X_full, feat_idx_0, feat_idx_1):
    """
    Builds a meshgrid prediction by fixing all non-visualized features
    at their median, and varying only the 2 selected features.
    Works for any number of features in the trained model.
    """
    n_features = X_full.shape[1]
    medians = np.median(X_full, axis=0)  # shape (n_features,)

    grid_points = xx.ravel().shape[0]
    # Start with all features at median
    grid = np.tile(medians, (grid_points, 1))  # shape (n_points, n_features)

    # Overwrite the 2 visualization features with meshgrid values
    grid[:, feat_idx_0] = xx.ravel()
    grid[:, feat_idx_1] = yy.ravel()

    Z = model.predict(grid)
    classes = model.classes_
    label_to_idx = {c: i for i, c in enumerate(classes)}
    Z = np.array([label_to_idx[z] for z in Z])
    return Z.reshape(xx.shape)


def get_proba_values_2d_slice(model, xx, yy, X_full, feat_idx_0, feat_idx_1, class_idx=0):
    """Same slice approach but for predict_proba."""
    n_features = X_full.shape[1]
    medians = np.median(X_full, axis=0)

    grid_points = xx.ravel().shape[0]
    grid = np.tile(medians, (grid_points, 1))
    grid[:, feat_idx_0] = xx.ravel()
    grid[:, feat_idx_1] = yy.ravel()

    proba = model.predict_proba(grid)[:, class_idx]
    return proba.reshape(xx.shape)

def get_support_vectors(model):
    """
    Extracts support vectors in original feature space.
    Handles both raw SVC and Pipeline(scaler → SVC).
    Returns array of shape (n_support_vectors, 2) or None.
    """

    if isinstance(model, Pipeline):
        # Find the SVC step
        svc = None
        scaler = None
        for name, step in model.steps:
            if isinstance(step, SVC):
                svc = step
            # grab the last transformer before SVC as the scaler
            elif hasattr(step, 'inverse_transform'):
                scaler = step

        if svc is None or not hasattr(svc, 'support_vectors_'):
            return None

        sv = svc.support_vectors_
        # Inverse transform back to original feature space
        if scaler is not None:
            sv = scaler.inverse_transform(sv)
        return sv

    elif isinstance(model, SVC):
        if hasattr(model, 'support_vectors_'):
            return model.support_vectors_
        
    return None

def build_boundary_figure(
    model, X_2d, y_true, split_label="Train",
    feature_names=None, class_names=None,
    use_proba=False,
    show_support_vectors=False,
    X_full=None,       # full feature matrix (all features)
    feat_idx_0=0,      # index of first viz feature in X_full
    feat_idx_1=1,      # index of second viz feature in X_full
):
    
    classes = model.classes_
    if class_names is None:
        class_names = [str(c) for c in classes]

    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Use X_full for meshgrid range if available, else X_2d
    X_for_range = X_2d
    xx, yy = make_meshgrid(X_for_range)

    # --- Get decision boundary Z values ---
    if X_full is not None and X_full.shape[1] > 2:
        Z = get_boundary_values_2d_slice(model, xx, yy, X_full, feat_idx_0, feat_idx_1)
    else:
        Z = get_boundary_values(model, xx, yy)

    if X_full is not None and X_full.shape[1] > 2:
        y_pred = model.predict(X_full)
    else:
        y_pred = model.predict(X_2d)
        
    correct_mask = y_true == y_pred

    fig = go.Figure()
    colorscale = COLOR_SCALES.get(n_classes, "RdBu")

    fig.add_trace(
        go.Contour(
            x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
            y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
            z=Z,
            colorscale=colorscale,
            opacity=0.35,
            showscale=False,
            contours=dict(coloring="fill", showlines=False),
            hoverinfo="skip",
            name="Boundary"
    ))

    #---Probability shading (binary only)---#
    if use_proba and hasattr(model, "predict_proba") and n_classes == 2:
        if X_full is not None and X_full.shape[1] > 2:
            Z_proba = get_proba_values_2d_slice(
                model, xx, yy, X_full, feat_idx_0, feat_idx_1, class_idx=1
            )
        else:
            Z_proba = get_proba_values(model, xx, yy, class_idx=1)

        fig.add_trace(go.Contour(
            x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
            y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
            z=Z_proba,
            colorscale=[[0, "rgba(67,97,238,0.0)"], [1, "rgba(247,37,133,0.35)"]],
            showscale=False,
            contours=dict(coloring="fill", showlines=False),
            hoverinfo="skip",
            name="Probability",
        ))

    #---Scatter points per class---#
    for i, cls in enumerate(classes):
        cls_mask = y_true == cls
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        name = class_names[i]

        correct = cls_mask & correct_mask
        if correct.any():
            fig.add_trace(go.Scatter(
                x=X_2d[correct, 0], y=X_2d[correct, 1],
                mode="markers",
                marker=dict(symbol="circle", color=color, size=8,
                            line=dict(width=1.5, color="white")),
                name=f"{name} ✓",
                legendgroup=name,
            ))

        wrong = cls_mask & ~correct_mask
        if wrong.any():
            fig.add_trace(go.Scatter(
                x=X_2d[wrong, 0], y=X_2d[wrong, 1],
                mode="markers",
                marker=dict(symbol="x", color=color, size=9,
                            line=dict(width=2, color="white")),
                name=f"{name} ✗",
                legendgroup=name,
            ))

    # --- Support vectors ---
    if show_support_vectors:
        sv = get_support_vectors(model)
        if sv is not None:
            # sv is in full feature space — extract just the 2 viz features
            sv_2d = sv[:, [feat_idx_0, feat_idx_1]]
            fig.add_trace(go.Scatter(
                x=sv_2d[:, 0], y=sv_2d[:, 1],
                mode="markers",
                marker=dict(symbol="circle-open", size=14, color="white",
                            line=dict(width=2, color="white"), opacity=0.85),
                name=f"Support Vectors ({len(sv)})",
                hovertemplate="Support Vector<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=f"{split_label} Set — Decision Boundary", font=dict(size=15)),
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(15,17,26,0.03)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=420,
    )
    return fig


def _make_full_grid_from_2d(X_2d, X_full, feat_idx_0, feat_idx_1):
    """For predict() on scatter points when model was trained on all features."""
    if X_full is None or X_full.shape[1] == 2:
        return X_2d
    medians = np.median(X_full, axis=0)
    grid = np.tile(medians, (X_2d.shape[0], 1))
    grid[:, feat_idx_0] = X_2d[:, 0]
    grid[:, feat_idx_1] = X_2d[:, 1]
    return grid
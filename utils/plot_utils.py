"""
Dataset preview panel utilities.
Renders: 2D scatter plot of features + class distribution bar chart.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# consistent color palette for classes
CLASS_COLORS = px.colors.qualitative.Set2

def plot_dataset(X:np.ndarray, y:np.ndarray, feature_names:list[str], class_names:list[str], title:str="Dataset Preview") -> go.Figure:
    """
    Build a side-by-side Plotly figure:
      Left:  2D scatter plot (first 2 features on axes, colored by class)
      Right: Class distribution bar chart
 
    Args:
        X:             Feature matrix (n_samples, n_features). Only first 2 cols used for scatter.
        y:             Integer class labels
        feature_names: Names for the feature axes. Defaults to ['Feature 0', 'Feature 1'].
        class_names:   Names for each class. Defaults to ['Class 0', 'Class 1', ...].
        title:         Figure title.
 
    Returns:
        Plotly Figure object (pass directly to st.plotly_chart)
    """
    n_classes = len(np.unique(y))
    if feature_names is None or len(feature_names) < 2:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Feature Space", "Class Distribution"],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.08
    )

    # ------Left: Scatter------
    for class_idx in np.unique(y):
        mask = y == class_idx
        color = CLASS_COLORS[class_idx % len(CLASS_COLORS)]
        label = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"

        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.8, line=dict(width=0.5, color="#0d0f14")),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"{feature_names[0]}: %{{x:.3f}}<br>"
                    f"{feature_names[1]}: %{{y:.3f}}<extra></extra>"
                )
            ),
            row=1,
            col=1
        )

    fig.update_xaxes(title_text=feature_names[0], row=1, col=1, showgrid=True, gridwidth=1)
    fig.update_yaxes(title_text=feature_names[1], row=1, col=1, showgrid=True, gridwidth=1)

        # ------Right: Bar chart------
    unique_classes, counts = np.unique(y, return_counts=True)
    bar_labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in unique_classes]
    bar_colors = [CLASS_COLORS[c % len(CLASS_COLORS)] for c in unique_classes]

    fig.add_trace(
        go.Bar(
            x=bar_labels,
            y=counts,
            marker_color = bar_colors,
            showlegend=False,
            text=counts,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2
    )

    fig.update_xaxes(title_text="Class", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2, showgrid=True, gridwidth=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)"
        ),
        height=420,
        margin=dict(l=40, r=40, t=70, b=40),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 15, 20, 1)",
    )
 
    return fig
import streamlit as st
import base64
import random
from pathlib import Path


def _image_to_base64(path: Path) -> str:
  with path.open("rb") as img_file:
    return base64.b64encode(img_file.read()).decode("utf-8")


_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_HERO_PREVIEW_OPTIONS = [
  {
    "file": "circles.png",
    "label": "Circles",
    "caption": "Concentric class geometry with a radial decision boundary.",
  },
  {
    "file": "linear.png",
    "label": "Linear",
    "caption": "Linearly separable classes where simple models shine.",
  },
  {
    "file": "xor.png",
    "label": "XOR",
    "caption": "Classic non-linear XOR pattern that defeats linear separators.",
  },
]

_hero_preview = random.choice(_HERO_PREVIEW_OPTIONS)
_HERO_PREVIEW_B64 = _image_to_base64(_ASSETS_DIR / _hero_preview["file"])
_HERO_PREVIEW_LABEL = _hero_preview["label"]
_HERO_PREVIEW_CAPTION = _hero_preview["caption"]

# ── Page-local CSS (supplements global styles in app.py) ─────────────────────
st.markdown("""
<style>
/* ── Eyebrow + headline ── */
.pg-eyebrow {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
    color: #4ade80; margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}
.pg-eyebrow-line { display: inline-block; width: 28px; height: 1px; background: #4ade80; }
.pg-headline {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 3rem; font-weight: 700; color: #ffffff;
    line-height: 1.1; letter-spacing: -0.03em;
    margin: 0 0 16px;
}
.pg-headline span { color: #4ade80; }
.pg-subhead {
    font-size: 14px; color: rgba(255,255,255,0.38);
    line-height: 1.75; max-width: 520px; margin-bottom: 28px;
}

/* ── Hero visual panel ── */
.pg-hero-visual {
  background: linear-gradient(160deg, rgba(74,222,128,0.06), rgba(13,15,20,0.92));
  border: 1px solid #1a1f2e;
  border-radius: 12px;
  padding: 12px;
}
.pg-hero-visual-head {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
}
.pg-hero-chip,
.pg-hero-chip-muted {
  font-family: 'JetBrains Mono','Courier New',monospace;
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 7px;
  border-radius: 999px;
}
.pg-hero-chip {
  color: rgba(74,222,128,0.8);
  border: 1px solid rgba(74,222,128,0.25);
  background: rgba(74,222,128,0.08);
}
.pg-hero-chip-muted {
  color: rgba(255,255,255,0.35);
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
.pg-hero-visual-img {
  width: 100%;
  border-radius: 10px;
  border: 1px solid #1a1f2e;
  display: block;
}
.pg-hero-visual-foot {
  margin-top: 9px;
  font-size: 11px;
  color: rgba(255,255,255,0.34);
  line-height: 1.55;
}

/* ── Section separator ── */
.pg-section-label {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 9px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #2d3550;
    display: flex; align-items: center; gap: 12px;
    margin: 0 0 20px;
}
.pg-section-rule { flex: 1; height: 1px; background: #1a1f2e; }

/* ── Step cards (3-column grid) ── */
.pg-step {
    background: #0d0f14;
    border: 1px solid #1a1f2e;
    border-radius: 10px;
    padding: 20px 22px 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.pg-step-num {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 11px; font-weight: 700;
    color: rgba(74,222,128,0.38); letter-spacing: 0.1em; margin-bottom: 2px;
}
.pg-step-title {
    font-size: 13px; font-weight: 600;
    color: rgba(255,255,255,0.88); margin-bottom: 4px;
}
.pg-step-body {
    font-size: 12px; color: rgba(255,255,255,0.28);
    line-height: 1.65; flex: 1;
}
.pg-tag-row { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 10px; }
.pg-tag {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 10px; padding: 2px 7px; border-radius: 4px;
    background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.28);
    border: 1px solid rgba(255,255,255,0.06); letter-spacing: 0.02em;
}
.pg-tag.g {
    background: rgba(74,222,128,0.07); color: rgba(74,222,128,0.6);
    border-color: rgba(74,222,128,0.14);
}
.pg-arrow { font-size: 13px; color: rgba(74,222,128,0.22); text-align: right; margin-top: 8px; }

/* ── Use-case blocks ── */
.pg-usecase-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }
.pg-usecase {
    background: #0d0f14; border: 1px solid #1a1f2e; border-radius: 10px;
    padding: 22px 20px; display: flex; flex-direction: column; gap: 10px;
}
.pg-usecase-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.14);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.pg-usecase-title {
    font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.85); margin-bottom: 2px;
}
.pg-usecase-desc { font-size: 12px; color: rgba(255,255,255,0.3); line-height: 1.65; }
.pg-usecase-tag {
    display: inline-block; margin-top: 4px;
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 9px; padding: 2px 8px; border-radius: 4px;
    background: rgba(74,222,128,0.06); color: rgba(74,222,128,0.5);
    border: 1px solid rgba(74,222,128,0.1); letter-spacing: 0.06em; text-transform: uppercase;
}
 
/* ── Gallery caption ── */
.pg-gallery-caption {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 10px; color: #aaaaaa; text-align: center;
    letter-spacing: 0.06em;
    text-align: center;
}
.pg-gallery-content {
    padding: 16px 16px 16px 24px;
}
.pg-gallery-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.75);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.pg-gallery-desc {
    font-size: 12px;
    color: rgba(255,255,255,0.35);
    line-height: 1.5;
}
            
/* ── Model table ── */
.pg-model-wrap { border: 1px solid #1a1f2e; border-radius: 10px; overflow: hidden; margin-bottom: 2rem; }
.pg-model-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.pg-model-table thead tr { background: #0d0f14; border-bottom: 1px solid #1e2330; }
.pg-model-table thead th {
    text-align: left; padding: 10px 16px;
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #2d3550; font-weight: 600;
}
.pg-model-table tbody tr { border-bottom: 1px solid #11141c; }
.pg-model-table tbody tr:last-child { border-bottom: none; }
.pg-model-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.pg-model-table tbody td { padding: 11px 16px; color: rgba(255,255,255,0.45); vertical-align: middle; }
.pg-model-table tbody td:first-child { color: rgba(255,255,255,0.82); font-weight: 500; }
.pg-family-tag {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    background: rgba(74,222,128,0.07); color: rgba(74,222,128,0.55);
    border: 1px solid rgba(74,222,128,0.12); white-space: nowrap;
}
.pg-reg {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 10px; color: rgba(255,255,255,0.22);
}

/* ── Expander (documentation) ── */
[data-testid="stExpander"] {
    background: #0d0f14 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 10px !important;
    margin-bottom: 0 !important;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono','Courier New',monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.35) !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover {
    color: rgba(74,222,128,0.7) !important;
}
[data-testid="stExpander"] summary svg {
    color: rgba(255,255,255,0.2) !important;
}
[data-testid="stExpanderDetails"] {
    border-top: 1px solid #1a1f2e !important;
    padding: 20px 18px !important;
}
[data-testid="stExpanderDetails"] p {
    font-size: 13px !important;
    color: rgba(255,255,255,0.55) !important;
    line-height: 1.65 !important;
}
[data-testid="stExpanderDetails"] strong {
    color: rgba(255,255,255,0.8) !important;
    font-weight: 600 !important;
}
[data-testid="stExpanderDetails"] small {
    color: rgba(255,255,255,0.3) !important;
}

/* ── Footer CTA cards ── */
.pg-footer-card {
    background: #0d0f14; border: 1px solid #1a1f2e; border-radius: 10px; padding: 18px 22px;
}
.pg-footer-label {
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #2d3550; margin-bottom: 8px;
}
.pg-footer-text { font-size: 13px; color: rgba(255,255,255,0.4); line-height: 1.65; }
.pg-footer-text strong { color: rgba(74,222,128,0.75); font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: logo header + status footer ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-header">
      <div class="sb-logo-row">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <polygon points="10,2 17,6 17,14 10,18 3,14 3,6"
                   fill="none" stroke="#4ade80" stroke-width="1.2"/>
          <polygon points="10,6 13.5,8 13.5,12 10,14 6.5,12 6.5,8"
                   fill="#4ade80" opacity="0.25"/>
        </svg>
        <span class="sb-brand">ML Playground</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([1.08, 0.92], gap="large")

with hero_left:
    st.markdown("""
    <div class="pg-eyebrow">
      <span class="pg-eyebrow-line"></span>
      Interactive ML Sandbox
    </div>
    <div class="pg-headline">Train without<br>writing <span>a line.</span></div>
    <p class="pg-subhead">
      Pick a dataset, configure your hyperparameters, and watch classical
      algorithms learn — live in the browser. No setup, no environment, no friction.
    </p>
    """, unsafe_allow_html=True)

    # CTA — uses real Streamlit button for the primary action
    if st.button("→  Pick a Dataset", width="content"):
        st.switch_page("pages/dataset.py")

with hero_right:
    st.markdown(f"""
    <div class="pg-hero-visual">
      <div class="pg-hero-visual-head">
        <span class="pg-hero-chip">Decision Boundary</span>
        <span class="pg-hero-chip-muted">{_HERO_PREVIEW_LABEL}</span>
      </div>
      <img class="pg-hero-visual-img" src="data:image/png;base64,{_HERO_PREVIEW_B64}" alt="{_HERO_PREVIEW_LABEL} decision boundary preview" />
      <div class="pg-hero-visual-foot">
        {_HERO_PREVIEW_CAPTION}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ── 3-step overview cards ─────────────────────────────────────────────────────
st.markdown("""
<div class="pg-section-label">
  How it works <span class="pg-section-rule"></span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="small")

with col1:
    st.markdown("""
    <div class="pg-step">
      <div class="pg-step-num">01 &middot;</div>
      <div class="pg-step-title">Pick a dataset</div>
      <div class="pg-step-body">Choose a synthetic shape or a real sklearn dataset. Tune noise and test-split ratio.</div>
      <div class="pg-tag-row">
        <span class="pg-tag g">Moons</span>
        <span class="pg-tag g">XOR</span>
        <span class="pg-tag">Iris</span>
        <span class="pg-tag">Wine</span>
      </div>
      <div class="pg-arrow">&rarr;</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="pg-step">
      <div class="pg-step-num">02 &middot;</div>
      <div class="pg-step-title">Select a model</div>
      <div class="pg-step-body">Configure every parameter with instant visual feedback on the decision boundary.</div>
      <div class="pg-tag-row">
        <span class="pg-tag g">kNN</span>
        <span class="pg-tag g">SVM</span>
        <span class="pg-tag">RF</span>
        <span class="pg-tag">Tree</span>
      </div>
      <div class="pg-arrow">&rarr;</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pg-step">
      <div class="pg-step-num">03 &middot;</div>
      <div class="pg-step-title">Evaluate results</div>
      <div class="pg-step-body">Inspect accuracy, confusion matrix, feature importances, and cross-val scores.</div>
      <div class="pg-tag-row">
        <span class="pg-tag g">Accuracy</span>
        <span class="pg-tag">Confusion</span>
        <span class="pg-tag">CV</span>
      </div>
      <div class="pg-arrow">&#8599;</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ── Gallery: decision boundary teasers ───────────────────────────────────────
st.markdown("""
<div class="pg-section-label">
  The gallery <span class="pg-section-rule"></span>
</div>
""", unsafe_allow_html=True)
 
g_col1, g_col2 = st.columns(2, gap="medium")
with g_col1:
    st.image("assets/moons.png", width="stretch")
    st.markdown("""
    <div class="pg-gallery-caption">
        <strong style='color: #4ade80; font-size: 16px;'>SVC</strong>
        <div class="pg-gallery-content">
            <div class="pg-gallery-label">Non-Linearity</div>
            <div class="pg-gallery-desc">Watch how SVM carves complex, circular boundaries around the 'Moons' dataset that linear models can't touch.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with g_col2:
    st.image("assets/blobs.png", width="stretch")
    st.markdown("""
    <div class="pg-gallery-caption">
        <strong style='color: #4ade80; font-size: 16px;'>KNN</strong>
        <div class="pg-gallery-content">
            <div class="pg-gallery-label">Instance-based</div>
            <div class="pg-gallery-desc">Flexible but sensitive to noise.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
 
 
# ── Why use this? ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-section-label">
  Why use this? <span class="pg-section-rule"></span>
</div>
<div class="pg-usecase-grid">
 
  <div class="pg-usecase">
    <div class="pg-usecase-icon">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <path d="M8 2L2 5v4c0 3 2.5 5 6 5s6-2 6-5V5L8 2z"
              stroke="#4ade80" stroke-width="1.2" stroke-linejoin="round"/>
        <path d="M5.5 8l2 2 3-3" stroke="#4ade80" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="pg-usecase-title">Education</div>
      <div class="pg-usecase-desc">
        See bias-variance tradeoff play out visually. Watch a Decision Tree overfit
        and a Random Forest smooth it out — no textbook needed.
      </div>
      <span class="pg-usecase-tag">Learn by doing</span>
    </div>
  </div>
 
  <div class="pg-usecase">
    <div class="pg-usecase-icon">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <rect x="2" y="2" width="5" height="5" rx="1"
              stroke="#4ade80" stroke-width="1.2"/>
        <rect x="9" y="2" width="5" height="5" rx="1"
              stroke="#4ade80" stroke-width="1.2"/>
        <rect x="2" y="9" width="5" height="5" rx="1"
              stroke="#4ade80" stroke-width="1.2"/>
        <path d="M9 11.5h5M11.5 9v5" stroke="#4ade80" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="pg-usecase-title">Prototyping</div>
      <div class="pg-usecase-desc">
        Before writing a single line of code, check whether a linear model can
        even handle your data's shape. Fail fast, iterate faster.
      </div>
      <span class="pg-usecase-tag">Model selection</span>
    </div>
  </div>
 
  <div class="pg-usecase">
    <div class="pg-usecase-icon">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <path d="M4 4h5l3 3v5H4V4z" stroke="#4ade80" stroke-width="1.2" stroke-linejoin="round"/>
        <path d="M9 4v3h3" stroke="#4ade80" stroke-width="1.2" stroke-linecap="round"/>
        <path d="M6 9h4M6 11h2" stroke="#4ade80" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="pg-usecase-title">Code Export</div>
      <div class="pg-usecase-desc">
        Skip the boilerplate. Every config you set here exports as a clean,
        fully-reproducible scikit-learn script — ready to paste into your project.
      </div>
      <span class="pg-usecase-tag">Zero boilerplate</span>
    </div>
  </div>
 
</div>
""", unsafe_allow_html=True)
 
st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

# ── Detailed documentation (collapsed by default) ─────────────────────────────
with st.expander("Show detailed documentation"):
    steps = [
        ("01", "Pick a Dataset",
         "Head to the **Dataset** page. Choose a synthetic dataset (moons, circles, XOR…) "
         "or a real sklearn dataset (Iris, Wine, Breast Cancer). Adjust sample count, noise, and test-split ratio in the sidebar."),
        ("02", "Select a Model",
         "Go to the **Train Model** page. Pick any of the 10 available classifiers. "
         "A short description and sklearn docs link appear automatically."),
        ("03", "Tune Hyperparameters",
         "Use the sidebar sliders, dropdowns, and toggles to adjust hyperparameters. "
         "Hover over any widget for a plain-English tooltip. Hit **Reset to defaults** to start over."),
        ("04", "Train & Evaluate",
         "Click **Train Model**. Accuracy, F1, Precision and Recall are shown for both train and "
         "test sets. Explore the Confusion Matrix, ROC curve, and Precision-Recall curve."),
        ("05", "Visualize the Boundary",
         "Scroll to the **Decision Boundary** section. See how the model carves up feature "
         "space — side-by-side for train and test."),
        ("06", "Dig Deeper",
         "Check **Model Insights** for feature importances, coefficients, learning curves, "
         "and validation curves. Read the Tips & Pitfalls tab for model-specific guidance."),
        ("07", "Export Your Code",
         "Scroll to **Export Code** at the bottom of the Train page. Copy or download a "
         "fully-reproducible Python script matching your exact setup."),
    ]
    left, right = st.columns(2, gap="medium")
    for i, (num, title, desc) in enumerate(steps):
        col = left if i % 2 == 0 else right
        with col:
            with st.container(border=False):
                st.markdown(f"**{num} · {title}**")
                st.caption(desc)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


# ── Model reference table ─────────────────────────────────────────────────────
st.markdown("""
<div class="pg-section-label">
  Model reference <span class="pg-section-rule"></span>
</div>
""", unsafe_allow_html=True)

models = [
    ("Logistic Regression",    "Linear",               "L1 / L2",               "Fast, interpretable, solid baseline"),
    ("K-Nearest Neighbors",    "Instance-based",       "None (implicit via k)",  "Simple, non-parametric, slows on large data"),
    ("Decision Tree",          "Tree",                 "Depth / split limits",   "Fully interpretable, prone to overfitting"),
    ("Random Forest",          "Ensemble · bagging",   "n_estimators, depth",    "Robust, handles high-dim well"),
    ("Gradient Boosting",      "Ensemble · boosting",  "LR, depth, subsample",   "Often top accuracy; slower to train"),
    ("Support Vector Machine", "Kernel-based",         "C, gamma",               "Great for small/medium data; kernel trick"),
    ("Naive Bayes",            "Probabilistic",        "var_smoothing / alpha",  "Very fast, works well with high-dim text"),
    ("LDA",                    "Linear · generative",  "Shrinkage",              "Assumes Gaussian classes, great baseline"),
    ("AdaBoost",               "Ensemble · boosting",  "n_estimators, LR",       "Sensitive to noise; elegant theory"),
    ("Voting Classifier",      "Ensemble · voting",    "Depends on sub-models",  "Combines strengths of multiple models"),
]

rows = "".join(f"""
  <tr>
    <td>{name}</td>
    <td><span class="pg-family-tag">{family}</span></td>
    <td><span class="pg-reg">{reg}</span></td>
    <td>{when}</td>
  </tr>""" for name, family, reg, when in models)

st.markdown(f"""
<div class="pg-model-wrap">
<table class="pg-model-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Family</th>
      <th>Key regularization</th>
      <th>When to use</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)


# ── Footer CTAs ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-section-label">
  Get started <span class="pg-section-rule"></span>
</div>
""", unsafe_allow_html=True)

fc1, fc2 = st.columns(2, gap="medium")
with fc1:
    st.markdown("""
    <div class="pg-footer-card">
      <div class="pg-footer-label">Navigation tip</div>
      <div class="pg-footer-text">
        Use the sidebar to move between the <strong>Dataset</strong>
        and <strong>Train Model</strong> pages.
      </div>
    </div>
    """, unsafe_allow_html=True)
with fc2:
    st.markdown("""
    <div class="pg-footer-card">
      <div class="pg-footer-label">Recommended first run</div>
      <div class="pg-footer-text">
        Start with <strong>Moons</strong> + <strong>Logistic Regression</strong>,
        then swap in <strong>SVM</strong> or <strong>Random Forest</strong>
        to see the difference.
      </div>
    </div>
    """, unsafe_allow_html=True)
import streamlit as st

st.set_page_config(
    page_title="Machine Learning Playground",
    page_icon="assets/flask.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global dark terminal styles ───────────────────────────────────────────────
# Applied here in app.py so they cascade to every page.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Page background ── */
.stApp {
    background-color: #0a0c10;
}
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    padding-left: 3.6rem !important;
    padding-right: 3.6rem !important;
}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background-color: #0d0f14 !important;
    border-right: 1px solid #1a1f2e !important;
    min-width: 230px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── Sidebar top logo area ── */
.sb-header {
    padding: 20px 18px 16px;
    border-bottom: 1px solid #1a1f2e;
    margin-bottom: 8px;
}
.sb-logo-row {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 3px;
}
.sb-brand {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #4ade80;
    text-transform: uppercase;
}
.sb-version {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 10px;
    color: #2a3045;
    letter-spacing: 0.06em;
    padding-left: 29px;
}

/* ── Sidebar nav section labels ── */
[data-testid="stSidebarNavItems"] {
    padding: 0 8px !important;
}
[data-testid="stSidebarNavSeparator"] p,
[data-testid="stSidebarNavLink"] + [data-testid="stSidebarNavSeparator"] p {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #2d3550 !important;
    font-weight: 600 !important;
    padding: 12px 8px 4px !important;
}

/* ── Sidebar nav links ── */
[data-testid="stSidebarNavLink"] {
    border-radius: 6px !important;
    padding: 7px 10px !important;
    margin: 1px 0 !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background: rgba(74,222,128,0.08) !important;
    border: none !important;
}
[data-testid="stSidebarNavLink"] span {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.02em !important;
    color: rgba(255,255,255,0.35) !important;
    font-weight: 400 !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] span {
    color: #4ade80 !important;
}

/* ── Sidebar section label (st.caption / st.markdown in sidebar) ── */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #2d3550 !important;
}

/* ── Sidebar text & widgets ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(255,255,255,0.45) !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #2d3550 !important;
    margin: 16px 0 6px !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1a1f2e !important;
    margin: 12px 0 !important;
}

/* ── Sidebar status footer ── */
.sb-status-bar {
    padding: 14px 18px;
    border-top: 1px solid #1a1f2e;
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 8px;
}
.sb-status-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #4ade80; flex-shrink: 0;
}
.sb-status-text {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 10px;
    color: #2d3550;
    letter-spacing: 0.06em;
}

/* ── Sidebar widgets (sliders, selects, inputs) ── */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4ade80 !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background: #4ade80 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #11141c !important;
    border-color: #1e2330 !important;
    color: rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] input {
    background: #11141c !important;
    border-color: #1e2330 !important;
    color: rgba(255,255,255,0.6) !important;
}

/* ── Main area typography ── */
h1, h2, h3 {
    color: #ffffff !important;
}
p, li {
    color: rgba(255,255,255);
}

/* ── Streamlit default widget overrides (main area) ── */
[data-baseweb="select"] > div {
    background: #0d0f14 !important;
    border-color: #1a1f2e !important;
}
.stButton > button {
    background: #4ade80 !important;
    color: #0a0c10 !important;
    border: none !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 7px !important;
}
.stButton > button:hover {
    background: #86efac !important;
}

/* ── Divider ── */
hr {
    border-color: #1a1f2e !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0d0f14;
    border: 1px solid #1a1f2e;
    border-radius: 10px;
    padding: 14px 12px;
}
[data-testid="stMetricLabel"] p {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #2d3550 !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 30px;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {
    background: #0d0f14 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a1f2e !important;
    border-radius: 10px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

pages = st.navigation({
    "ML Playground": [
        st.Page("pages/home.py", title="Home"),
        st.Page("pages/dataset.py", title="Dataset"),
        st.Page("pages/model.py", title="Train Model")
    ]
})

pages.run()
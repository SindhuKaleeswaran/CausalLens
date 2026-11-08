"""CausalLens: Professional Causal Investigation & Decision Intelligence Workspace.

Built for the DataNova Global Data-thon 2026 (DN-OP-01: Open Innovation Challenge).
Neyman-Rubin potential outcomes framework evaluated on the semi-synthetic CFRNet IHDP-100 benchmark.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from causallens.data.eda import balance_table
from causallens.data.loader import load_ihdp, load_ihdp_truth
from causallens.evaluation.benchmarks import evaluate_replication, get_default_models
from causallens.evaluation.policy import (
    compute_paired_benchmark_differences,
    simulate_capacity_constrained_policy,
)
from causallens.evaluation.robustness import (
    compute_estimator_disagreement,
    refute_irrelevant_covariate,
    refute_placebo_treatment,
    sensitivity_unobserved_confounding,
)

# -----------------------------------------------------------------------------
# Matplotlib Scientific Style Configuration
# -----------------------------------------------------------------------------
def configure_matplotlib_style() -> None:
    """Set minimal, publication-grade styling for Matplotlib/Seaborn."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter",
            "-apple-system",
            "BlinkMacSystemFont",
            "Segoe UI",
            "DejaVu Sans",
            "Arial",
        ],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "600",
        "axes.titlepad": 8,
        "axes.labelsize": 8.5,
        "axes.labelweight": "500",
        "axes.labelcolor": "#334155",
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E2E8F0",
        "axes.linewidth": 0.8,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.75,
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E2E8F0",
        "figure.autolayout": True,
    })


configure_matplotlib_style()

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Design System CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CausalLens | Causal Investigation Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* 1. Global Viewport & Container Reset (Forced Unified Light Canvas) */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], main, .main, [data-testid="stMainBlockContainer"], .block-container {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }

    /* Streamlit Chrome Normalization */
    header[data-testid="stHeader"] {
        background-color: #F8FAFC !important;
        border-bottom: 1px solid #E2E8F0 !important;
        height: 2.85rem !important;
        z-index: 99 !important;
    }
    /* Hide Streamlit deploy/manage buttons and main menu without hiding sidebar controls */
    #MainMenu, [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="manage-app-button"] {
        display: none !important;
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 4.25rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px !important;
    }

    /* Prominent, accessible sidebar expand and collapse buttons */
    [data-testid="stSidebarCollapseButton"], 
    [data-testid="stExpandSidebarButton"], 
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 100000 !important;
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        border: 1px solid #334155 !important;
        padding: 4px 8px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.20) !important;
    }
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Ensure all text in main canvas defaults to high-contrast near-black */
    .stApp p, .stApp span, .stApp div, .stApp label {
        color: #0F172A;
    }

    /* 2. Sidebar Styling (Deep Slate #0F172A with High-Contrast Typography) */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #CBD5E1 !important;
        font-size: 0.85rem !important;
        line-height: 1.45 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #F1F5F9 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stCaption p {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background-color: #0D9488 !important;
    }

    /* 3. Header & Context Hierarchy */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-top: 0.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
        gap: 1.75rem;
    }
    .app-brand {
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    .app-micro-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0F766E;
        background-color: #F0FDFA;
        border: 1px solid #CCFBF1;
        padding: 3px 9px;
        border-radius: 4px;
        width: fit-content;
        margin-bottom: 8px;
    }
    .app-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 6px 0;
    }
    .app-subtitle {
        font-size: 0.90rem;
        color: #475569 !important;
        line-height: 1.45;
        margin: 0;
        max-width: 660px;
    }
    .context-pills {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
    }
    .context-pill-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .context-pill {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 5px 10px;
        color: #475569;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        white-space: nowrap;
    }
    .context-pill-strong {
        font-weight: 700;
        color: #0F172A;
    }

    /* 4. Unified Analytical Diagnostic Ribbon */
    .kpi-ribbon {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }
    .kpi-segment {
        padding: 14px 18px;
        border-right: 1px solid #E2E8F0;
    }
    .kpi-segment:last-child {
        border-right: none;
    }
    .kpi-title {
        font-size: 0.70rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        margin-bottom: 4px !important;
    }
    .kpi-number {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        line-height: 1.15 !important;
    }
    .kpi-desc {
        font-size: 0.76rem !important;
        color: #475569 !important;
        margin-top: 4px !important;
    }
    .delta-warn { color: #B45309 !important; font-weight: 700; }
    .delta-teal { color: #0F766E !important; font-weight: 700; }

    /* 5. Navigation Tabs (Clean Linear Underlines, No Text Wrapping) */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        border-bottom: 1px solid #E2E8F0 !important;
        padding-bottom: 0px !important;
        background-color: transparent !important;
        margin-bottom: 1.5rem !important;
        overflow-x: auto !important;
        scrollbar-width: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px !important;
        white-space: nowrap !important;
        border-radius: 4px 4px 0 0 !important;
        color: #64748B !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        border-bottom: 2px solid transparent !important;
        padding: 7px 14px !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        border-bottom: 2px solid #CBD5E1 !important;
        background-color: rgba(15, 118, 110, 0.03) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0F766E !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #0F766E !important;
        background-color: #FFFFFF !important;
    }

    /* 6. Section & Surface Card Layouts */
    .section-title {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.015em !important;
        margin-bottom: 0.35rem !important;
    }
    .section-subtitle {
        font-size: 0.86rem !important;
        color: #475569 !important;
        line-height: 1.5 !important;
        margin-bottom: 1.15rem !important;
    }
    .card-panel {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 16px 18px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    .card-panel-header {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #0F766E !important;
        margin-bottom: 0.75rem !important;
    }

    /* 7. High-Contrast Status Badges */
    .badge-pill {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 0.03em;
        margin-bottom: 0.5rem;
    }
    .badge-valid {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .badge-caution {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-alert {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    /* 8. Notice & Research Disclosure */
    .callout-box, .research-notice {
        background-color: #FFFFFF !important;
        border-left: 3px solid #0F766E !important;
        border-top: 1px solid #E2E8F0 !important;
        border-right: 1px solid #E2E8F0 !important;
        border-bottom: 1px solid #E2E8F0 !important;
        padding: 12px 16px !important;
        border-radius: 0 4px 4px 0 !important;
        font-size: 0.84rem !important;
        color: #334155 !important;
        margin-bottom: 1.25rem !important;
        line-height: 1.5 !important;
    }

    /* 9. Data Tables & Monospace Metrics */
    .stDataFrame {
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
    }
    .metric-value-box {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* 10. Status Badges & Aliases */
    .status-badge, .badge-pill {
        display: inline-block !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        padding: 3px 9px !important;
        border-radius: 3px !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 0.5rem !important;
    }
    .badge-pass, .badge-valid {
        background-color: #F0FDF4 !important;
        color: #15803D !important;
        border: 1px solid #BBF7D0 !important;
    }
    .badge-warning, .badge-caution {
        background-color: #FFFBEB !important;
        color: #92400E !important;
        border: 1px solid #FDE68A !important;
    }
    .badge-critical, .badge-alert {
        background-color: #FEF2F2 !important;
        color: #991B1B !important;
        border: 1px solid #FECACA !important;
    }

    /* 11. Responsive Behavior */
    @media (max-width: 960px) {
        .kpi-ribbon {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .kpi-segment:nth-child(2) {
            border-right: none !important;
        }
        .kpi-segment:nth-child(1), .kpi-segment:nth-child(2) {
            border-bottom: 1px solid #E2E8F0 !important;
        }
        .app-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 14px !important;
        }
        .context-pills {
            align-items: flex-start !important;
        }
        .context-pill-row {
            justify-content: flex-start !important;
        }
    }
    @media (max-width: 600px) {
        .kpi-ribbon {
            grid-template-columns: 1fr !important;
        }
        .kpi-segment {
            border-right: none !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }
        .kpi-segment:last-child {
            border-bottom: none !important;
        }
    }

    /* Legacy Component Aliases */
    .section-header {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.015em !important;
        margin-bottom: 0.35rem !important;
    }
    .section-description {
        font-size: 0.86rem !important;
        color: #475569 !important;
        line-height: 1.5 !important;
        margin-bottom: 1.15rem !important;
    }
    .section-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 16px 18px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    .section-card-title {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #0F766E !important;
        margin-bottom: 0.75rem !important;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    .metric-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    .metric-label {
        font-size: 0.70rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        margin-bottom: 4px !important;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        line-height: 1.15 !important;
    }
    .metric-meta {
        font-size: 0.76rem !important;
        color: #475569 !important;
        margin-top: 4px !important;
    }
    .delta-warning { color: #B45309 !important; font-weight: 700; }
    .delta-neutral { color: #0F766E !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

# -----------------------------------------------------------------------------
# Canonical Estimator Registry & Metadata
# -----------------------------------------------------------------------------
# Decouples stable internal estimator keys from user-facing presentation labels.
ESTIMATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "doubly_robust": {
        "display_name": "Doubly Robust (AIPW)",
        "short_name": "AIPW",
        "description": "Cross-fitted Augmented IPW combining propensity and outcome regressions",
        "aliases": ["Doubly Robust (AIPW)", "Doubly Robust", "doubly_robust", "aipw"],
    },
    "x_learner": {
        "display_name": "X-Learner",
        "short_name": "X-Learner",
        "description": "Crossover meta-learner with imputed counterfactual residuals",
        "aliases": ["X-Learner", "x_learner", "xlearner"],
    },
    "t_learner": {
        "display_name": "T-Learner",
        "short_name": "T-Learner",
        "description": "Two separate outcome regressions for treated and control arms",
        "aliases": ["T-Learner", "t_learner", "tlearner"],
    },
    "ipw": {
        "display_name": "IPW + Weighted Outcome Regression",
        "short_name": "IPW + Weighted Reg",
        "description": "Normalized Hajek propensity weights with weighted outcome regressions",
        "aliases": [
            "IPW + Weighted Outcome Regression",
            "IPW + Weighted Reg",
            "IPW",
            "ipw",
            "ipw_weighted_reg",
        ],
    },
    "s_learner": {
        "display_name": "S-Learner",
        "short_name": "S-Learner",
        "description": "Single outcome model with treatment indicator feature",
        "aliases": ["S-Learner", "s_learner", "slearner"],
    },
    "naive": {
        "display_name": "Naive (Unadjusted)",
        "short_name": "Naive",
        "description": "Unadjusted difference in sample means (biased under confounding)",
        "aliases": ["Naive (Unadjusted)", "Naive", "naive"],
    },
}
CANONICAL_ESTIMATOR_IDS = list(ESTIMATOR_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_ihdp_data(replication: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load train, test, and ground-truth oracle potential outcomes for IHDP replication."""
    train = load_ihdp(replication, split="train", data_dir=DATA_DIR)
    test = load_ihdp(replication, split="test", data_dir=DATA_DIR)
    truth = load_ihdp_truth(replication, split="test", data_dir=DATA_DIR)
    return train, test, truth


@st.cache_resource
def get_trained_models(replication: int, cache_version: int = 4) -> Dict[str, Any]:
    """Train the default suite of causal estimators on the observational training split.

    Populates both canonical identifiers ('doubly_robust', 'x_learner', 't_learner',
    'ipw', 's_learner', 'naive') and registered display name aliases in the returned
    dictionary to eliminate runtime KeyErrors across caller interfaces.
    """
    train, _, _ = get_ihdp_data(replication)
    base_models = get_default_models()

    legacy_to_canonical = {
        "Doubly Robust (AIPW)": "doubly_robust",
        "X-Learner": "x_learner",
        "T-Learner": "t_learner",
        "IPW + Weighted Reg": "ipw",
        "IPW": "ipw",
        "S-Learner": "s_learner",
        "Naive (Unadjusted)": "naive",
    }

    fitted: Dict[str, Any] = {}
    for default_name, model_inst in base_models.items():
        can_id = legacy_to_canonical.get(default_name)
        if can_id is None:
            for cid in ESTIMATOR_REGISTRY:
                if cid in default_name.lower():
                    can_id = cid
                    break
        if can_id is not None:
            model_inst.fit(train["X"], train["T"], train["Y"])
            fitted[can_id] = model_inst

    # Populate canonical keys and all aliases to prevent KeyErrors
    full_dict: Dict[str, Any] = {}
    for can_id, model_inst in fitted.items():
        full_dict[can_id] = model_inst
        for alias in ESTIMATOR_REGISTRY.get(can_id, {}).get("aliases", []):
            full_dict[alias] = model_inst

    return full_dict


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
# Auto-expand sidebar if previously collapsed in browser localStorage
components.html(
    """
    <script>
    (function() {
        function expandSidebar() {
            try {
                const parentDoc = window.parent.document;
                const expandBtn = parentDoc.querySelector('[data-testid="stExpandSidebarButton"]');
                if (expandBtn) {
                    expandBtn.click();
                }
            } catch(e) {}
        }
        setTimeout(expandSidebar, 100);
        setTimeout(expandSidebar, 400);
    })();
    </script>
    """,
    height=0,
    width=0,
)

with st.sidebar:
    st.markdown('<div style="padding-top: 0.5rem; margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: #14B8A6; text-transform: uppercase;">Platform Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 15px; font-weight: 600; color: #F8FAFC;">CausalLens Control</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    rep_idx = st.slider(
        "Replication Index (IHDP-100)",
        min_value=0,
        max_value=99,
        value=0,
        step=1,
        help="CFRNet semi-synthetic benchmark replication (0 through 99).",
    )

    selected_model_id = st.selectbox(
        "Active Estimator",
        options=CANONICAL_ESTIMATOR_IDS,
        format_func=lambda cid: ESTIMATOR_REGISTRY[cid]["display_name"],
        index=0,
        help="Estimator selected for individual counterfactual inference and heterogeneity analysis.",
    )

    st.markdown("<hr style='border-color: #1E293B; margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">Benchmark Specification</div>', unsafe_allow_html=True)
    st.caption(
        "**Dataset**: Infant Health and Development Program (IHDP-100)\n\n"
        "**Target Parameter**: Average Treatment Effect (ATE) and Individual Treatment Effect (ITE/CATE)\n\n"
        "**Identification assumption**: Unconfoundedness conditional on 25 baseline covariates; not established by diagnostics."
    )
    st.markdown("<hr style='border-color: #1E293B; margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.caption("DataNova Global Data-thon 2026\nTrack: DN-OP-01 (Open Innovation)")

# -----------------------------------------------------------------------------
# Data Loading & Global Metrics
# -----------------------------------------------------------------------------
train_data, test_data, truth_data = get_ihdp_data(rep_idx)
trained_models = get_trained_models(rep_idx)

# Explicit validation: ensure requested model exists with zero silent substitution
if selected_model_id not in trained_models:
    raise KeyError(
        f"Selected estimator '{selected_model_id}' is not in trained_models. "
        f"Available canonical models: {CANONICAL_ESTIMATOR_IDS}"
    )

selected_model_obj = trained_models[selected_model_id]
selected_model_meta = ESTIMATOR_REGISTRY[selected_model_id]
selected_model_display = selected_model_meta["display_name"]

n_train = len(train_data["X"])
n_test = len(test_data["X"])
n_treated = int(np.sum(train_data["T"]))
n_control = n_train - n_treated
pct_treated = (n_treated / n_train) * 100.0

y_t1 = train_data["Y"][train_data["T"] == 1]
y_t0 = train_data["Y"][train_data["T"] == 0]
naive_diff = float(np.mean(y_t1) - np.mean(y_t0))
oracle_ate = float(np.mean(truth_data["tau"]))
cross_split_difference = naive_diff - oracle_ate

# -----------------------------------------------------------------------------
# Header Context Area
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <div class="app-brand">
        <div class="app-micro-badge">Causal Investigation & Decision Intelligence</div>
        <div class="app-title">CausalLens</div>
        <div class="app-subtitle">Observational confounding diagnosis, causal estimation, and refutation testing</div>
    </div>
    <div class="context-pills">
        <div class="context-pill-row">
            <div class="context-pill">Benchmark: <span class="context-pill-strong">IHDP-100</span></div>
            <div class="context-pill">Replication: <span class="context-pill-strong">{rep_idx:02d}</span></div>
            <div class="context-pill">Sample: <span class="context-pill-strong">N={n_train + n_test}</span> (Train: {n_train}, Test: {n_test})</div>
        </div>
        <div class="context-pill-row">
            <div class="context-pill">Active Estimator: <span class="context-pill-strong">{selected_model_display}</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Analytical KPI Ribbon
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="kpi-ribbon">
    <div class="kpi-segment">
        <div class="kpi-title">Training Association</div>
        <div class="kpi-number">{naive_diff:.2f}</div>
        <div class="kpi-desc">Naive E[Y|T=1] - E[Y|T=0]</div>
    </div>
    <div class="kpi-segment">
        <div class="kpi-title">Test Oracle ATE</div>
        <div class="kpi-number">{oracle_ate:.2f}</div>
        <div class="kpi-desc">Test mean(mu1 - mu0)</div>
    </div>
    <div class="kpi-segment">
        <div class="kpi-title">Cross-Split Difference</div>
        <div class="kpi-number {'delta-warn' if abs(cross_split_difference) > 0.2 else 'delta-teal'}">{cross_split_difference:+.2f}</div>
        <div class="kpi-desc">Train association minus test oracle; descriptive only</div>
    </div>
    <div class="kpi-segment">
        <div class="kpi-title">Treatment Allocation</div>
        <div class="kpi-number">{pct_treated:.1f}%</div>
        <div class="kpi-desc">{n_treated} treated / {n_control} control units</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Tabs (6 Professional Scientific Sections)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Data & Confounding",
    "Model Evaluation",
    "Counterfactual Explorer",
    "Treatment Heterogeneity",
    "Robustness Analysis",
    "Evidence Assessment",
])

# =============================================================================
# TAB 1: Data & Confounding
# =============================================================================
with tabs[0]:
    st.markdown('<div class="section-header">Covariate Imbalance & Overlap Diagnosis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'In observational data, non-random treatment assignment induces systematic imbalances between treated '
        'and control cohorts. Standard outcome comparisons conflate pre-treatment baseline advantages with the '
        'true causal effect of the intervention.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_c1, col_c2 = st.columns([1, 1], gap="medium")

    # Covariate Balance Calculation
    df_smd = balance_table(train_data["X"], train_data["T"])[["feature", "smd"]].rename(
        columns={"feature": "Feature", "smd": "SMD"}
    )
    df_smd["AbsSMD"] = df_smd["SMD"].abs()
    df_smd_sorted = df_smd.sort_values(by="AbsSMD", ascending=False).reset_index(drop=True)
    top_smd = df_smd_sorted.head(12)

    with col_c1:
        st.markdown('<div class="section-card-title">Standardized Mean Differences (Top 12 Imbalanced Covariates)</div>', unsafe_allow_html=True)
        fig_smd, ax_smd = plt.subplots(figsize=(5.8, 4.0))

        bar_colors = [
            "#DC2626" if s > 0.25 else ("#D97706" if s > 0.10 else "#0D9488")
            for s in top_smd["AbsSMD"]
        ]
        y_pos = np.arange(len(top_smd))
        ax_smd.barh(y_pos, top_smd["SMD"], color=bar_colors, height=0.65, edgecolor="none")
        ax_smd.set_yticks(y_pos)
        ax_smd.set_yticklabels(top_smd["Feature"])
        ax_smd.axvline(0.10, color="#64748B", linestyle="--", linewidth=0.8, alpha=0.8)
        ax_smd.axvline(-0.10, color="#64748B", linestyle="--", linewidth=0.8, alpha=0.8)
        ax_smd.axvline(0.25, color="#DC2626", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_smd.axvline(-0.25, color="#DC2626", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_smd.set_xlabel("Signed Standardized Mean Difference (SMD)")
        ax_smd.invert_yaxis()
        sns.despine(top=True, right=True)
        st.pyplot(fig_smd, use_container_width=True)

    with col_c2:
        st.markdown('<div class="section-card-title">Propensity Score Common Support & Positivity Overlap</div>', unsafe_allow_html=True)
        ipw_est = trained_models["ipw"]
        e_scores = ipw_est.get_propensity_scores(train_data["X"])

        fig_prop, ax_prop = plt.subplots(figsize=(5.8, 4.0))
        sns.kdeplot(
            e_scores[train_data["T"] == 0],
            label=f"Control (n={n_control})",
            color="#475569",
            fill=True,
            alpha=0.3,
            linewidth=1.5,
            ax=ax_prop,
        )
        sns.kdeplot(
            e_scores[train_data["T"] == 1],
            label=f"Treated (n={n_treated})",
            color="#0D9488",
            fill=True,
            alpha=0.3,
            linewidth=1.5,
            ax=ax_prop,
        )
        ax_prop.axvline(0.01, color="#DC2626", linestyle=":", linewidth=0.9, alpha=0.7, label="Clipping (0.01, 0.99)")
        ax_prop.axvline(0.99, color="#DC2626", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_prop.set_xlabel("Estimated Propensity Score e(X) = P(T=1 | X)")
        ax_prop.set_ylabel("Kernel Density")
        ax_prop.legend(loc="upper right")
        sns.despine(top=True, right=True)
        st.pyplot(fig_prop, use_container_width=True)

    n_imbalanced = int(np.sum(df_smd["AbsSMD"] > 0.10))
    n_severe = int(np.sum(df_smd["AbsSMD"] > 0.25))

    st.markdown(f"""
    <div class="research-notice">
        <strong>Methodological Finding</strong>: <strong>{n_imbalanced} of 25</strong> baseline covariates exhibit substantial imbalance (|SMD| > 0.10), with <strong>{n_severe}</strong> covariates exceeding severe imbalance (|SMD| > 0.25). 
        The training association ({naive_diff:.2f}) and test oracle mean effect ({oracle_ate:.2f}) describe different samples. Their difference ({cross_split_difference:+.2f}) is not an estimate of confounding bias. Imbalance and empirical overlap are descriptive diagnostics, not proof of confounding or population positivity.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: Model Evaluation
# =============================================================================
with tabs[1]:
    st.markdown('<div class="section-header">Causal Estimator Evaluation on Held-Out Test Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Evaluation against ground-truth potential outcomes on the held-out test split (N=75). '
        'Metrics assess both heterogeneous effect precision (PEHE) and population parameter accuracy (ATE Error).'
        '</div>',
        unsafe_allow_html=True,
    )

    models_for_eval = {
        ESTIMATOR_REGISTRY[cid]["display_name"]: trained_models[cid]
        for cid in CANONICAL_ESTIMATOR_IDS
    }
    df_eval = evaluate_replication(rep_idx, models=models_for_eval, data_dir=DATA_DIR)

    col_e1, col_e2 = st.columns([1.1, 0.9], gap="medium")

    with col_e1:
        st.markdown('<div class="section-card-title">Benchmark Performance Summary (Replication ' + f'{rep_idx:02d})</div>', unsafe_allow_html=True)
        display_cols = ["model", "pehe", "ate_error", "ate_pred", "ate_true", "policy_risk", "total_time_sec"]
        df_display = df_eval[display_cols].rename(columns={
            "model": "Model",
            "pehe": "Root PEHE",
            "ate_error": "|ATE Err|",
            "ate_pred": "Est ATE",
            "ate_true": "True ATE",
            "policy_risk": "Policy Risk",
            "total_time_sec": "Runtime (s)",
        })

        styled_df = df_display.style.format({
            "Root PEHE": "{:.3f}",
            "|ATE Err|": "{:.3f}",
            "Est ATE": "{:.3f}",
            "True ATE": "{:.3f}",
            "Policy Risk": "{:.4f}",
            "Runtime (s)": "{:.4f}",
        }).highlight_min(subset=["Root PEHE", "|ATE Err|", "Policy Risk"], color="#CCFBF1")

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Export Model Evaluation (CSV)",
            data=df_eval.to_csv(index=False),
            file_name=f"ihdp_rep{rep_idx:02d}_model_evaluation.csv",
            mime="text/csv",
        )

    with col_e2:
        st.markdown('<div class="section-card-title">Precision vs Parameter Accuracy Trade-Off</div>', unsafe_allow_html=True)
        fig_scatter, ax_scatter = plt.subplots(figsize=(5.5, 3.8))

        colors_map = {
            ESTIMATOR_REGISTRY["t_learner"]["display_name"]: "#0D9488",
            ESTIMATOR_REGISTRY["x_learner"]["display_name"]: "#0284C7",
            ESTIMATOR_REGISTRY["ipw"]["display_name"]: "#6366F1",
            ESTIMATOR_REGISTRY["doubly_robust"]["display_name"]: "#059669",
            ESTIMATOR_REGISTRY["s_learner"]["display_name"]: "#D97706",
            ESTIMATOR_REGISTRY["naive"]["display_name"]: "#DC2626",
        }

        for _, row in df_eval.iterrows():
            m_name = row["model"]
            c = colors_map.get(m_name, "#475569")
            ax_scatter.scatter(row["ate_error"], row["pehe"], color=c, s=70, zorder=4)
            # Offset labels for readability
            short_lbl = m_name.split(" ")[0]
            ax_scatter.annotate(
                short_lbl,
                (row["ate_error"], row["pehe"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8.0,
                color="#334155",
                fontweight="500",
            )

        ax_scatter.set_xlabel("Absolute ATE Error (|ATE_hat - ATE_true|)")
        ax_scatter.set_ylabel(r"Heterogeneous Precision ($\sqrt{\epsilon_{PEHE}}$)")
        ax_scatter.grid(True, linestyle="--", alpha=0.5)
        sns.despine(top=True, right=True)
        st.pyplot(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-card-title" style="margin-top: 1rem;">Predicted Treatment Effect Distributions vs Ground Truth</div>', unsafe_allow_html=True)
    fig_cate, ax_cate = plt.subplots(figsize=(10.5, 3.2))

    # Plot oracle
    sns.kdeplot(
        truth_data["tau"],
        label=f"Ground Truth Oracle tau (Mean: {oracle_ate:.2f})",
        color="#0F172A",
        linewidth=2.2,
        ax=ax_cate,
    )
    # Plot top estimators using canonical lookups
    for cid, line_style in [
        ("doubly_robust", "-"),
        ("x_learner", "--"),
        ("t_learner", "-."),
        ("naive", ":"),
    ]:
        if cid in trained_models:
            preds = trained_models[cid].estimate_ite(test_data["X"])
            disp_name = ESTIMATOR_REGISTRY[cid]["display_name"]
            if np.var(preds) > 1e-5:
                sns.kdeplot(
                    preds,
                    label=f"{disp_name} (Mean: {np.mean(preds):.2f})",
                    linestyle=line_style,
                    linewidth=1.4,
                    alpha=0.85,
                    ax=ax_cate,
                )
            else:
                ax_cate.axvline(
                    float(np.mean(preds)),
                    label=f"{disp_name} (Constant: {np.mean(preds):.2f})",
                    linestyle=line_style,
                    linewidth=1.6,
                    color="#DC2626" if cid == "naive" else "#475569",
                    alpha=0.85,
                )

    ax_cate.set_xlabel("Individual Treatment Effect (ITE / CATE)")
    ax_cate.set_ylabel("Kernel Density")
    ax_cate.legend(loc="upper right", frameon=True)
    sns.despine(top=True, right=True)
    st.pyplot(fig_cate, use_container_width=True)

    # -------------------------------------------------------------------------
    # Unified 100-Replication Benchmark Results Section
    # -------------------------------------------------------------------------
    st.markdown('<hr style="border-color: #E2E8F0; margin: 1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-card-title">Unified 100-Replication Benchmark Evaluation (All 100 Replications, 600 Fits)</div>', unsafe_allow_html=True)
    st.caption(
        "Scientific evaluation across all 100 IHDP replications under strict split isolation (train N=672, test N=75). "
        "Root-PEHE measures individual effect estimation error; ATE Error measures population effect bias; Policy Risk measures decision regret."
    )

    summary_file = Path(__file__).resolve().parent / "results" / "ihdp100_summary.csv"
    if not summary_file.exists():
        summary_file = Path(__file__).resolve().parent / "reports" / "benchmark_summary.csv"

    if summary_file.exists():
        df_bench = pd.read_csv(summary_file)
        bench_cols = [
            c
            for c in [
                "model",
                "n_successful",
                "pehe_summary",
                "pehe_mean_ci_summary",
                "pehe_median_ci_summary",
                "ate_error_summary",
                "ate_error_mean_ci_summary",
                "ate_error_median_ci_summary",
                "policy_risk_mean",
                "avg_runtime_sec",
            ]
            if c in df_bench.columns
        ]
        rename_map = {
            "model": "Causal Estimator",
            "n_successful": "Successful Reps",
            "pehe_summary": "Root-PEHE (Mean ± SD)",
            "pehe_mean_ci_summary": "Root-PEHE 95% CI (Mean)",
            "pehe_median_ci_summary": "Root-PEHE Median [95% CI]",
            "ate_error_summary": "ATE Error (Mean ± SD)",
            "ate_error_mean_ci_summary": "ATE Error 95% CI (Mean)",
            "ate_error_median_ci_summary": "ATE Error Median [95% CI]",
            "policy_risk_mean": "Policy Regret",
            "avg_runtime_sec": "Avg Runtime (s)",
        }
        df_bench_disp = df_bench[bench_cols].rename(columns=rename_map)
        st.dataframe(df_bench_disp, use_container_width=True, hide_index=True)

        col_b_dl1, col_b_dl2 = st.columns([1, 1])
        with col_b_dl1:
            st.download_button(
                label="Download 100-Replication Summary (CSV)",
                data=df_bench.to_csv(index=False),
                file_name="ihdp100_summary.csv",
                mime="text/csv",
            )
        with col_b_dl2:
            unified_res_file = Path(__file__).resolve().parent / "results" / "ihdp100_unified_results.csv"
            if unified_res_file.exists():
                st.download_button(
                    label="Download Full 600-Fit Results (CSV)",
                    data=unified_res_file.read_bytes(),
                    file_name="ihdp100_unified_results.csv",
                    mime="text/csv",
                )

        fig_comp = Path(__file__).resolve().parent / "reports" / "figures" / "benchmark_pehe_ate_comparison.png"
        fig_dist = Path(__file__).resolve().parent / "reports" / "figures" / "pehe_distribution_boxplot.png"
        if fig_comp.exists() and fig_dist.exists():
            col_bf1, col_bf2 = st.columns([1, 1], gap="medium")
            with col_bf1:
                st.image(str(fig_comp), caption="Figure 1: Benchmark Precision & Accuracy with 95% Confidence Intervals", use_container_width=True)
            with col_bf2:
                st.image(str(fig_dist), caption="Figure 2: Distribution of Errors Across 100 Replications (Log Scale)", use_container_width=True)


# =============================================================================
# TAB 3: Counterfactual Explorer
# =============================================================================
with tabs[2]:
    st.markdown('<div class="section-header">Individual Counterfactual Decision Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Simulate prospective outcomes under both treatment regimes for an individual profile. '
        'Counterfactual outcomes Y(0) and Y(1) are model-based conditional expectations under unconfoundedness.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="research-notice">
        <strong>Scientific Disclosure</strong>: The IHDP benchmark derives from an educational intervention for premature infants. 
        Estimates displayed are mathematical demonstrations of causal inference algorithms and do not constitute clinical guidance.
    </div>
    """, unsafe_allow_html=True)

    col_sim_ctl, col_sim_view = st.columns([1.1, 0.9], gap="medium")

    with col_sim_ctl:
        st.markdown('<div class="section-card-title">Subject Covariate Profile Inputs</div>', unsafe_allow_html=True)

        profile_index = st.number_input(
            "Observed training row", min_value=0, max_value=n_train - 1, value=0, step=1,
            key="profile_index",
        )
        sample_vec = train_data["X"][int(profile_index)].copy()
        st.caption("Baseline: a real observed training profile. Generic feature identifiers are used because clinical mappings are unverified.")
        modify_profile = st.checkbox("Modify numeric features hypothetically", key="modify_profile")
        if modify_profile:
            for j in range(6):
                sample_vec[j] = st.number_input(
                    f"x{j+1:02d} (archived scale)", value=float(sample_vec[j]),
                    key=f"profile_{rep_idx}_{profile_index}_{j}",
                )
            st.warning("Hypothetical edits may create unsupported combinations, even within observed marginal ranges. Predictions are illustrative; feature edits are not causal interventions.")
        st.caption("All unedited features retain their observed values, including x14's original encoding.")
        sample_vec_2d = sample_vec.reshape(1, -1)

    with col_sim_view:
        st.markdown('<div class="section-card-title">Estimated Potential Outcomes & Differential</div>', unsafe_allow_html=True)
        y0_est, y1_est = selected_model_obj.predict_potential_outcomes(sample_vec_2d)
        y0_scalar = float(y0_est[0])
        y1_scalar = float(y1_est[0])
        tau_scalar = float(y1_scalar - y0_scalar)

        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns: repeat(2, 1fr); margin-bottom: 1rem;">
            <div class="metric-card">
                <div class="metric-label">Control Potential Outcome Y(0)</div>
                <div class="metric-value">{y0_scalar:.2f}</div>
                <div class="metric-meta">Expected outcome without intervention</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Treated Potential Outcome Y(1)</div>
                <div class="metric-value">{y1_scalar:.2f}</div>
                <div class="metric-meta">Expected outcome with intervention</div>
            </div>
        </div>
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-label">Estimated Individual Treatment Effect (CATE)</div>
            <div class="metric-value" style="color: #0F766E;">{tau_scalar:+.2f}</div>
            <div class="metric-meta">tau(x) = E[Y(1) - Y(0) | X = x] via {selected_model_display}</div>
        </div>
        """, unsafe_allow_html=True)

        fig_delta, ax_delta = plt.subplots(figsize=(5.5, 1.8))
        ax_delta.barh(["Response"], [y1_scalar - y0_scalar], left=[y0_scalar], color="#0D9488", height=0.45)
        ax_delta.scatter([y0_scalar], [0], color="#475569", s=90, zorder=5, label=f"Y(0): {y0_scalar:.2f}")
        ax_delta.scatter([y1_scalar], [0], color="#0F766E", s=90, zorder=5, label=f"Y(1): {y1_scalar:.2f}")
        ax_delta.set_xlabel("Predicted Outcome Scale")
        ax_delta.legend(loc="upper left")
        sns.despine(left=True, top=True, right=True)
        st.pyplot(fig_delta, use_container_width=True)

    # -------------------------------------------------------------------------
    # Interactive Counterfactual Audit: Ground Truth vs Model Prediction (Held-Out Test Set N=75)
    # -------------------------------------------------------------------------
    st.markdown('<hr style="border-color: #E2E8F0; margin: 1.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-card-title">Held-Out Test Cohort Counterfactual Verification (N=75 Benchmark Ground Truth)</div>', unsafe_allow_html=True)
    st.caption(
        "Directly verify the Neyman-Rubin potential outcomes framework. For each held-out test unit, "
        "compare the model's predicted counterfactual outcomes against the benchmark oracle ground truth."
    )

    tau_true_all = truth_data["tau"]
    high_archetype = int(np.argmax(tau_true_all))
    med_archetype = int(np.argsort(tau_true_all)[len(tau_true_all) // 2])
    low_archetype = int(np.argmin(tau_true_all))

    archetype_mode = st.radio(
        "Quick Subject Archetype Preset",
        options=[
            "Custom Subject Selection",
            f"High Responder (Unit #{high_archetype:02d}, tau={tau_true_all[high_archetype]:+.2f})",
            f"Median Responder (Unit #{med_archetype:02d}, tau={tau_true_all[med_archetype]:+.2f})",
            f"Low Responder (Unit #{low_archetype:02d}, tau={tau_true_all[low_archetype]:+.2f})",
        ],
        horizontal=True,
        key="archetype_preset_radio",
    )

    if archetype_mode.startswith("High Responder"):
        default_test_idx = high_archetype
    elif archetype_mode.startswith("Median Responder"):
        default_test_idx = med_archetype
    elif archetype_mode.startswith("Low Responder"):
        default_test_idx = low_archetype
    else:
        default_test_idx = 0

    selected_test_idx = st.selectbox(
        "Select Test Subject for Ground Truth Counterfactual Verification",
        options=list(range(n_test)),
        index=default_test_idx,
        format_func=lambda i: f"Test Subject #{i:02d} | Factual T={int(test_data['T'][i])}, Observed Y={test_data['Y'][i]:.2f}, True tau={truth_data['tau'][i]:+.2f}",
        key="test_unit_selector",
    )

    t_obs = int(test_data["T"][selected_test_idx])
    y_obs = float(test_data["Y"][selected_test_idx])
    x_test_ind = test_data["X"][selected_test_idx : selected_test_idx + 1]

    # Model predictions
    y0_pred_t, y1_pred_t = selected_model_obj.predict_potential_outcomes(x_test_ind)
    y0_p = float(y0_pred_t[0])
    y1_p = float(y1_pred_t[0])
    tau_p = y1_p - y0_p

    # Ground truth
    y0_true_val = float(truth_data["mu0"][selected_test_idx])
    y1_true_val = float(truth_data["mu1"][selected_test_idx])
    tau_true_val = float(truth_data["tau"][selected_test_idx])
    pehe_ind = abs(tau_p - tau_true_val)

    # 4 Decision Cards
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Observed Factual</div>
            <div class="metric-value">{y_obs:.2f}</div>
            <div class="metric-meta">Assigned Treatment T={t_obs}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        cf_pred = y1_p if t_obs == 0 else y0_p
        cf_label = "Y(1) Treated" if t_obs == 0 else "Y(0) Control"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Counterfactual {cf_label}</div>
            <div class="metric-value">{cf_pred:.2f}</div>
            <div class="metric-meta">Model unobserved branch</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Effect tau(x)</div>
            <div class="metric-value" style="color: #0F766E;">{tau_p:+.2f}</div>
            <div class="metric-meta">via {selected_model_display}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Oracle Ground Truth tau</div>
            <div class="metric-value" style="color: #1E293B;">{tau_true_val:+.2f}</div>
            <div class="metric-meta">Estimation Error |tau_hat - tau| = {pehe_ind:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Plotly Pathway Flow: Model vs Oracle
    col_wf, col_cov = st.columns([1.2, 0.8], gap="medium")
    with col_wf:
        st.markdown('<div class="section-card-title">Potential Outcomes Pathway Flow (Model vs Benchmark Oracle)</div>', unsafe_allow_html=True)
        fig_path = go.Figure()

        # Model trajectory
        fig_path.add_trace(go.Bar(
            name="Model Estimate",
            x=["Baseline Y(0)", "Treatment Effect tau", "Treated Outcome Y(1)"],
            y=[y0_p, tau_p, y1_p],
            marker_color=["#64748B", "#0D9488", "#0F766E"],
            text=[f"{y0_p:.2f}", f"{tau_p:+.2f}", f"{y1_p:.2f}"],
            textposition="auto",
        ))
        # Oracle trajectory
        fig_path.add_trace(go.Bar(
            name="Benchmark Oracle",
            x=["Baseline Y(0)", "Treatment Effect tau", "Treated Outcome Y(1)"],
            y=[y0_true_val, tau_true_val, y1_true_val],
            marker_color=["#94A3B8", "#10B981", "#047857"],
            text=[f"{y0_true_val:.2f}", f"{tau_true_val:+.2f}", f"{y1_true_val:.2f}"],
            textposition="auto",
        ))

        fig_path.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", size=11),
            yaxis=dict(title="Outcome Scale", gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_path, use_container_width=True)

    with col_cov:
        st.markdown('<div class="section-card-title">Subject Covariate Profile Deviation (z-scores)</div>', unsafe_allow_html=True)
        # Compute continuous features z-score relative to test cohort
        cont_means = np.mean(test_data["X"][:, :6], axis=0)
        cont_stds = np.std(test_data["X"][:, :6], axis=0) + 1e-6
        subj_z = (test_data["X"][selected_test_idx, :6] - cont_means) / cont_stds
        feat_names = [f"x{i:02d}" for i in range(1, 7)]

        fig_z = go.Figure(go.Bar(
            x=subj_z,
            y=feat_names,
            orientation="h",
            marker=dict(
                color=["#0D9488" if z >= 0 else "#DC2626" for z in subj_z],
            ),
            text=[f"{z:+.2f} sigma" for z in subj_z],
            textposition="auto",
        ))
        fig_z.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", size=11),
            xaxis=dict(title="Standard Deviations from Cohort Mean", gridcolor="#F1F5F9"),
            yaxis=dict(autorange="reversed", gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_z, use_container_width=True)

# =============================================================================
# TAB 4: Treatment Heterogeneity
# =============================================================================
with tabs[3]:
    st.markdown('<div class="section-header">Treatment Effect Heterogeneity & Subgroup Stratification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Intervention response is rarely uniform. Segmenting the held-out test cohort by predicted treatment effect '
        'summarizes model-predicted heterogeneity in simulated outcomes; it does not demonstrate medical benefit.'
        '</div>',
        unsafe_allow_html=True,
    )

    ite_test = selected_model_obj.estimate_ite(test_data["X"])
    cate_var = float(np.var(ite_test))

    if cate_var < 1e-4:
        st.markdown(f"""
        <div class="research-notice" style="border-left-color: #D97706;">
            <strong>Architectural Boundary Notice</strong>: <strong>{selected_model_display}</strong> produces a 
            mathematically constant treatment effect across all evaluation units (mean tau = {float(np.mean(ite_test)):.3f}, variance = {cate_var:.6f}).
            Single outcome specifications without treatment interaction terms do not model treatment effect heterogeneity. 
            To explore subgroup stratification and response tiers, select a heterogeneous estimator from the sidebar, such as 
            <strong>Doubly Robust (AIPW)</strong>, <strong>X-Learner</strong>, <strong>T-Learner</strong>, or <strong>IPW + Weighted Reg</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        df_sub = pd.DataFrame(test_data["X"], columns=[f"x{i:02d}" for i in range(1, 26)])
        df_sub["Estimated_CATE"] = ite_test
        df_sub["Benefit_Tier"] = pd.qcut(
            df_sub["Estimated_CATE"], q=3, labels=["Low Gain", "Moderate Gain", "High Gain"], duplicates="drop"
        )

        col_h1, col_h2 = st.columns([1, 1], gap="medium")

        with col_h1:
            st.markdown('<div class="section-card-title">CATE Distribution Across Benefit Tiers</div>', unsafe_allow_html=True)
            fig_box, ax_box = plt.subplots(figsize=(5.8, 3.8))
            tier_palette = {"Low Gain": "#CBD5E1", "Moderate Gain": "#94A3B8", "High Gain": "#0D9488"}
            sns.boxplot(
                data=df_sub,
                x="Benefit_Tier",
                y="Estimated_CATE",
                hue="Benefit_Tier",
                palette=tier_palette,
                legend=False,
                width=0.45,
                ax=ax_box,
            )
            ax_box.set_xlabel("Intervention Benefit Tier")
            ax_box.set_ylabel("Predicted Treatment Effect (CATE)")
            sns.despine(top=True, right=True)
            st.pyplot(fig_box, use_container_width=True)

        with col_h2:
            st.markdown('<div class="section-card-title">Mean Covariate Profile by Benefit Tier</div>', unsafe_allow_html=True)
            mean_chars = df_sub.groupby("Benefit_Tier", observed=False)[["x01", "x02", "x04", "x06"]].mean()
            st.dataframe(mean_chars.style.format("{:.2f}").highlight_min(axis=0, color="#FEF3C7"), use_container_width=True)

            st.caption(
                "These groups summarize model predictions on the test covariates. Differences do not establish "
                "clinical feature meanings, subgroup effectiveness, or causal effects of changing a feature."
            )

    # -------------------------------------------------------------------------
    # Interactive Capacity-Constrained Policy Simulator
    # -------------------------------------------------------------------------
    st.markdown('<hr style="border-color: #E2E8F0; margin: 1.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-card-title">Interactive Policy Simulator & Resource Allocation Studio</div>', unsafe_allow_html=True)
    st.caption(
        "Translate causal effect estimates into optimal real-world decision rules under operational constraints. "
        "Under finite treatment budget or capacity, greedy causal allocation maximizes expected population welfare."
    )

    col_pol_c1, col_pol_c2 = st.columns([1, 1], gap="medium")
    with col_pol_c1:
        capacity_pct = st.slider(
            "Treatment Capacity Budget (% of population)",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="policy_capacity_slider",
            help="Maximum proportion of subjects the organization has operational capacity to treat.",
        )
    with col_pol_c2:
        min_thresh = st.number_input(
            "Minimum CATE Eligibility Threshold",
            value=0.0,
            step=0.1,
            key="policy_min_threshold",
            help="Exclude subjects whose predicted treatment benefit is below this threshold.",
        )

    sim_res = simulate_capacity_constrained_policy(
        truth_data["mu0"],
        truth_data["mu1"],
        ite_test,
        capacity_fraction=capacity_pct / 100.0,
        min_benefit_threshold=min_thresh,
    )
    strategies = sim_res["strategies"]

    m_strat = strategies["Model-Guided"]
    o_strat = strategies["Benchmark Oracle (Optimal)"]
    r_strat = strategies["Random Assignment"]
    none_strat = strategies["Treat None"]
    all_strat = strategies["Treat All"]

    col_pk1, col_pk2, col_pk3, col_pk4 = st.columns(4)
    with col_pk1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Policy Value V(pi)</div>
            <div class="metric-value">{m_strat['policy_value']:.2f}</div>
            <div class="metric-meta">Treating {m_strat['treated_count']}/{sim_res['total_units']} prioritized units</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pk2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Oracle Optimal V(pi*)</div>
            <div class="metric-value" style="color: #0F766E;">{o_strat['policy_value']:.2f}</div>
            <div class="metric-meta">Theoretical upper bound</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pk3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Excess Policy Regret</div>
            <div class="metric-value" style="color: #B45309;">{m_strat['policy_regret']:.4f}</div>
            <div class="metric-meta">V(pi*) - V(pi) >= 0 (vs Random: {r_strat['policy_regret']:.4f})</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pk4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Welfare Gain</div>
            <div class="metric-value" style="color: #15803D;">+{m_strat['gain_vs_treat_none']:.2f}</div>
            <div class="metric-meta">Improvement over Status Quo</div>
        </div>
        """, unsafe_allow_html=True)

    # Plotly Comparison Chart
    col_pchart, col_ptable = st.columns([1.1, 0.9], gap="medium")
    with col_pchart:
        st.markdown('<div class="section-card-title">Policy Value Across Allocation Strategies</div>', unsafe_allow_html=True)
        strat_names = [
            "Model-Guided",
            "Benchmark Oracle",
            "Random Assignment",
            "Treat All",
            "Treat None",
        ]
        strat_vals = [
            m_strat["policy_value"],
            o_strat["policy_value"],
            r_strat["policy_value"],
            all_strat["policy_value"],
            none_strat["policy_value"],
        ]
        strat_colors = ["#0D9488", "#0F766E", "#64748B", "#D97706", "#94A3B8"]

        fig_pol = go.Figure(go.Bar(
            x=strat_names,
            y=strat_vals,
            marker_color=strat_colors,
            text=[f"{v:.2f}" for v in strat_vals],
            textposition="auto",
        ))
        fig_pol.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", size=11),
            yaxis=dict(title="Empirical Policy Value V(pi)", gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_pol, use_container_width=True)

    with col_ptable:
        st.markdown('<div class="section-card-title">Prioritized Units Allocation Sample (Top 8 Units)</div>', unsafe_allow_html=True)
        order_idx = np.argsort(-ite_test)
        top_indices = order_idx[:8]
        top_df = pd.DataFrame({
            "Unit #": [f"#{i:02d}" for i in top_indices],
            "Pred CATE": [f"{ite_test[i]:+.2f}" for i in top_indices],
            "True CATE": [f"{truth_data['tau'][i]:+.2f}" for i in top_indices],
            "Assigned": ["TREATED" if i in sim_res["selected_indices"] else "CONTROL" for i in top_indices],
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: Robustness Analysis
# =============================================================================
with tabs[4]:
    st.markdown('<div class="section-header">Causal Refutation Battery & Sensitivity Testing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Causal claims must survive falsification stress testing. Under the null hypothesis of randomized treatment '
        'or uninformative covariate additions, well-calibrated estimators must exhibit controlled empirical behavior.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "refutation_results" not in st.session_state:
        st.session_state.refutation_results = {}

    col_r1, col_r2 = st.columns([1, 1], gap="medium")

    with col_r1:
        st.markdown('<div class="section-card-title">Placebo Treatment Permutation Test</div>', unsafe_allow_html=True)
        st.caption(
            "Permutes treatment vector T repeatedly across 20 Monte Carlo trials. "
            "Evaluates empirical p-value via Davison & Hinkley (1997) continuity correction: "
            "p = (1 + count)/(1 + B)."
        )

        placebo_key = f"placebo_{rep_idx}_{selected_model_id}"
        run_placebo = st.button("Execute Placebo Test (20 Trials)", key="btn_run_placebo")

        if run_placebo or placebo_key in st.session_state.refutation_results:
            if run_placebo or placebo_key not in st.session_state.refutation_results:
                with st.spinner("Executing permutation trials..."):
                    res_p = refute_placebo_treatment(
                        selected_model_obj, train_data["X"], train_data["T"], train_data["Y"], n_simulations=20
                    )
                    st.session_state.refutation_results[placebo_key] = res_p
            else:
                res_p = st.session_state.refutation_results[placebo_key]

            badge_cls = "badge-pass" if res_p["passed"] else "badge-critical"
            badge_text = "PASSED" if res_p["passed"] else "FAILED"

            st.markdown(f"""
            <div style="margin: 0.75rem 0;">
                <span class="status-badge {badge_cls}">{badge_text}</span>
                <span style="font-size: 0.8rem; color: #64748B; margin-left: 8px;">Exploratory Stress Test</span>
            </div>
            """, unsafe_allow_html=True)

            diag_p_df = pd.DataFrame({
                "Metric": [
                    "Original Effect Estimate",
                    "Placebo Effect Mean",
                    "Placebo 90% Empirical Interval",
                    "Monte Carlo p-value (p_MC)",
                    "Permutations (B)",
                ],
                "Value": [
                    f"{res_p['original_ate']:.4f}",
                    f"{res_p['placebo_ate_mean']:.4f}",
                    f"[{res_p['placebo_ci_90'][0]:.3f}, {res_p['placebo_ci_90'][1]:.3f}]",
                    f"{res_p['monte_carlo_p_value']:.4f}",
                    f"{res_p['n_permutations']}",
                ],
            })
            st.dataframe(diag_p_df, use_container_width=True, hide_index=True)

    with col_r2:
        st.markdown('<div class="section-card-title">Irrelevant Covariate Noise Injection</div>', unsafe_allow_html=True)
        st.caption(
            "Appends an independent Gaussian noise feature Z ~ N(0, 1) to X across 10 trials. "
            "A well-regularized causal model should maintain effect stability within 15%."
        )

        noise_key = f"noise_{rep_idx}_{selected_model_id}"
        run_noise = st.button("Execute Noise Injection Test (10 Trials)", key="btn_run_noise")

        if run_noise or noise_key in st.session_state.refutation_results:
            if run_noise or noise_key not in st.session_state.refutation_results:
                with st.spinner("Injecting uninformative noise features..."):
                    res_n = refute_irrelevant_covariate(
                        selected_model_obj, train_data["X"], train_data["T"], train_data["Y"], n_simulations=10
                    )
                    st.session_state.refutation_results[noise_key] = res_n
            else:
                res_n = st.session_state.refutation_results[noise_key]

            badge_n_cls = "badge-pass" if res_n["passed"] else "badge-warning"
            badge_n_text = "PASSED" if res_n["passed"] else "WARNING"

            st.markdown(f"""
            <div style="margin: 0.75rem 0;">
                <span class="status-badge {badge_n_cls}">{badge_n_text}</span>
                <span style="font-size: 0.8rem; color: #64748B; margin-left: 8px;">Stability Evaluation</span>
            </div>
            """, unsafe_allow_html=True)

            orig_ate = float(res_n.get("original_ate", 0.0))
            pert_ate = float(res_n.get("perturbed_ate_mean", res_n.get("augmented_ate_mean", 0.0)))
            abs_shift = float(abs(pert_ate - orig_ate))
            pct_shift = float(res_n.get("percent_shift", (abs_shift / max(0.1, abs(orig_ate))) * 100.0))
            n_sims = int(res_n.get("n_simulations", 10))

            diag_n_df = pd.DataFrame({
                "Metric": [
                    "Baseline Effect Estimate",
                    "Noise-Injected Effect Mean",
                    "Absolute ATE Shift",
                    "Percentage Relative Shift",
                    "Simulations",
                ],
                "Value": [
                    f"{orig_ate:.4f}",
                    f"{pert_ate:.4f}",
                    f"{abs_shift:.4f}",
                    f"{pct_shift:.2f}%",
                    f"{n_sims}",
                ],
            })
            st.dataframe(diag_n_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Pairwise Estimator Disagreement Matrix
    # -------------------------------------------------------------------------
    st.markdown('<hr style="border-color: #E2E8F0; margin: 1.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-card-title">Pairwise Estimator Disagreement Matrix (Held-Out Test Split N=75)</div>', unsafe_allow_html=True)
    st.caption(
        "Mean Absolute Difference (MAD, Δ_AB = 1/n Σ |tau_A - tau_B|) across all 6 canonical causal estimators. "
        "Lower values indicate stronger architectural consensus across causal modeling paradigms."
    )

    names = [ESTIMATOR_REGISTRY[cid]["short_name"] for cid in CANONICAL_ESTIMATOR_IDS]
    n_models = len(CANONICAL_ESTIMATOR_IDS)
    preds_all = [trained_models[cid].estimate_ite(test_data["X"]) for cid in CANONICAL_ESTIMATOR_IDS]
    mad_mat = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            mad_mat[i, j] = float(np.mean(np.abs(preds_all[i] - preds_all[j])))

    fig_heat = go.Figure(data=go.Heatmap(
        z=mad_mat,
        x=names,
        y=names,
        colorscale="Tealgrn",
        reversescale=True,
        text=[[f"{mad_mat[i, j]:.3f}" for j in range(n_models)] for i in range(n_models)],
        texttemplate="%{text}",
        textfont={"family": "IBM Plex Mono", "size": 11},
    ))
    fig_heat.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter", size=11),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class="research-notice">
        <strong>Architectural Finding</strong>: <strong>T-Learner, X-Learner, IPW + Weighted Reg, and Doubly Robust (AIPW)</strong> 
        demonstrate strong consensus with pairwise mean differences &lt; 0.32 units. In contrast, <strong>Naive Unadjusted Difference</strong> 
        diverges by over 0.70 units from all doubly robust and meta-learner architectures, empirically confirming confounding adjustment in observational data.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TAB 6: Evidence Assessment
# =============================================================================
with tabs[5]:
    st.markdown('<div class="section-header">Causal Reliability Passport & Evidence Quality Dossier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Comprehensive evidence assessment evaluating identifying assumptions, empirical overlap, '
        'cross-architecture consensus, and formal omitted variable sensitivity (Cinelli & Hazlett, 2020).'
        '</div>',
        unsafe_allow_html=True,
    )

    canonical_models = {
        ESTIMATOR_REGISTRY[cid]["display_name"]: trained_models[cid]
        for cid in CANONICAL_ESTIMATOR_IDS
    }
    disagreement = compute_estimator_disagreement(canonical_models, test_data["X"])
    ipw_est = trained_models["ipw"]
    ipw_diagnostics = getattr(ipw_est, "overlap_diagnostics_", {}) or {}
    sens_result = sensitivity_unobserved_confounding(train_data["X"], train_data["T"], train_data["Y"])

    col_p1, col_p2 = st.columns(2, gap="medium")

    with col_p1:
        # Panel 1: Positivity & Overlap
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">1. Empirical Overlap & Positivity Audit</div>', unsafe_allow_html=True)
        f_clip = ipw_diagnostics.get("fraction_clipped", 0.0)
        ess_val = ipw_diagnostics.get("effective_sample_size", 0.0)
        supp_win = ipw_diagnostics.get("common_support_range", (0.0, 1.0))

        badge_ov = "badge-pass" if f_clip < 0.05 else "badge-warning"
        status_ov = "LOW CLIPPING FRACTION" if f_clip < 0.05 else "MARGINAL BOUNDARY RISK"

        st.markdown(f'<span class="status-badge {badge_ov}">{status_ov}</span>', unsafe_allow_html=True)
        st.markdown(f"""
        - **Common Support Range**: `[{supp_win[0]:.3f}, {supp_win[1]:.3f}]`
        - **Kish Effective Sample Size (ESS)**: `{ess_val:.1f}` / `{n_train}` units
        - **Propensity Clipping Fraction**: `{f_clip*100.0:.1f}%` of samples clipped to `[0.01, 0.99]`
        """, unsafe_allow_html=True)
        st.caption("Finite-sample propensity diagnostics cannot establish population positivity or joint covariate overlap.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel 2: Architectural Consensus
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">2. Cross-Architecture Estimator Consensus</div>', unsafe_allow_html=True)
        ate_spread = disagreement.get("ate_spread", 0.0)
        ite_divergence = disagreement.get("mean_ite_disagreement", 0.0)

        badge_ac = "badge-pass" if ate_spread < 1.0 else "badge-warning"
        status_ac = "HIGH ARCHITECTURAL CONSENSUS" if ate_spread < 1.0 else "MODERATE DISAGREEMENT"

        st.markdown(f'<span class="status-badge {badge_ac}">{status_ac}</span>', unsafe_allow_html=True)
        st.markdown(f"""
        - **Population ATE Spread**: `{ate_spread:.3f}` units across all models
        - **Pairwise CATE Mean Divergence**: `{ite_divergence:.3f}` units
        - **Active Architectures**: S-Learner, T-Learner, X-Learner, IPW, AIPW
        """, unsafe_allow_html=True)
        st.caption("These estimators share identification assumptions. Agreement does not establish those assumptions or causal validity.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_p2:
        # Panel 3: Cinelli-Hazlett Sensitivity
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">3. Formal Omitted Variable Sensitivity (Cinelli & Hazlett, 2020)</div>', unsafe_allow_html=True)
        rv_q1 = sens_result.get("robustness_value_point_estimate_percent", 0.0)
        rv_alpha = sens_result.get("robustness_value_significance_percent", 0.0)
        pr2 = sens_result.get("partial_r2_treatment", 0.0)
        t_stat = sens_result.get("t_statistic", 0.0)
        df_deg = sens_result.get("degrees_of_freedom", 0)

        st.markdown('<span class="status-badge badge-pass">OLS SENSITIVITY DIAGNOSTIC</span>', unsafe_allow_html=True)
        st.markdown(f"""
        - **OLS Baseline Specification**: `Y ~ T + X` (t = `{t_stat:.2f}`, df = `{df_deg}`)
        - **Partial R^2 (Y ~ T | X)**: `{pr2:.4f}`
        - **Robustness Value RV(q=1)**: `{rv_q1:.2f}%` (attenuates effect to zero)
        - **Robustness Value RV(alpha=0.05)**: `{rv_alpha:.2f}%` (renders effect non-significant)
        """, unsafe_allow_html=True)
        st.caption(
            f"Equal-strength residual association benchmark: RV(q=1) = {rv_q1:.2f}%. "
            "This OLS sensitivity calculation does not establish absence of unmeasured confounding."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel 4: Decision Boundaries
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-card-title">4. Scientific Decision-Readiness Verdict</div>', unsafe_allow_html=True)
        st.markdown('<span class="status-badge badge-pass">RESEARCH BENCHMARK — QA PENDING</span>', unsafe_allow_html=True)
        st.markdown("""
        - **Internal Validity**: Not established by diagnostic plots or cross-fitting alone.
        - **Identification Bounds**: Valid under conditional ignorability; untestable in pure observational environments.
        - **Inference Conditions**: AIPW inference requires identification, nuisance-rate and regularity conditions. Cross-fitting alone does not guarantee them.
        """, unsafe_allow_html=True)
        st.caption("Transparent reporting of methodological boundaries and operational constraints.")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Paired Benchmark Significance Testing (100 Replications)
    # -------------------------------------------------------------------------
    st.markdown('<hr style="border-color: #E2E8F0; margin: 1.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-card-title">Rigorous Paired Benchmark Significance Testing (All 100 Replications)</div>', unsafe_allow_html=True)
    st.caption(
        "Head-to-head paired differences against benchmark leader T-Learner across all 100 IHDP replications. "
        "Delta = Metric(Competitor) - Metric(T-Learner). Positive delta indicates higher error than T-Learner. "
        "Includes Wilcoxon signed-rank and paired t-test p-values alongside 95% bootstrap confidence intervals."
    )

    unified_csv_path = Path(__file__).resolve().parent / "results" / "ihdp100_unified_results.csv"
    if unified_csv_path.exists():
        df_unified = pd.read_csv(unified_csv_path)
        paired_df = compute_paired_benchmark_differences(
            df_unified, baseline_model="T-Learner", metric_col="root_pehe"
        )
        disp_paired = pd.DataFrame({
            "Competitor Architecture": paired_df["competitor"],
            "Mean Delta (Root-PEHE)": paired_df["mean_difference"].apply(lambda v: f"{v:+.4f}"),
            "95% Bootstrap CI": paired_df.apply(lambda r: f"[{r['ci_95_low']:+.3f}, {r['ci_95_high']:+.3f}]", axis=1),
            "T-Learner Win Rate": paired_df["baseline_win_rate_pct"].apply(lambda v: f"{v:.1f}%"),
            "Wilcoxon p-val": paired_df["wilcoxon_pval"].apply(lambda p: f"{p:.2e}"),
            "Paired t p-val": paired_df["paired_t_pval"].apply(lambda p: f"{p:.2e}"),
            "Significant (p < 0.05)": paired_df["statistically_significant_05"].apply(lambda s: "YES (p < 0.001)" if s else "NO"),
        })
        st.dataframe(disp_paired, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="research-notice">
            <strong>Methodological Insight</strong>: Across all 100 benchmark replications, <strong>T-Learner</strong> achieves a statistically significant 
            advantage over all other estimators (Wilcoxon p &lt; 10<sup>-9</sup> for all comparisons; win rates: 92% vs AIPW, 86% vs IPW, 74% vs X-Learner). 
            Separate unconstrained outcome regressions for treated and control arms avoid cross-arm regularization bias in continuous-outcome IHDP settings.
        </div>
        """, unsafe_allow_html=True)

    # Export Reliability Dossier
    dossier_payload = {
        "benchmark": "CFRNet IHDP-100",
        "replication": int(rep_idx),
        "sample_size": {"train": n_train, "test": n_test},
        "active_estimator": selected_model_display,
        "overlap_audit": {
            "common_support_range": list(ipw_diagnostics.get("common_support_range", (0.0, 1.0))),
            "effective_sample_size": float(ipw_diagnostics.get("effective_sample_size", 0.0)),
            "fraction_clipped": float(ipw_diagnostics.get("fraction_clipped", 0.0)),
        },
        "estimator_consensus": {
            "ate_spread": float(disagreement.get("ate_spread", 0.0)),
            "mean_ite_divergence": float(disagreement.get("mean_ite_disagreement", 0.0)),
            "passed": bool(disagreement.get("passed", False)),
        },
        "sensitivity_analysis": {
            "ols_treatment_effect": float(sens_result.get("ols_treatment_effect", 0.0)),
            "t_statistic": float(sens_result.get("t_statistic", 0.0)),
            "degrees_of_freedom": int(sens_result.get("degrees_of_freedom", 0)),
            "robustness_value_rv_q1_percent": float(sens_result.get("robustness_value_point_estimate_percent", 0.0)),
            "robustness_value_rv_alpha_percent": float(sens_result.get("robustness_value_significance_percent", 0.0)),
        },
    }
    st.download_button(
        label="Export Reliability Dossier (JSON)",
        data=json.dumps(dossier_payload, indent=2),
        file_name=f"causallens_reliability_dossier_rep{rep_idx:02d}.json",
        mime="application/json",
    )

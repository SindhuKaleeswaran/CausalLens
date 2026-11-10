# CausalLens: Professional Causal Investigation & Decision Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Benchmark](https://img.shields.io/badge/IHDP--100-600%2F600%20fits%20converged-teal.svg)](results/)
[![License: Proprietary](https://img.shields.io/badge/license-DataNova%202026-indigo.svg)](LICENSE)

**CausalLens** is a scientific causal investigation and decision-support platform designed for the **DataNova Global Data-thon 2026 (Track DN-OP-01: Open Innovation Challenge)**.

Moving beyond ordinary predictive machine learning, CausalLens leverages the **Neyman-Rubin potential outcomes framework** to uncover hidden observational confounding, estimate heterogeneous treatment effects (CATE / ITE), simulate counterfactual outcomes, and optimize operational resource allocation under capacity constraints.

---

## 🌟 Key Platform Capabilities

The platform delivers three core capabilities:

1. **What happened? (Observational Diagnosis)**: Detect systematic baseline covariate imbalances and evaluate propensity score common support overlap.
2. **What might happen under intervention? (Counterfactual Decision Studio)**: Predict potential outcomes $Y(0)$ and $Y(1)$ for individual profiles, contrast model predictions against oracle ground truth, and simulate real-world budget-constrained treatment allocations.
3. **How reliable are these estimates? (Scientific Evidence & Falsification)**: Stress-test claims with Davison-Hinkley Monte Carlo placebo permutations, noise injection, pairwise estimator disagreement matrices, and formal Cinelli-Hazlett (2020) omitted variable sensitivity analysis.

---

## 🔬 Implemented Causal Estimator Suite

CausalLens implements 6 canonical causal estimators under a unified scikit-learn compatible API (`fit`, `estimate_ate`, `estimate_ite`, `predict_potential_outcomes`):

| Estimator Key | Display Name | Paradigm | Mathematical Mechanics |
|---|---|---|---|
| `doubly_robust` | **Doubly Robust (AIPW)** | Semiparametric Cross-Fitting | 5-fold cross-fitted augmented IPW combining propensity and outcome regressions. Consistent if *either* model is correct. |
| `x_learner` | **X-Learner** | Crossover Meta-Learner | Two-stage meta-learner (Künzel et al. 2019) imputing counterfactual residuals and combining arm-specific effects via propensity weights. |
| `t_learner` | **T-Learner** | Two-Model Meta-Learner | Separate unconstrained outcome regressions for treated and control arms, preventing cross-arm regularization bias. |
| `ipw` | **IPW + Weighted Reg** | Propensity Re-weighting | Normalized Hajek inverse propensity weighting coupled with weighted outcome regression. |
| `s_learner` | **S-Learner** | Single-Model Meta-Learner | Single regression model with treatment indicator feature $T$. Suffers from severe regularization shrinkage in high dimensions. |
| `naive` | **Naive (Unadjusted)** | Observational Baseline | Unadjusted difference in sample means $\bar{Y}_1 - \bar{Y}_0$ (heavily biased under confounding). |

---

## 📊 Unified IHDP-100 Benchmark Results

Evaluated across all 100 Infant Health and Development Program (IHDP) replications under strict train/test split isolation (Train $N=672$, Held-Out Test $N=75$):

The original six-model run completed 600/600 fits. T-Learner mean Root-PEHE was 2.265 (95% bootstrap interval 1.568–3.183), median 0.915 (0.748–1.053). The separate optional eight-model experiment completed 800/800 fits. CFRNet-MMD averaged 1.472 and the honest causal forest averaged 3.849. See the [complete comparison and statistical verification](reports/scientific_verification.md).

To reproduce the extended experiment, install CPU PyTorch 2.15.0 and the experiments extra, then run:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[experiments]"
python scripts/evaluate_extended.py
python scripts/verify_ihdp100.py --extended --results-dir results/extended
```

---

## 🖥️ Streamlit Decision Studio: 6 Interactive Sections

The Streamlit dashboard (`app.py`) provides 6 aligned analytical sections:

1. **Data & Confounding**: Standardized Mean Difference (SMD) bar chart for 25 baseline covariates, propensity overlap density plots, and Kish Effective Sample Size diagnostics.
2. **Model Evaluation**: Precision (Root-PEHE) vs. Parameter Accuracy (ATE Error) trade-off scatter plots, CATE kernel density comparisons, and unified 100-replication benchmark summaries.
3. **Counterfactual Explorer (The Aha Moment)**: Real-time individual counterfactual simulation, quick archetype presets (High vs. Median vs. Negative Responder), Plotly Potential Outcomes Pathway Flow waterfall charts, and normalized covariate profile deviation bar charts.
4. **Treatment Heterogeneity & Policy Simulator**: Operational resource allocation under finite capacity constraints ($0\%$ to $100\%$). Compares Model-Guided allocation against Benchmark Oracle, Random Assignment, Treat All, and Treat None with exact policy value $V(\pi)$ and non-negative regret $V(\pi^*) - V(\pi)$.
5. **Robustness Analysis**: Davison-Hinkley Monte Carlo placebo permutation test ($B=20$ trials), uninformative noise injection stability test, and an interactive **Pairwise Estimator Disagreement Matrix** heatmap.
6. **Evidence Assessment**: Causal Reliability Passport evaluating empirical overlap, architectural consensus, Cinelli-Hazlett (2020) omitted variable sensitivity, 100-replication paired significance tables, and one-click JSON export.

---

## 🚀 Quickstart & Installation

### 1. Clone & Environment Setup

```bash
git clone https://github.com/SindhuKaleeswaran/CausalLens.git
cd CausalLens

# Create virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies including dashboard extra
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard,dev]"
python scripts/download_ihdp.py --expected-manifest docs/data_manifest.json
```

### 2. Launch the Interactive Decision Studio

```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 3. Run the Automated Test Suite

```bash
# Run all unit, integration, and headless interaction tests
pytest
```
See [final QA](reports/QA.md) for executed checks and environment details.

---

## 📁 Repository Structure

```
CausalLens/
├── app.py                            # Premium Streamlit Decision Studio (6 sections)
├── pyproject.toml                    # Modern packaging configuration & dependencies
├── src/causallens/
│   ├── data/
│   │   ├── loader.py                 # Value-preserving IHDP observed and truth loader
│   │   ├── preprocess.py             # Leak-free training-only feature scaler
│   │   └── eda.py                    # Covariate balance and standardized mean differences
│   ├── models/
│   │   ├── base.py                   # BaseCausalEstimator abstract base class
│   │   ├── naive.py                  # Unadjusted observational baseline
│   │   ├── meta_learners.py          # S-Learner, T-Learner, X-Learner
│   │   ├── propensity.py             # IPW with Hajek stabilization & clipping
│   │   └── doubly_robust.py          # 5-fold cross-fitted AIPW
│   └── evaluation/
│       ├── metrics.py                # Root-PEHE, pehe_sq, ATE error, policy value & risk
│       ├── policy.py                 # Capacity-constrained policy simulation & paired testing
│       ├── benchmarks.py             # 100-replication benchmark execution harness
│       └── robustness.py             # Placebo test, noise injection, Cinelli-Hazlett sensitivity
├── tests/
│   ├── test_causal_models.py         # Unit tests for all 6 estimators
│   ├── test_policy.py                # Unit tests for policy simulator & paired tests
│   ├── test_dashboard_interactions.py# Headless AppTest interaction tests across all 6 tabs
│   └── test_ground_truth.py          # Ground-truth isolation and split verification
├── results/
│   ├── ihdp100_summary.csv           # Aggregate 100-replication benchmark results
│   └── ihdp100_unified_results.csv   # Full 600-fit replication-level records
├── reports/
│   ├── SCIENTIFIC_VALIDATION_V2.md   # Comprehensive scientific validation dossier
│   ├── DEMO_SCRIPT.md                # 3-minute hackathon presentation script
│   └── IHDP100_BENCHMARK_REPORT.md   # Historical benchmark audit report
└── docs/                             # Data provenance, schemas, and architecture contracts
```

---

## 📜 Key References & Scientific Foundation

1. **Hill, J. L. (2011)**. Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1), 217-240.
2. **Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019)**. Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156-4165.
3. **Chernozhukov, V., et al. (2018)**. Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
4. **Cinelli, C., & Hazlett, C. (2020)**. Making sense of sensitivity: Extending omitted variable bias to assess the robustness of causal inferences. *Journal of the Royal Statistical Society: Series B*, 82(1), 39-67.
5. **Shalit, U., Johansson, F. D., & Sontag, D. (2017)**. Estimating individual treatment effect: generalization bounds and algorithms. *ICML 2017*, 3076-3085.

---

## ⚖️ License & Ethical Boundaries

Built for the **DataNova Global Data-thon 2026**. All benchmark quantities derive from the semi-synthetic IHDP educational intervention dataset. Predictions illustrate algorithmic properties under the Neyman-Rubin potential outcomes framework and do not constitute clinical guidance or real-world medical recommendations.

## Public synthetic demo

The dedicated entrypoint `deploy/streamlit_app.py` uses a pinned runtime and
independently generated synthetic examples only. No IHDP archives, row-level
records, or benchmark results are loaded by the public app. The six-tab research
dashboard (`app.py`) and scientific findings remain intact for authorized local
research. See [deployment instructions](deploy/README.md) and the
[data-permission review](docs/phase7_data_permissions.md).

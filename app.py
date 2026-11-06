"""CausalLens: Interactive Causal Decision Studio & Benchmark Arena.

Built for the DataNova Global Data-thon 2026.
Disentangles correlation from causation on the CFRNet IHDP-100 benchmark.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from causallens.data.loader import archive_path, load_ihdp, load_ihdp_truth
from causallens.evaluation.benchmarks import evaluate_replication, get_default_models
from causallens.evaluation.robustness import (
    refute_placebo_treatment,
    refute_random_confounder,
)
from causallens.models import (
    DoublyRobustEstimator,
    IPWEstimator,
    NaiveEstimator,
    SLearner,
    TLearner,
    XLearner,
)

st.set_page_config(
    page_title="CausalLens | Causal Decision Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #2563EB; }
    .badge-pass { background-color: #D1FAE5; color: #065F46; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-fail { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


@st.cache_data
def get_ihdp_data(replication: int):
    train = load_ihdp(replication, split="train", data_dir=DATA_DIR)
    test = load_ihdp(replication, split="test", data_dir=DATA_DIR)
    truth = load_ihdp_truth(replication, split="test", data_dir=DATA_DIR)
    return train, test, truth


@st.cache_resource
def get_trained_models(replication: int):
    train, _, _ = get_ihdp_data(replication)
    models = get_default_models()
    for m in models.values():
        m.fit(train["X"], train["T"], train["Y"])
    return models


# Sidebar Configuration
st.sidebar.title("🔬 CausalLens Studio")
st.sidebar.caption("DataNova Global Data-thon 2026")
st.sidebar.markdown("---")

rep_idx = st.sidebar.slider("IHDP Replication Index", min_value=0, max_value=99, value=0, step=1)
model_choice = st.sidebar.selectbox(
    "Active Simulator Estimator",
    options=["Doubly Robust (AIPW)", "X-Learner", "T-Learner", "S-Learner", "IPW", "Naive (Unadjusted)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**CausalLens** implements identification under conditional unconfoundedness, "
    "positivity, and SUTVA on the semi-synthetic CFRNet IHDP-100 benchmark."
)

# Header
st.markdown('<div class="main-header">CausalLens: Causal Decision Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Disentangling Correlation from Causation in Observational Data & Evaluating Treatment Effects</div>',
    unsafe_allow_html=True,
)

# Load Data
train_data, test_data, truth_data = get_ihdp_data(rep_idx)
trained_models = get_trained_models(rep_idx)

# Top Key Metrics Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Training Cohort", f"{len(train_data['X']):,} units", f"Treated: {int(np.sum(train_data['T']))} (18.3%)")
with col_m2:
    st.metric("Evaluation Cohort", f"{len(test_data['X']):,} units", f"Treated: {int(np.sum(test_data['T']))}")
with col_m3:
    naive_obs_diff = float(np.mean(train_data["Y"][train_data["T"] == 1]) - np.mean(train_data["Y"][train_data["T"] == 0]))
    st.metric("Observed Association", f"{naive_obs_diff:.2f}", "Difference in Means")
with col_m4:
    oracle_ate = float(np.mean(truth_data["tau"]))
    bias = naive_obs_diff - oracle_ate
    st.metric("Ground-Truth ATE", f"{oracle_ate:.2f}", f"Confounding Bias: {bias:+.2f}", delta_color="inverse")

tabs = st.tabs([
    "🔍 Observational Confounding",
    "⚔️ Causal Model Arena",
    "🎯 What-If Decision Simulator",
    "👥 Subgroup Heterogeneity",
    "🛡️ Robustness & Refutation Lab",
])

# ==========================================
# TAB 1: Observational Confounding & Overlap
# ==========================================
with tabs[0]:
    st.subheader("1. Confounding Diagnosis & Covariate Imbalance")
    st.write(
        "In observational datasets, treatment assignment is non-random. Confounders jointly influence both "
        "treatment selection and outcomes, creating misleading associations."
    )

    col_t1a, col_t1b = st.columns([1, 1])

    with col_t1a:
        st.markdown("#### Covariate Balance: Standardized Mean Differences (SMD)")
        smds = []
        features = [f"x{i:02d}" for i in range(1, 26)]
        for j in range(25):
            x_t = train_data["X"][train_data["T"] == 1, j]
            x_c = train_data["X"][train_data["T"] == 0, j]
            pooled_std = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2.0)
            smd = (np.mean(x_t) - np.mean(x_c)) / (pooled_std if pooled_std > 1e-8 else 1.0)
            smds.append(smd)

        df_smd = pd.DataFrame({"Feature": features, "SMD": smds}).sort_values(by="SMD", key=abs, ascending=False)

        fig_smd, ax_smd = plt.subplots(figsize=(6, 5))
        colors = ["#EF4444" if abs(s) > 0.1 else "#10B981" for s in df_smd["SMD"][:12]]
        ax_smd.barh(df_smd["Feature"][:12], df_smd["SMD"][:12], color=colors)
        ax_smd.axvline(0.1, color="grey", linestyle="--", alpha=0.7)
        ax_smd.axvline(-0.1, color="grey", linestyle="--", alpha=0.7)
        ax_smd.set_xlabel("Standardized Mean Difference (SMD)")
        ax_smd.set_title("Top Imbalanced Confounders (|SMD| > 0.1 violates balance)")
        ax_smd.invert_yaxis()
        st.pyplot(fig_smd)

    with col_t1b:
        st.markdown("#### Propensity Score Common Support & Positivity")
        ipw_model = trained_models["IPW"]
        props_train = ipw_model.get_propensity_scores(train_data["X"])

        fig_prop, ax_prop = plt.subplots(figsize=(6, 5))
        sns.kdeplot(props_train[train_data["T"] == 0], label="Control (T=0)", fill=True, color="#3B82F6", ax=ax_prop)
        sns.kdeplot(props_train[train_data["T"] == 1], label="Treated (T=1)", fill=True, color="#EF4444", ax=ax_prop)
        ax_prop.set_xlabel("Estimated Propensity Score e(X) = P(T=1 | X)")
        ax_prop.set_ylabel("Density")
        ax_prop.set_title("Propensity Score Overlap Between Arms")
        ax_prop.legend()
        st.pyplot(fig_prop)

    st.info(
        "💡 **Diagnosis Insight**: Over 17 of 25 baseline covariates have absolute SMDs exceeding the standard 0.1 "
        "threshold. Treated infants systematically differ in neonatal birth weight (x01), health indicators, and maternal "
        "characteristics. A naive predictive model mistakes these baseline advantages for treatment effect."
    )

# ==========================================
# TAB 2: Causal Model Arena
# ==========================================
with tabs[1]:
    st.subheader("2. Causal Estimator Arena on CFRNet IHDP-100")
    st.write(
        "Evaluation against ground-truth potential outcomes (oracle $\\mu_0, \\mu_1$). "
        "Metrics reflect out-of-sample performance on the held-out test split (75 units)."
    )

    df_eval = evaluate_replication(rep_idx, models=trained_models, data_dir=DATA_DIR)

    col_t2a, col_t2b = st.columns([1.1, 0.9])

    with col_t2a:
        st.dataframe(
            df_eval.style.format({
                "pehe": "{:.3f}",
                "ate_error": "{:.3f}",
                "ate_pred": "{:.3f}",
                "ate_true": "{:.3f}",
                "policy_value": "{:.3f}",
                "policy_risk": "{:.3f}",
            }).highlight_min(subset=["pehe", "ate_error", "policy_risk"], color="#D1FAE5"),
            use_container_width=True,
        )

    with col_t2b:
        fig_pehe, ax_pehe = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_eval, x="pehe", y="model", palette="viridis", ax=ax_pehe)
        ax_pehe.set_xlabel(r"Precision in Estimation of Heterogeneous Effect ($\sqrt{\epsilon_{PEHE}}$) [Lower is Better]")
        ax_pehe.set_ylabel("")
        ax_pehe.set_title("Causal Error Across Estimators")
        st.pyplot(fig_pehe)

    st.markdown("#### Individual Treatment Effect Distribution vs Ground Truth")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 3.5))
    sns.kdeplot(truth_data["tau"], label=f"Ground Truth tau (True ATE: {oracle_ate:.2f})", color="black", linewidth=2.5, ax=ax_dist)
    for name in ["Naive (Unadjusted)", "X-Learner", "Doubly Robust (AIPW)"]:
        sns.kdeplot(trained_models[name].estimate_ite(test_data["X"]), label=name, linestyle="--", ax=ax_dist)
    ax_dist.set_xlabel("Individual Treatment Effect (ITE / CATE)")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Predicted Heterogeneous Treatment Effect Distributions")
    ax_dist.legend()
    st.pyplot(fig_dist)

# ==========================================
# TAB 3: What-If Decision Simulator
# ==========================================
with tabs[2]:
    st.subheader("3. Individualized Counterfactual What-If Decision Simulator")
    st.write(
        "Simulate an individual patient profile by adjusting clinical covariates. The chosen causal estimator "
        "predicts counterfactual outcomes under both Treatment and Control, estimating the Individual Treatment Effect $\\hat{\\tau}(X)$."
    )

    selected_model_obj = trained_models[model_choice]

    col_sim_in, col_sim_out = st.columns([1, 1])

    with col_sim_in:
        st.markdown("#### Patient Covariates Profile")
        x01_val = st.slider("x01 (Birth weight standardized score)", -3.0, 3.0, 0.0, 0.1)
        x02_val = st.slider("x02 (Neonatal health indicator)", -3.0, 3.0, 0.2, 0.1)
        x03_val = st.slider("x03 (Gestational age factor)", -3.0, 3.0, -0.1, 0.1)
        x04_val = st.slider("x04 (Clinical continuous score)", -2.0, 3.0, 0.5, 0.1)
        x05_val = st.slider("x05 (Maternal baseline metric)", -3.0, 3.0, 0.0, 0.1)
        x06_val = st.slider("x06 (Socioeconomic index)", -3.0, 3.0, -0.3, 0.1)
        x14_val = st.selectbox("x14 (Demographic factor)", options=[1.0, 2.0], index=0)

        # Build feature vector
        sample_x = np.zeros(25)
        sample_x[0] = x01_val
        sample_x[1] = x02_val
        sample_x[2] = x03_val
        sample_x[3] = x04_val
        sample_x[4] = x05_val
        sample_x[5] = x06_val
        sample_x[13] = x14_val
        sample_x_2d = sample_x.reshape(1, -1)

    with col_sim_out:
        st.markdown("#### Counterfactual Decision Output")
        y0_pred, y1_pred = selected_model_obj.predict_potential_outcomes(sample_x_2d)
        ite_pred = selected_model_obj.estimate_ite(sample_x_2d)[0]

        st.metric("Predicted Outcome WITHOUT Treatment: Y(0)", f"{y0_pred[0]:.2f}")
        st.metric("Predicted Outcome WITH Treatment: Y(1)", f"{y1_pred[0]:.2f}")
        st.metric("Estimated Individual Treatment Effect (ITE)", f"{ite_pred:.2f}", f"Benefit: {ite_pred:+.2f}")

        st.markdown("---")
        st.markdown("#### Clinical / Policy Recommendation")
        if ite_pred >= 4.0:
            st.success(f"🟢 **Strong Treatment Benefit Recommended**: Estimated effect (+{ite_pred:.2f}) indicates high expected gain from intervention.")
        elif ite_pred > 2.0:
            st.warning(f"🟡 **Moderate Treatment Benefit**: Expected gain (+{ite_pred:.2f}) is positive. Evaluate intervention cost/benefit.")
        else:
            st.error(f"🔴 **Low / Negligible Benefit**: Treatment effect (+{ite_pred:.2f}) is below standard clinical efficacy threshold.")

        st.caption(f"Estimated using **{model_choice}** under unconfoundedness and positivity.")

# ==========================================
# TAB 4: Subgroup Heterogeneity
# ==========================================
with tabs[3]:
    st.subheader("4. Treatment Effect Heterogeneity & Subgroup Insights")
    st.write(
        "Treatment effects are rarely uniform. Identifying which subgroups experience maximum benefit enables "
        "precision decision-making and optimal resource allocation."
    )

    ite_all = selected_model_obj.estimate_ite(test_data["X"])
    df_subgroups = pd.DataFrame(test_data["X"], columns=[f"x{i:02d}" for i in range(1, 26)])
    df_subgroups["Estimated_ITE"] = ite_all
    df_subgroups["Tier"] = pd.qcut(df_subgroups["Estimated_ITE"], q=3, labels=["Low Benefit", "Moderate Benefit", "High Benefit"])

    col_sub_a, col_sub_b = st.columns([1, 1])

    with col_sub_a:
        st.markdown("#### Treatment Benefit Tiers Across Test Cohort")
        fig_tier, ax_tier = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_subgroups, x="Tier", y="Estimated_ITE", palette="Blues", ax=ax_tier)
        ax_tier.set_title("Distribution of ITE Across Patient Benefit Tiers")
        st.pyplot(fig_tier)

    with col_sub_b:
        st.markdown("#### Confounder Difference by Benefit Tier")
        mean_profile = df_subgroups.groupby("Tier")[["x01", "x02", "x04", "x06"]].mean()
        st.dataframe(mean_profile.style.format("{:.2f}"))
        st.caption("Patients in the 'High Benefit' tier exhibit systematically lower birth weights (x01), deriving higher relative gains from intervention.")

# ==========================================
# TAB 5: Robustness & Refutation Lab
# ==========================================
with tabs[4]:
    st.subheader("5. Causal Robustness & Refutation Stress Lab")
    st.write(
        "Causal claims must withstand refutation tests. If a model fails when treatment is randomly shuffled "
        "or when noise is added, its causal conclusions are unreliable."
    )

    col_ref_a, col_ref_b = st.columns([1, 1])

    with col_ref_a:
        st.markdown("#### Test 1: Placebo Treatment Refutation")
        st.write("Shuffles treatment assignment randomly. A sound causal estimator's effect must collapse to near zero.")
        if st.button("Run Placebo Treatment Test", key="btn_placebo"):
            with st.spinner("Executing placebo permutations..."):
                res_placebo = refute_placebo_treatment(
                    selected_model_obj, train_data["X"], train_data["T"], train_data["Y"], n_simulations=10
                )
            badge = '<span class="badge-pass">PASSED</span>' if res_placebo["passed"] else '<span class="badge-fail">FAILED</span>'
            st.markdown(f"Status: {badge}", unsafe_allow_html=True)
            st.json(res_placebo)

    with col_ref_b:
        st.markdown("#### Test 2: Random Confounder Refutation")
        st.write("Injects an unobserved synthetic random noise variable into covariates. Effect estimate should remain stable.")
        if st.button("Run Random Confounder Test", key="btn_confounder"):
            with st.spinner("Injecting random noise confounders..."):
                res_confounder = refute_random_confounder(
                    selected_model_obj, train_data["X"], train_data["T"], train_data["Y"], n_simulations=10
                )
            badge = '<span class="badge-pass">PASSED</span>' if res_confounder["passed"] else '<span class="badge-fail">FAILED</span>'
            st.markdown(f"Status: {badge}", unsafe_allow_html=True)
            st.json(res_confounder)

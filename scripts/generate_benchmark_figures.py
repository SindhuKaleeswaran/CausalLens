"""Generate publication-quality benchmark visualization figures."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from causallens.data.loader import load_ihdp, load_ihdp_truth
from causallens.models import (
    DoublyRobustEstimator,
    IPWEstimator,
    NaiveEstimator,
    SLearner,
    TLearner,
    XLearner,
)


def set_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def main():
    set_style()
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(results_dir / "ihdp100_summary.csv")
    detailed_df = pd.read_csv(results_dir / "ihdp100_unified_results.csv")

    model_colors = {
        "T-Learner": "#0D9488",             # Teal
        "X-Learner": "#0284C7",             # Sky blue
        "IPW + Weighted Reg": "#6366F1",    # Indigo
        "Doubly Robust (AIPW)": "#059669",  # Emerald green
        "Naive (Unadjusted)": "#DC2626",    # Crimson red
        "S-Learner": "#D97706",             # Amber
    }

    # 1. PEHE & ATE Error comparison bar chart with 95% CI
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Sort models by root_pehe_mean
    sorted_sum = summary_df.sort_values("root_pehe_mean")
    models = sorted_sum["model"].tolist()
    colors = [model_colors.get(m, "#475569") for m in models]

    # Subplot A: Root PEHE Mean & 95% CI
    ax = axes[0]
    y_err = [
        sorted_sum["root_pehe_mean"] - sorted_sum["root_pehe_ci95_low"],
        sorted_sum["root_pehe_ci95_high"] - sorted_sum["root_pehe_mean"],
    ]
    bars = ax.bar(models, sorted_sum["root_pehe_mean"], yerr=y_err, capsize=4, color=colors, alpha=0.85, edgecolor="#1E293B", linewidth=0.8)
    ax.set_ylabel(r"Root-PEHE ($\sqrt{\epsilon_{PEHE}}$) — Mean (95% bootstrap CI)")
    ax.set_title("Heterogeneous Treatment Effect Precision", fontweight="bold")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add numeric labels on bars
    for bar, val in zip(bars, sorted_sum["root_pehe_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="600")

    # Subplot B: Absolute ATE Error Mean & 95% CI
    ax = axes[1]
    y_err_ate = [
        sorted_sum["ate_error_mean"] - sorted_sum["ate_error_ci95_low"],
        sorted_sum["ate_error_ci95_high"] - sorted_sum["ate_error_mean"],
    ]
    bars2 = ax.bar(models, sorted_sum["ate_error_mean"], yerr=y_err_ate, capsize=4, color=colors, alpha=0.85, edgecolor="#1E293B", linewidth=0.8)
    ax.set_ylabel("Absolute ATE Error — Mean (95% bootstrap CI)")
    ax.set_title("Population Effect Estimation Accuracy", fontweight="bold")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars2, sorted_sum["ate_error_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="600")

    plt.tight_layout()
    fig.savefig(fig_dir / "benchmark_pehe_ate_comparison.png")
    plt.close()
    print("Saved benchmark_pehe_ate_comparison.png")

    # 2. Boxplot / distribution across all 100 replications
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    sns.boxplot(
        data=detailed_df,
        x="model",
        y="root_pehe",
        order=models,
        palette=model_colors,
        ax=ax,
        fliersize=3,
        linewidth=1.2,
    )
    ax.set_ylabel(r"Root PEHE ($\sqrt{\epsilon_{PEHE}}$)")
    ax.set_title("Distribution of Root PEHE (100 Replications)", fontweight="bold")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax = axes[1]
    sns.boxplot(
        data=detailed_df,
        x="model",
        y="ate_error",
        order=models,
        palette=model_colors,
        ax=ax,
        fliersize=3,
        linewidth=1.2,
    )
    ax.set_ylabel(r"Absolute ATE Error ($|\hat{\tau} - \tau_{true}|$)")
    ax.set_title("Distribution of ATE Error (100 Replications)", fontweight="bold")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(fig_dir / "pehe_distribution_boxplot.png")
    plt.close()
    print("Saved pehe_distribution_boxplot.png")

    # 3. Precision vs Accuracy Trade-off Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        sub = detailed_df[detailed_df["model"] == m]
        ax.scatter(
            sub["ate_error"],
            sub["root_pehe"],
            color=model_colors[m],
            alpha=0.45,
            s=28,
            label=f"{m} (Med: {sub['root_pehe'].median():.2f})",
        )
    ax.set_xlabel("Absolute ATE Error (|ATE_hat - ATE_true|)")
    ax.set_ylabel(r"Root-PEHE ($\sqrt{\epsilon_{PEHE}}$)")
    ax.set_title("Precision vs Parameter Accuracy Trade-off (All 600 Fits)", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_dir / "ate_error_scatter_tradeoff.png")
    plt.close()
    print("Saved ate_error_scatter_tradeoff.png")

    # 4. Replication 0: Treatment Effect Density vs Oracle
    train_data = load_ihdp(0, split="train", data_dir=root / "data" / "raw")
    test_data = load_ihdp(0, split="test", data_dir=root / "data" / "raw")
    truth_data = load_ihdp_truth(0, split="test", data_dir=root / "data" / "raw")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.kdeplot(truth_data["tau"], label=f"Ground Truth Oracle tau (True ATE: {truth_data['tau'].mean():.2f})", color="#0F172A", linewidth=2.5, ax=ax)

    est_specs = [
        ("T-Learner", TLearner(), "#0D9488", "-"),
        ("X-Learner", XLearner(), "#0284C7", "--"),
        ("Doubly Robust (AIPW)", DoublyRobustEstimator(), "#059669", "-."),
        ("IPW + Weighted Reg", IPWEstimator(), "#6366F1", ":"),
        ("S-Learner", SLearner(), "#D97706", "-"),
    ]
    for name, est, c, ls in est_specs:
        est.fit(train_data["X"], train_data["T"], train_data["Y"])
        ite = est.estimate_ite(test_data["X"])
        sns.kdeplot(ite, label=f"{name} (Est: {ite.mean():.2f})", color=c, linestyle=ls, linewidth=1.6, ax=ax)

    ax.set_xlabel("Treatment Effect tau(x)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted CATE Distributions vs Ground Truth (Replication 0, Test Split N=75)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_dir / "treatment_effect_density_rep0.png")
    plt.close()
    print("Saved treatment_effect_density_rep0.png")

    # 5. Policy Risk & Policy Value Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    bars_pr = ax.bar(models, sorted_sum["policy_risk_mean"], color=colors, alpha=0.85, edgecolor="#1E293B", linewidth=0.8)
    ax.set_ylabel("Policy Risk / Regret (Mean across 100 Reps)")
    ax.set_title("Policy Decision Regret relative to Oracle", fontweight="bold")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars_pr, sorted_sum["policy_risk_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="600")

    ax = axes[1]
    sorted_corr = summary_df.dropna(subset=["cate_pearson_r_mean"]).sort_values("cate_pearson_r_mean", ascending=False)
    m_corr = sorted_corr["model"].tolist()
    c_corr = [model_colors.get(m, "#475569") for m in m_corr]
    bars_c = ax.bar(m_corr, sorted_corr["cate_pearson_r_mean"], color=c_corr, alpha=0.85, edgecolor="#1E293B", linewidth=0.8)
    ax.set_ylabel("Pearson Correlation r(tau_hat, tau_true)")
    ax.set_title("CATE Ranking & Heterogeneity Correlation", fontweight="bold")
    ax.set_xticklabels(m_corr, rotation=30, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars_c, sorted_corr["cate_pearson_r_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="600")

    plt.tight_layout()
    fig.savefig(fig_dir / "policy_risk_and_correlation.png")
    plt.close()
    print("Saved policy_risk_and_correlation.png")

    extended = results_dir / "extended" / "ihdp100_summary.csv"
    if extended.exists():
        comparison = pd.read_csv(extended).sort_values("root_pehe_mean")
        labels = comparison["model"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
        for ax, col, label in zip(axes, ("root_pehe", "ate_error", "policy_risk"),
                                  ("Root-PEHE", "Absolute ATE error", "Policy regret")):
            mean = comparison[col + "_mean"]
            bounds = [mean - comparison[col + "_mean_ci95_low"], comparison[col + "_mean_ci95_high"] - mean]
            ax.errorbar(mean, labels, xerr=bounds, fmt="o", color="#0F766E", capsize=3)
            ax.set_xlabel(label + " (outcome units)")
        axes[0].invert_yaxis()
        fig.suptitle("IHDP-100 extended experiment · replication means and 95% bootstrap intervals")
        fig.tight_layout()
        fig.savefig(fig_dir / "extended_ihdp100.png")
        plt.close(fig)


if __name__ == "__main__":
    main()

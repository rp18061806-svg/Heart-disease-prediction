"""Exploratory data analysis: reusable plotting functions.

Every function saves a PNG under ``reports/figures/`` and returns the
Matplotlib figure so it can also be shown inline in the notebooks.

Usage (generate every EDA figure):
    python -m src.eda
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PALETTE = {0: "#2E86AB", 1: "#D1495B"}
LABELS = config.TARGET_LABELS
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Outcome"] = out[config.TARGET].map(LABELS)
    return out


def plot_missing_values(raw: pd.DataFrame) -> plt.Figure:
    """Explicit NaNs vs. zero-coded missing values per column."""
    explicit = raw.isna().sum()
    zero_coded = pd.Series({c: int((raw[c] == 0).sum()) if c in ("RestingBP", "Cholesterol") else 0
                            for c in raw.columns})
    frame = pd.DataFrame({"Explicit NaN": explicit, "Zero used as 'not recorded'": zero_coded})
    frame = frame.loc[config.RAW_FEATURES]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    frame.plot.bar(ax=ax, color=["#8D99AE", "#EF8354"])
    ax.set_title("Missing-value analysis (labelled data)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Number of records")
    for i, v in enumerate(frame["Zero used as 'not recorded'"]):
        if v:
            ax.annotate(f"{v} ({100 * v / len(raw):.1f}%)", (i + 0.12, v), ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "01_missing_values")


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar([LABELS[i] for i in counts.index], counts.values,
                       color=[PALETTE[i] for i in counts.index])
    axes[0].bar_label(bars, labels=[f"{v} ({100 * v / counts.sum():.1f}%)" for v in counts.values])
    axes[0].set_title("Target distribution: HeartDisease")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Number of patients")
    axes[1].pie(counts.values, labels=[LABELS[i] for i in counts.index], autopct="%1.1f%%",
                colors=[PALETTE[i] for i in counts.index], startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title(f"Class balance (ratio {counts.max() / counts.min():.2f} : 1)")
    return _save(fig, "02_target_distribution")


def plot_numeric_histograms(df: pd.DataFrame) -> plt.Figure:
    data = _labelled(df)
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(data=data, x=col, hue="Outcome", kde=True, ax=ax, element="step",
                     palette=[PALETTE[0], PALETTE[1]], hue_order=[LABELS[0], LABELS[1]])
        ax.set_title(f"Distribution of {col} by outcome")
        ax.set_ylabel("Count")
    axes.flat[-1].axis("off")
    fig.suptitle("Numerical feature distributions (Cholesterol/RestingBP zeros treated as missing)",
                 y=1.02, fontsize=13)
    return _save(fig, "03_numeric_histograms")


def plot_boxplots(df: pd.DataFrame) -> plt.Figure:
    data = _labelled(df)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=data, x="Outcome", y=col, hue="Outcome", ax=ax, legend=False,
                    palette=[PALETTE[0], PALETTE[1]], order=[LABELS[0], LABELS[1]])
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Box plots of numerical features by outcome", y=1.03, fontsize=13)
    return _save(fig, "04_boxplots_by_target")


def plot_categorical_distributions(df: pd.DataFrame) -> plt.Figure:
    data = _labelled(df)
    cols = config.CATEGORICAL_FEATURES + ["FastingBS"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, cols):
        order = data[col].value_counts().index
        sns.countplot(data=data, x=col, hue="Outcome", order=order, ax=ax,
                      palette=[PALETTE[0], PALETTE[1]], hue_order=[LABELS[0], LABELS[1]])
        rate = data.groupby(col)[config.TARGET].mean().reindex(order)
        ax.set_title(f"{col}  (disease rate: " + ", ".join(f"{k}={v:.0%}" for k, v in rate.items()) + ")",
                     fontsize=10)
        ax.set_ylabel("Number of patients")
        ax.legend(title="Outcome", fontsize=8)
    fig.suptitle("Categorical feature distributions by outcome", y=1.01, fontsize=13)
    return _save(fig, "05_categorical_distributions")


def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    numeric = df[config.NUMERIC_FEATURES + config.BINARY_NUMERIC_FEATURES + [config.TARGET]]
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix (numeric features, zeros treated as missing)")
    return _save(fig, "06_correlation_matrix")


def plot_feature_target_association(df: pd.DataFrame) -> plt.Figure:
    """Point-biserial r for numeric, Cramér's V for categorical features."""
    from scipy.stats import chi2_contingency

    rows = []
    for col in config.NUMERIC_FEATURES:
        valid = df[[col, config.TARGET]].dropna()
        rows.append((col, abs(valid.corr().iloc[0, 1]), "Numeric |r|"))
    for col in config.CATEGORICAL_FEATURES + ["FastingBS"]:
        table = pd.crosstab(df[col], df[config.TARGET])
        chi2 = chi2_contingency(table)[0]
        v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
        rows.append((col, v, "Categorical Cramér's V"))
    assoc = pd.DataFrame(rows, columns=["Feature", "Strength", "Measure"]).sort_values("Strength")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = assoc["Measure"].map({"Numeric |r|": "#2E86AB", "Categorical Cramér's V": "#F18F01"})
    ax.barh(assoc["Feature"], assoc["Strength"], color=colors)
    for i, v in enumerate(assoc["Strength"]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Association strength with HeartDisease (0 = none, 1 = perfect)")
    ax.set_title("Univariate feature-target association")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2E86AB", label="Numeric: |point-biserial r|"),
                       Patch(color="#F18F01", label="Categorical: Cramér's V")], loc="lower right")
    return _save(fig, "07_feature_target_association")


def plot_cholesterol_artifact(raw: pd.DataFrame) -> plt.Figure:
    """Show why zero cholesterol must not be used as a real measurement."""
    recorded = raw["Cholesterol"] > 0
    rates = pd.Series({
        "Cholesterol recorded": raw.loc[recorded, config.TARGET].mean(),
        "Cholesterol = 0 (not recorded)": raw.loc[~recorded, config.TARGET].mean(),
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bars = axes[0].bar(rates.index, rates.values * 100, color=["#2E86AB", "#EF8354"])
    axes[0].bar_label(bars, labels=[f"{v:.1%}\n(n={n})" for v, n in
                                    zip(rates.values, [recorded.sum(), (~recorded).sum()])])
    axes[0].set_ylabel("Heart disease rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Disease rate depends on whether cholesterol was recorded")
    data = _labelled(raw)
    sns.histplot(data=data, x="Cholesterol", hue="Outcome", bins=40, ax=axes[1],
                 palette=[PALETTE[0], PALETTE[1]], hue_order=[LABELS[0], LABELS[1]])
    axes[1].set_title("Raw Cholesterol: spike at 0 is a data-collection artefact")
    axes[1].set_xlabel("Cholesterol (mg/dl, raw)")
    axes[1].set_ylabel("Count")
    return _save(fig, "08_cholesterol_zero_artifact")


def plot_outlier_summary(df: pd.DataFrame) -> plt.Figure:
    """IQR outlier counts per numeric feature (after zero->NaN cleaning)."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"Feature": col, "Below lower fence": int((s < low).sum()),
                     "Above upper fence": int((s > high).sum())})
    frame = pd.DataFrame(rows).set_index("Feature")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    frame.plot.bar(stacked=True, ax=ax, color=["#8D99AE", "#D1495B"])
    ax.set_title("Tukey (1.5×IQR) outlier counts after cleaning")
    ax.set_ylabel("Number of records")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, "09_outlier_summary")


def plot_pairwise_key_features(df: pd.DataFrame) -> plt.Figure:
    data = _labelled(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=data, x="Age", y="MaxHR", hue="Outcome", alpha=0.7, ax=axes[0],
                    palette=[PALETTE[0], PALETTE[1]], hue_order=[LABELS[0], LABELS[1]])
    ages = np.linspace(data["Age"].min(), data["Age"].max(), 50)
    axes[0].plot(ages, 220 - ages, "k--", lw=1, label="Age-predicted max (220 − age)")
    axes[0].legend()
    axes[0].set_title("MaxHR vs Age by outcome")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Maximum heart rate achieved (bpm)")
    sns.boxplot(data=data, x="ST_Slope", y="Oldpeak", hue="Outcome", ax=axes[1],
                order=["Up", "Flat", "Down"], palette=[PALETTE[0], PALETTE[1]],
                hue_order=[LABELS[0], LABELS[1]])
    axes[1].set_title("Oldpeak by ST_Slope and outcome")
    axes[1].set_xlabel("ST segment slope")
    axes[1].set_ylabel("Oldpeak (ST depression, mm)")
    return _save(fig, "10_key_feature_comparisons")


def run_all() -> None:
    """Generate every EDA figure from the labelled data."""
    from src.preprocessing import clean_raw_data

    raw = pd.read_csv(config.TRAIN_FILE)
    clean = clean_raw_data(raw)
    plot_missing_values(raw)
    plot_target_distribution(clean)
    plot_numeric_histograms(clean)
    plot_boxplots(clean)
    plot_categorical_distributions(clean)
    plot_correlation_matrix(clean)
    plot_feature_target_association(clean)
    plot_cholesterol_artifact(raw)
    plot_outlier_summary(clean)
    plot_pairwise_key_features(clean)
    plt.close("all")
    print(f"EDA figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run_all()

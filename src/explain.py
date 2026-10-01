"""Explainable AI utilities built on SHAP.

SHAP values describe how much each input *moved the model's output* for a
given prediction, relative to the average prediction on background data.
They are **model feature contributions, not medical causes**.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TREE_MODELS = ("RandomForest", "GradientBoosting", "DecisionTree", "XGB", "LGBM")


def transformed_feature_names(pipe) -> list[str]:
    """Column names after preprocessing (one-hot columns expanded)."""
    return list(pipe.named_steps["prep"].get_feature_names_out())


def original_feature_of(name: str) -> str:
    """Map an encoded column (e.g. ``ChestPainType_ASY``) to its source feature."""
    for cat in config.CATEGORICAL_FEATURES:
        if name.startswith(cat + "_"):
            return cat
    return name


def build_explainer(pipe, background_raw: pd.DataFrame):
    """Pick the most appropriate (fast, exact where possible) SHAP explainer."""
    import shap

    model = pipe.named_steps["model"]
    background = pipe.named_steps["prep"].transform(background_raw)
    kind = type(model).__name__
    if any(k in kind for k in TREE_MODELS):
        return shap.TreeExplainer(model, data=background, model_output="probability"), "tree"
    if kind == "LogisticRegression":
        return shap.LinearExplainer(model, background), "linear"
    summary = shap.kmeans(background, 20)
    return shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], summary), "kernel"


def shap_values_for(explainer, pipe, X_raw: pd.DataFrame) -> np.ndarray:
    """SHAP values for the positive class, shape (n_samples, n_encoded_features)."""
    Xt = pipe.named_steps["prep"].transform(X_raw)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def aggregate_to_original(values: np.ndarray, encoded_names: list[str]) -> pd.DataFrame:
    """Sum SHAP values of one-hot columns so each original feature gets one value."""
    frame = pd.DataFrame(values, columns=encoded_names)
    groups = [original_feature_of(n) for n in encoded_names]
    return frame.T.groupby(groups, sort=False).sum().T


def explain_single(pipe, explainer, record: pd.DataFrame, top_k: int = 11) -> pd.DataFrame:
    """Readable explanation for one patient record.

    Returns a DataFrame with columns: feature, value, contribution, direction.
    """
    names = transformed_feature_names(pipe)
    vals = shap_values_for(explainer, pipe, record)
    agg = aggregate_to_original(vals, names).iloc[0]
    rows = []
    for feat, contrib in agg.items():
        if feat == "MaxHR_pct_predicted":
            value = float(record["MaxHR"].iloc[0]) / (220 - float(record["Age"].iloc[0]))
            value = f"{value:.0%}"
        else:
            value = record[feat].iloc[0]
            if feat == "Cholesterol" and pd.isna(value):
                value = "not provided (imputed)"
            elif isinstance(value, (float, np.floating)) and float(value).is_integer() and feat != "Oldpeak":
                value = int(value)
        rows.append({
            "feature": feat,
            "value": value,
            "contribution": float(contrib),
            "direction": "↑ increases predicted risk" if contrib > 0 else "↓ decreases predicted risk",
        })
    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").head(top_k).reset_index(drop=True)


def global_importance(pipe, explainer, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original feature over ``X_raw``."""
    names = transformed_feature_names(pipe)
    vals = shap_values_for(explainer, pipe, X_raw)
    agg = aggregate_to_original(vals, names).abs().mean().sort_values(ascending=False)
    return agg.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})


def save_global_plots(pipe, explainer, X_raw: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Write SHAP summary (beeswarm) and bar plots to reports/figures/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    names = transformed_feature_names(pipe)
    Xt = pd.DataFrame(pipe.named_steps["prep"].transform(X_raw), columns=names)
    vals = shap_values_for(explainer, pipe, X_raw)

    plt.figure()
    shap.summary_plot(vals, Xt, show=False, max_display=15)
    plt.title(f"SHAP summary (beeswarm) - {model_name}\n"
              "Each dot is a patient; x = contribution to P(heart disease)", fontsize=11)
    plt.xlabel("SHAP value (impact on predicted probability of heart disease)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "17_shap_summary_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    agg = aggregate_to_original(vals, names).abs().mean().sort_values(ascending=False)
    imp = agg.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    fig, ax = plt.subplots(figsize=(8, 5))
    t = imp.sort_values("mean_abs_shap")
    ax.barh(t["feature"], t["mean_abs_shap"], color="#D1495B")
    for i, v in enumerate(t["mean_abs_shap"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (average change in predicted probability)")
    ax.set_title(f"Global feature importance (SHAP) - {model_name}\n"
                 "One-hot columns summed per original feature; not medical causation")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "18_shap_global_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return imp

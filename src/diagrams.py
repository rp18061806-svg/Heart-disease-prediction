"""Render the system-architecture and project-workflow diagrams as PNG files.

Usage:
    python -m src.diagrams
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

STAGE_COLORS = {
    "user": "#264653", "ui": "#2A9D8F", "prep": "#E9C46A", "model": "#F4A261",
    "output": "#E76F51", "data": "#457B9D", "ml": "#8AB17D", "deploy": "#6D597A",
}


def _box(ax, x, y, w, h, text, color, sub=None, fontsize=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#1d1d1d", lw=1.2, alpha=0.95))
    dark = color in ("#264653", "#457B9D", "#6D597A", "#2A9D8F", "#E76F51")
    ax.text(x, y + (0.1 if sub else 0), text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white" if dark else "#1d1d1d")
    if sub:
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.2,
                color="white" if dark else "#1d1d1d")


def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#333"))


def architecture() -> None:
    steps = [
        ("User", "enters patient attributes (no PII)", "user"),
        ("Streamlit UI", "app.py · form widgets · navigation", "ui"),
        ("Input Validation", "required fields · ranges · allowed categories", "prep"),
        ("Preprocessing Pipeline", "clean_raw_data: whitespace, 0 → missing", "prep"),
        ("Feature Transformation", "MaxHR % predicted · impute · scale · one-hot", "prep"),
        ("Trained ML Model", "final_model.joblib (sklearn Pipeline)", "model"),
        ("Prediction", "class at threshold 0.5", "output"),
        ("Probability", "P(heart disease) → Low / Moderate / High band", "output"),
        ("Explainable AI", "SHAP contributions per feature", "output"),
        ("Result Dashboard", "metric cards · gauge · charts · disclaimer", "ui"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(steps) * 1.25 + 0.2)
    ax.axis("off")
    ys = [len(steps) * 1.25 - i * 1.25 - 0.5 for i in range(len(steps))]
    for (title, sub, kind), y in zip(steps, ys):
        _box(ax, 5, y, 6.2, 0.85, title, STAGE_COLORS[kind], sub)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        _arrow(ax, (5, y1 - 0.43), (5, y2 + 0.43))
    # side annotations
    ax.text(9.1, (ys[2] + ys[4]) / 2, "same code path\nas training\n(no skew)", ha="center", va="center",
            fontsize=8.5, style="italic", color="#555")
    ax.set_title("System Architecture — Disease Prediction & Risk Assessment System", fontsize=13,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def workflow() -> None:
    steps = [
        ("Dataset", "Kaggle Heart Failure Prediction", "data"),
        ("Data Collection", "heart_train.csv (734) + heart_test.csv (184, unlabelled)", "data"),
        ("Data Cleaning", "types · spaces · 0 → missing · duplicates", "prep"),
        ("EDA", "distributions · associations · outliers", "prep"),
        ("Feature Engineering", "MaxHR_pct_predicted (ablation-tested)", "prep"),
        ("Train/Test Split", "stratified 80/20, random_state=42", "ml"),
        ("Preprocessing", "Pipeline + ColumnTransformer (fit on train only)", "ml"),
        ("Model Training", "8 classifiers", "ml"),
        ("Cross Validation", "stratified 5-fold", "ml"),
        ("Hyperparameter Tuning", "RandomizedSearchCV on top 3", "ml"),
        ("Model Evaluation", "held-out test: ROC-AUC, recall, F1, CM", "model"),
        ("Model Selection", "CV ROC-AUC, recall tie-break", "model"),
        ("Model Serialization", "joblib pipeline + metadata JSON", "model"),
        ("Streamlit Application", "prediction · risk · SHAP", "ui"),
        ("Hugging Face Deployment", "Docker Space running streamlit run app.py", "deploy"),
    ]
    # two-column snake layout for readability
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    col_x = [3, 9]
    per_col = 8
    pos = []
    for i in range(len(steps)):
        c = 0 if i < per_col else 1
        r = i if c == 0 else i - per_col
        pos.append((col_x[c], 8.0 - r * 1.05))
    for (title, sub, kind), (x, y) in zip(steps, pos):
        _box(ax, x, y, 4.8, 0.78, title, STAGE_COLORS[kind], sub, fontsize=10.5)
    for i in range(len(steps) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if x1 == x2:
            _arrow(ax, (x1, y1 - 0.39), (x2, y2 + 0.39))
        else:
            _arrow(ax, (x1 + 2.4, y1), (x2 - 2.4, y2 + 0.2))
    ax.set_title("Project Workflow — from dataset to deployment", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "workflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    workflow()
    print("Wrote reports/architecture.png and reports/workflow.png")

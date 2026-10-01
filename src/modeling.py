"""Model definitions, cross-validation, tuning and evaluation utilities."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
RS = config.RANDOM_STATE


def get_models() -> dict:
    """Return the candidate classifiers with sensible defaults.

    XGBoost and LightGBM are included only if they import successfully, so the
    project still trains in a minimal environment.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RS),
        "Decision Tree": DecisionTreeClassifier(random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=1),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=1,
                                          tree_method="hist")
    except ImportError:  # pragma: no cover
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=RS, verbose=-1, n_jobs=1)
    except ImportError:  # pragma: no cover
        pass
    return models


# Search spaces used by RandomizedSearchCV (keys refer to the "model" step).
PARAM_SPACES = {
    "Logistic Regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
        "model__class_weight": [None, "balanced"],
    },
    "Decision Tree": {
        "model__max_depth": randint(2, 12),
        "model__min_samples_leaf": randint(1, 30),
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": randint(200, 800),
        "model__max_depth": [None, 4, 6, 8, 10, 14],
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": [None, "balanced"],
    },
    "K-Nearest Neighbors": {
        "model__n_neighbors": randint(3, 40),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Support Vector Machine": {
        "model__C": loguniform(1e-2, 1e2),
        "model__gamma": loguniform(1e-4, 1e0),
        "model__kernel": ["rbf"],
    },
    "Gradient Boosting": {
        "model__n_estimators": randint(50, 400),
        "model__learning_rate": loguniform(1e-2, 3e-1),
        "model__max_depth": randint(2, 5),
        "model__subsample": uniform(0.6, 0.4),
        "model__min_samples_leaf": randint(1, 20),
    },
    "XGBoost": {
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(1e-2, 3e-1),
        "model__max_depth": randint(2, 6),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__min_child_weight": randint(1, 10),
        "model__reg_lambda": loguniform(1e-2, 1e1),
    },
    "LightGBM": {
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(1e-2, 3e-1),
        "model__num_leaves": randint(4, 32),
        "model__min_child_samples": randint(5, 40),
        "model__subsample": uniform(0.6, 0.4),
        "model__subsample_freq": [1],
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(1e-2, 1e1),
    },
}


def make_pipeline(estimator, **prep_kwargs) -> Pipeline:
    """Full leakage-safe pipeline: preprocessing + classifier."""
    return Pipeline([("prep", build_preprocessor(**prep_kwargs)), ("model", estimator)])


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)


def cross_validate_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Stratified K-fold CV; returns mean and std for each metric."""
    res = cross_validate(pipe, X, y, cv=cv_splitter(), scoring=SCORING, n_jobs=-1)
    out = {}
    for m in SCORING:
        scores = res[f"test_{m}"]
        out[f"cv_{m}_mean"] = float(np.mean(scores))
        out[f"cv_{m}_std"] = float(np.std(scores))
    return out


def test_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: float = 0.5) -> dict:
    """Metrics on the held-out test split."""
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compare_models(X_train, y_train, X_test, y_test, models: dict | None = None) -> pd.DataFrame:
    """Train every baseline model; report CV and held-out metrics + training time."""
    rows = []
    for name, est in (models or get_models()).items():
        pipe = make_pipeline(est)
        cv = cross_validate_pipeline(pipe, X_train, y_train)
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        tm = test_metrics(pipe, X_test, y_test)
        rows.append({
            "Model": name,
            "Accuracy": tm["accuracy"], "Precision": tm["precision"], "Recall": tm["recall"],
            "F1 Score": tm["f1"], "ROC-AUC": tm["roc_auc"],
            "CV ROC-AUC (mean)": cv["cv_roc_auc_mean"], "CV ROC-AUC (std)": cv["cv_roc_auc_std"],
            "CV F1 (mean)": cv["cv_f1_mean"], "CV F1 (std)": cv["cv_f1_std"],
            "CV Recall (mean)": cv["cv_recall_mean"], "CV Accuracy (mean)": cv["cv_accuracy_mean"],
            "Training time (s)": train_time,
        })
        print(f"  {name:24s} CV AUC={cv['cv_roc_auc_mean']:.3f}±{cv['cv_roc_auc_std']:.3f} "
              f"CV F1={cv['cv_f1_mean']:.3f}  test AUC={tm['roc_auc']:.3f}")
    return pd.DataFrame(rows).sort_values("CV ROC-AUC (mean)", ascending=False).reset_index(drop=True)


def tune_model(name: str, X_train, y_train, n_iter: int = 40) -> RandomizedSearchCV:
    """RandomizedSearchCV (ROC-AUC) over the model's search space."""
    search = RandomizedSearchCV(
        make_pipeline(get_models()[name]), PARAM_SPACES[name], n_iter=n_iter,
        scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"}, refit="roc_auc",
        cv=cv_splitter(), random_state=RS, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


# --------------------------------------------------------------------------
# Evaluation figures
# --------------------------------------------------------------------------
def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    long = table.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.barplot(data=long, x="Model", y="Score", hue="Metric", ax=ax, palette="viridis")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Baseline model comparison on the held-out test split")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, "11_model_comparison")

    fig, ax = plt.subplots(figsize=(10, 5))
    t = table.sort_values("CV ROC-AUC (mean)")
    ax.barh(t["Model"], t["CV ROC-AUC (mean)"], xerr=t["CV ROC-AUC (std)"], color="#2E86AB", capsize=4)
    for i, (m, s) in enumerate(zip(t["CV ROC-AUC (mean)"], t["CV ROC-AUC (std)"])):
        ax.text(m + s + 0.003, i, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("5-fold cross-validated ROC-AUC (mean ± std)")
    ax.set_title("Cross-validation ROC-AUC of baseline models (training split)")
    _save(fig, "12_cv_roc_auc")


def plot_evaluation(pipe, X_test, y_test, model_name: str) -> dict:
    """Confusion matrix, ROC, PR curves and threshold analysis for the final model."""
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    names = [config.TARGET_LABELS[0], config.TARGET_LABELS[1]]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=names, cmap="Blues", ax=ax,
                                            colorbar=False)
    ax.set_title(f"Confusion matrix - {model_name}\n(held-out test split, threshold 0.5)")
    _save(fig, "13_confusion_matrix")

    fig, ax = plt.subplots(figsize=(6, 5))
    roc = RocCurveDisplay.from_predictions(y_test, proba, name=model_name, ax=ax)
    roc.line_.set_color("#D1495B")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    ax.set_title("ROC curve (held-out test split)")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right")
    _save(fig, "14_roc_curve")

    fig, ax = plt.subplots(figsize=(6, 5))
    pr = PrecisionRecallDisplay.from_predictions(y_test, proba, name=model_name, ax=ax)
    pr.line_.set_color("#2E86AB")
    ax.axhline(y_test.mean(), ls="--", color="k", lw=1, label=f"Prevalence baseline ({y_test.mean():.2f})")
    ax.set_title("Precision-Recall curve (held-out test split)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    _save(fig, "15_precision_recall_curve")

    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        p = (proba >= t).astype(int)
        rows.append({"threshold": t, "Precision": precision_score(y_test, p, zero_division=0),
                     "Recall": recall_score(y_test, p), "F1": f1_score(y_test, p)})
    th = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, c in zip(["Precision", "Recall", "F1"], ["#2E86AB", "#D1495B", "#3B8B5A"]):
        ax.plot(th["threshold"], th[col], label=col, color=c)
    ax.axvline(0.5, ls="--", color="grey", lw=1, label="Default threshold 0.5")
    ax.set_xlabel("Decision threshold on P(heart disease)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold trade-off (held-out test split)")
    ax.legend()
    _save(fig, "16_threshold_analysis")

    report = classification_report(y_test, pred, target_names=names, output_dict=True)
    return report

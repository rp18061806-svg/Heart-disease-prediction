"""Generate (and optionally execute) the four project notebooks.

The notebooks are thin, readable wrappers around the tested project modules, so
no logic exists only inside a notebook.

Usage:
    python -m src.build_notebooks            # build + execute (requires trained model)
    python -m src.build_notebooks --no-exec  # build only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
import config
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)"""


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


def nb_data_analysis():
    return [
        md("""# 01 · Data Analysis — Dataset Inspection & Profiling
**Project:** Disease Prediction & Risk Assessment System

This notebook inspects the raw Kaggle files *before* any modelling decision is made:
schema, data types, missing values, duplicates, cardinality, descriptive statistics,
outliers, target identification and class balance."""),
        code(SETUP),
        md("## 1. Load the raw files"),
        code("""train_raw = pd.read_csv(config.TRAIN_FILE)
unlabeled_raw = pd.read_csv(config.UNLABELED_FILE)
for name, df in [("heart_train.csv", train_raw), ("heart_test.csv", unlabeled_raw)]:
    print(f"{name:16s} rows={df.shape[0]:4d}  columns={df.shape[1]}")
print("\\nColumns only in heart_train.csv:", sorted(set(train_raw.columns) - set(unlabeled_raw.columns)))
train_raw.head()"""),
        md("## 2. Schema, data types, missing values and cardinality"),
        code("""schema = pd.DataFrame({
    "dtype": train_raw.dtypes.astype(str),
    "missing": train_raw.isna().sum(),
    "missing_%": (train_raw.isna().mean() * 100).round(2),
    "n_unique": train_raw.nunique(),
    "top_value_share": train_raw.apply(lambda s: s.value_counts(normalize=True).iloc[0]).round(3),
    "example": train_raw.iloc[0],
})
schema["constant"] = schema["n_unique"] <= 1
schema["near_constant (>=95%)"] = schema["top_value_share"] >= 0.95
schema"""),
        code("""print("Duplicate rows (labelled file):", train_raw.duplicated().sum())
print("Duplicate rows (unlabelled file):", unlabeled_raw.duplicated().sum())
print("Rows present in both files:", pd.merge(train_raw.drop(columns=config.TARGET), unlabeled_raw).shape[0])"""),
        md("## 3. Categorical columns — levels and leading/trailing spaces"),
        code("""for col in train_raw.select_dtypes(exclude="number").columns:
    s = train_raw[col]
    spaces = (s.astype(str) != s.astype(str).str.strip()).any()
    print(f"{col:15s} cardinality={s.nunique()}  spaces={spaces}  levels={s.value_counts().to_dict()}")"""),
        md("## 4. Numerical statistics (min, quartiles, mean, median, std, skew)"),
        code("""num = train_raw.select_dtypes("number")
stats = num.describe().T
stats["median"] = num.median()
stats["skew"] = num.skew().round(3)
stats["zeros"] = (num == 0).sum()
stats["negatives"] = (num < 0).sum()
stats.round(3)"""),
        md("""## 5. Impossible / suspicious values
A resting blood pressure or serum cholesterol of **0** is physiologically impossible. In this
dataset, zero is the code used by some source cohorts for *not recorded*."""),
        code("""zero_chol = train_raw["Cholesterol"] == 0
print("Cholesterol == 0:", zero_chol.sum(), f"({zero_chol.mean():.1%})")
print("RestingBP == 0 (labelled):", (train_raw["RestingBP"] == 0).sum(),
      "| (unlabelled):", (unlabeled_raw["RestingBP"] == 0).sum())
print("Oldpeak < 0 (ST elevation, valid):", (train_raw["Oldpeak"] < 0).sum())
print()
print("Heart-disease rate when Cholesterol recorded:", round(train_raw.loc[~zero_chol, config.TARGET].mean(), 3))
print("Heart-disease rate when Cholesterol == 0   :", round(train_raw.loc[zero_chol, config.TARGET].mean(), 3))"""),
        md("""**Decision:** zeros in `Cholesterol` and `RestingBP` are converted to missing and imputed with the
*training-split* median inside the pipeline. A "cholesterol missing" indicator is **not** used, because
the missingness pattern identifies the source cohort (a collection artefact) rather than physiology.
The ablation in `train.py` quantifies what that choice costs."""),
        md("## 6. Outlier screening (Tukey 1.5 × IQR, after zero → missing)"),
        code("""from src.preprocessing import clean_raw_data
clean = clean_raw_data(train_raw)
rows = []
for col in config.NUMERIC_FEATURES:
    s = clean[col].dropna(); q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                 "n_below": int((s < lo).sum()), "n_above": int((s > hi).sum()),
                 "min": s.min(), "max": s.max()})
pd.DataFrame(rows).round(2)"""),
        md("""**Outlier treatment:** all remaining extreme values (e.g. RestingBP 180–200, Cholesterol > 500,
MaxHR 60, Oldpeak −2.6 or 5.6) are physiologically *possible* and are **retained**. Tree ensembles are
insensitive to monotone scale; for distance/linear models, standard scaling limits their influence."""),
        md("## 7. Target variable identification and class balance"),
        code("""target_candidates = [c for c in train_raw.columns if c not in unlabeled_raw.columns]
print("Target candidates (in labelled file only):", target_candidates)
vc = train_raw[config.TARGET].value_counts().sort_index()
print(pd.DataFrame({"count": vc, "percent": (vc / vc.sum() * 100).round(2)}))
print("Imbalance ratio (majority : minority) =", round(vc.max() / vc.min(), 3))"""),
        md("""`HeartDisease` is the only column absent from the unlabelled file, is binary (0/1), and matches the
Kaggle dataset documentation — it is the target. The 55 : 45 split is **mildly imbalanced**, so no
resampling is required; stratified splitting and recall/F1/ROC-AUC reporting are used instead."""),
        md("## 8. Correlations among numeric columns"),
        code("""corr = clean[config.NUMERIC_FEATURES + ["FastingBS", config.TARGET]].corr().round(3)
high = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i+1:] if abs(corr.loc[a, b]) >= 0.7]
print("Pairs with |r| >= 0.7:", high or "none")
corr"""),
        md("## 9. Saved profile\n`python -m src.profile_dataset` writes the full profile to `reports/dataset_profile.json`."),
        code("""profile = json.loads((config.REPORTS_DIR / "dataset_profile.json").read_text())
print(json.dumps({k: profile[k] for k in ["target_candidates", "column_sets_match", "overlap_rows_between_files"]}, indent=2))"""),
    ]


def nb_eda():
    figs = [
        ("01_missing_values", "plot_missing_values(train_raw)",
         "Only zero-coded missingness exists: 129 cholesterol values (17.6 %) are 0; no explicit NaNs."),
        ("02_target_distribution", "plot_target_distribution(clean)",
         "Mild imbalance (≈55 % positive) → stratified splitting; no resampling needed."),
        ("03_numeric_histograms", "plot_numeric_histograms(clean)",
         "Diseased patients are older on average, reach lower MaxHR and show larger Oldpeak."),
        ("04_boxplots_by_target", "plot_boxplots(clean)",
         "Clear location shifts for Age, MaxHR and Oldpeak; RestingBP and Cholesterol overlap heavily."),
        ("05_categorical_distributions", "plot_categorical_distributions(clean)",
         "ST_Slope (Flat/Down), asymptomatic chest pain and exercise angina are strongly associated with disease."),
        ("06_correlation_matrix", "plot_correlation_matrix(clean)",
         "No pair of numeric features exceeds |r| = 0.7 → no redundancy-driven removal needed."),
        ("07_feature_target_association", "plot_feature_target_association(clean)",
         "Ranking of univariate association: ST_Slope, ChestPainType, ExerciseAngina, MaxHR, Oldpeak lead."),
        ("08_cholesterol_zero_artifact", "plot_cholesterol_artifact(train_raw)",
         "Records with Cholesterol = 0 have ~90 % disease rate — a collection artefact, not biology."),
        ("09_outlier_summary", "plot_outlier_summary(clean)",
         "Few IQR outliers remain after zero → NaN; all are plausible and are retained."),
        ("10_key_feature_comparisons", "plot_pairwise_key_features(clean)",
         "Diseased patients cluster below the age-predicted maximum HR line → motivates MaxHR_pct_predicted."),
    ]
    cells = [
        md("""# 02 · Exploratory Data Analysis
All plots are produced by functions in `src/eda.py` and saved to `reports/figures/`.
EDA is descriptive; no statistic computed here is fed into model fitting."""),
        code(SETUP),
        code("""from src.preprocessing import clean_raw_data
from src.eda import *
train_raw = pd.read_csv(config.TRAIN_FILE)
clean = clean_raw_data(train_raw)
print(clean.shape)"""),
    ]
    for i, (name, call, note) in enumerate(figs, start=1):
        title = name.split("_", 1)[1].replace("_", " ").title()
        show = f"fig = {call}\nplt.close(fig)\ndisplay(Image(filename=str(config.FIGURES_DIR / '{name}.png')))"
        cells += [md(f"## {i}. {title}"), code(show), md(f"**Observation:** {note}")]
    cells += [
        md("## 11. Interactive view (Plotly)"),
        code("""import plotly.express as px
d = clean.assign(Outcome=clean[config.TARGET].map(config.TARGET_LABELS))
fig = px.scatter(d, x="Age", y="MaxHR", color="Outcome", symbol="ExerciseAngina",
                 color_discrete_map={"No Heart Disease": "#2E86AB", "Heart Disease": "#D1495B"},
                 title="Age vs MaxHR by outcome (symbol = exercise angina)",
                 labels={"MaxHR": "Max heart rate (bpm)", "Age": "Age (years)"})
fig.write_html(config.FIGURES_DIR / "interactive_age_maxhr.html")
fig.show()"""),
        md("""## 12. EDA summary
| Finding | Consequence for the pipeline |
|---|---|
| No NaNs, but `Cholesterol`/`RestingBP` use 0 for *not recorded* | Convert 0 → NaN, median-impute inside the pipeline |
| Zero-cholesterol rows are ~90 % positive (cohort artefact) | Do **not** add a missingness flag |
| Mild class imbalance (55/45) | Stratified split + CV; report recall, F1, ROC-AUC |
| Remaining outliers are plausible | Retain; scale for linear/distance models |
| No highly collinear numeric pairs | Keep all features |
| MaxHR depends on age | Engineer `MaxHR_pct_predicted = MaxHR / (220 − Age)` |"""),
    ]
    return cells


def nb_training():
    return [
        md("""# 03 · Model Training, Cross-Validation & Tuning
Leakage-safe workflow: split first, then fit **every** learned transformation inside a scikit-learn
`Pipeline` on the training split only. The full reproducible run is `python train.py`; this notebook
re-runs the baseline comparison live and loads the tuning artefacts it produced."""),
        code(SETUP),
        code("""from src import modeling
from src.preprocessing import build_preprocessor
train = pd.read_csv(config.PROCESSED_DATA_DIR / "train_split.csv")
test = pd.read_csv(config.PROCESSED_DATA_DIR / "test_split.csv")
X_train, y_train = train[config.RAW_FEATURES], train[config.TARGET]
X_test, y_test = test[config.RAW_FEATURES], test[config.TARGET]
print(f"train={len(train)}  test={len(test)}  pos-rate train={y_train.mean():.3f} test={y_test.mean():.3f}")"""),
        md("## 1. The preprocessing pipeline"),
        code("""prep = build_preprocessor()
prep"""),
        code("""prep.fit(X_train)
print("Encoded feature names:", list(prep.get_feature_names_out()))"""),
        md("## 2. Feature-engineering ablation (5-fold CV, training split)"),
        code("""pd.read_csv(config.REPORTS_DIR / "feature_engineering_ablation.csv").round(4)"""),
        md("## 3. Feature-selection check (5-fold CV, training split)"),
        code("""pd.read_csv(config.REPORTS_DIR / "feature_selection_check.csv").round(4)"""),
        md("## 4. Baseline comparison of 8 classifiers (re-run live)"),
        code("""comparison = modeling.compare_models(X_train, y_train, X_test, y_test)
comparison.round(4)"""),
        code("""display(Image(filename=str(config.FIGURES_DIR / "11_model_comparison.png")))
display(Image(filename=str(config.FIGURES_DIR / "12_cv_roc_auc.png")))"""),
        md("""## 5. Hyperparameter tuning
`RandomizedSearchCV` (40 candidates × 5 folds, `refit="roc_auc"`, F1 and recall also tracked) is run
for the three models with the highest baseline CV ROC-AUC. Randomised search is used because the
spaces are continuous/large; an exhaustive grid would be unnecessarily expensive."""),
        code("""import inspect
print(inspect.getsource(modeling.tune_model))
pd.read_csv(config.REPORTS_DIR / "tuning_results.csv").round(4)"""),
        code("""print(json.dumps(json.loads(config.BEST_PARAMS_PATH.read_text()), indent=2))"""),
        md("""## 6. Model-selection rule (decided *before* looking at the test split)
1. Highest mean 5-fold CV ROC-AUC on the training split.
2. Models within 0.005 of the best are treated as tied (CV std is ≈ 0.02).
3. Ties are broken by higher CV **recall** (a missed case is the costlier error in screening),
   then by simplicity."""),
        code("""import train as training_script
print(inspect.getsource(training_script.select_final))
print("Selected:", json.loads(config.METADATA_PATH.read_text())["model_name"])"""),
    ]


def nb_evaluation():
    return [
        md("""# 04 · Final Model Evaluation & Explainable AI
The saved pipeline (`models/final_model.joblib`) is evaluated **once** on the held-out 20 % split."""),
        code(SETUP),
        code("""import joblib
from sklearn.metrics import classification_report, confusion_matrix
from src import modeling, explain
model = joblib.load(config.MODEL_PATH)
meta = json.loads(config.METADATA_PATH.read_text())
test = pd.read_csv(config.PROCESSED_DATA_DIR / "test_split.csv")
X_test, y_test = test[config.RAW_FEATURES], test[config.TARGET]
print("Model:", meta["model_name"])
model"""),
        md("## 1. Metrics on the held-out test split"),
        code("""m = modeling.test_metrics(model, X_test, y_test)
pd.Series({k: v for k, v in m.items() if k != "confusion_matrix"}).round(4).to_frame("value")"""),
        code("""pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=list(config.TARGET_LABELS.values()), digits=3))
print("Confusion matrix [[TN FP] [FN TP]]:\\n", confusion_matrix(y_test, pred))"""),
        md("## 2. Confusion matrix, ROC and Precision-Recall curves"),
        code("""for f in ["13_confusion_matrix", "14_roc_curve", "15_precision_recall_curve", "16_threshold_analysis"]:
    display(Image(filename=str(config.FIGURES_DIR / f"{f}.png")))"""),
        md("""**Reading the curves:** ROC-AUC is the probability that a randomly chosen diseased patient receives a
higher score than a randomly chosen healthy one. The threshold plot shows that lowering the threshold
below 0.5 raises recall at the expense of precision — a policy choice for a real screening setting."""),
        md("## 3. Cross-validation vs held-out performance"),
        code("""cv = meta["evaluation_metrics"]["cross_validation"]
pd.DataFrame({"5-fold CV (train split)": [cv["cv_roc_auc"], cv["cv_f1"], cv["cv_recall"]],
              "CV std": [cv["cv_roc_auc_std"], cv["cv_f1_std"], cv["cv_recall_std"]],
              "Held-out test": [m["roc_auc"], m["f1"], m["recall"]]},
             index=["ROC-AUC", "F1", "Recall"]).round(4)"""),
        md("## 4. Global explainability (SHAP)"),
        code("""display(Image(filename=str(config.FIGURES_DIR / "17_shap_summary_beeswarm.png")))
display(Image(filename=str(config.FIGURES_DIR / "18_shap_global_importance.png")))
pd.read_csv(config.REPORTS_DIR / "shap_global_importance.csv").round(4)"""),
        md("## 5. Local explanation — *why did the model make this prediction?*"),
        code("""bg = pd.read_csv(config.SHAP_BACKGROUND_PATH)
explainer, kind = explain.build_explainer(model, bg)
for idx in [int(np.argmax(model.predict_proba(X_test)[:, 1])), int(np.argmin(model.predict_proba(X_test)[:, 1]))]:
    rec = X_test.iloc[[idx]]
    p = model.predict_proba(rec)[0, 1]
    print(f"Test row {idx}: P(heart disease) = {p:.3f}, true label = {y_test.iloc[idx]}")
    display(explain.explain_single(model, explainer, rec, top_k=5).round(4))"""),
        md("""> **Important:** SHAP values are *model feature contributions*: they quantify how the trained model
> uses each input for this prediction. They are associations learned from a historical dataset and
> are **not evidence of medical causation**."""),
        md("## 6. Scoring the unlabelled Kaggle file (demonstration only — no ground truth)"),
        code("""preds = pd.read_csv(config.PROCESSED_DATA_DIR / "unlabeled_predictions.csv")
print(preds["predicted_class"].value_counts().rename(index=config.TARGET_LABELS))
preds.head()"""),
    ]


def build(execute: bool = True) -> None:
    NB_DIR.mkdir(exist_ok=True)
    books = {
        "01_data_analysis.ipynb": nb_data_analysis(),
        "02_eda.ipynb": nb_eda(),
        "03_model_training.ipynb": nb_training(),
        "04_model_evaluation.ipynb": nb_evaluation(),
    }
    for name, cells in books.items():
        nb = new_notebook(cells=cells, metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}})
        path = NB_DIR / name
        if execute:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbformat.write(nb, path)
        print(f"{'executed' if execute else 'built'}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-exec", action="store_true")
    build(execute=not parser.parse_args().no_exec)
    sys.exit(0)

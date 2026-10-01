"""🩺 Disease Prediction & Risk Assessment System - Streamlit application.

Run locally:
    streamlit run app.py

Privacy: patient inputs live only in the current browser session's memory
(``st.session_state``). Nothing is written to disk, logged, or transmitted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from predict import ModelNotFoundError, load_metadata, load_model, predict_record, record_to_frame
from src import explain

st.set_page_config(
    page_title="Disease Prediction & Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants for the UI
# --------------------------------------------------------------------------
RISK_COLORS = {"Low": "#2A9D8F", "Moderate": "#E9A23B", "High": "#D1495B"}
CHEST_PAIN = {
    "ASY": "Asymptomatic (ASY)",
    "NAP": "Non-anginal pain (NAP)",
    "ATA": "Atypical angina (ATA)",
    "TA": "Typical angina (TA)",
}
RESTING_ECG = {
    "Normal": "Normal",
    "ST": "ST-T wave abnormality (ST)",
    "LVH": "Left ventricular hypertrophy (LVH)",
}
ST_SLOPE = {"Up": "Upsloping (Up)", "Flat": "Flat", "Down": "Downsloping (Down)"}
FEATURE_DESCRIPTIONS = {
    "Age": "Age of the patient (years)",
    "Sex": "Sex of the patient (M = male, F = female)",
    "ChestPainType": "Chest pain type: TA typical angina, ATA atypical angina, NAP non-anginal pain, ASY asymptomatic",
    "RestingBP": "Resting blood pressure (mm Hg)",
    "Cholesterol": "Serum cholesterol (mg/dl); 0 in the raw data means 'not recorded'",
    "FastingBS": "Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)",
    "RestingECG": "Resting ECG: Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy)",
    "MaxHR": "Maximum heart rate achieved during exercise testing (bpm)",
    "ExerciseAngina": "Exercise-induced angina (Y = yes, N = no)",
    "Oldpeak": "ST depression induced by exercise relative to rest (mm)",
    "ST_Slope": "Slope of the peak-exercise ST segment: Up, Flat, Down",
    "HeartDisease": "Target: 1 = heart disease, 0 = normal",
}
EXAMPLES = {
    "Lower-risk example": dict(Age=40, Sex="F", ChestPainType="ATA", RestingBP=120, Cholesterol=200,
                               chol_missing=False, FastingBS=0, RestingECG="Normal", MaxHR=170,
                               ExerciseAngina="N", Oldpeak=0.0, ST_Slope="Up"),
    "Higher-risk example": dict(Age=63, Sex="M", ChestPainType="ASY", RestingBP=150, Cholesterol=260,
                                chol_missing=False, FastingBS=1, RestingECG="ST", MaxHR=110,
                                ExerciseAngina="Y", Oldpeak=2.0, ST_Slope="Flat"),
}
DEFAULTS = dict(Age=54, Sex="M", ChestPainType="ASY", RestingBP=130, Cholesterol=220, chol_missing=False,
                FastingBS=0, RestingECG="Normal", MaxHR=140, ExerciseAngina="N", Oldpeak=0.5, ST_Slope="Flat")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1200px;}
      .card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 1rem 1.2rem;
             background: rgba(46,134,171,.05); height: 100%;}
      .card h4 {margin: 0 0 .4rem 0; font-size: 1.02rem;}
      .card p {margin: 0; font-size: .92rem; opacity: .9;}
      .disclaimer {border-left: 5px solid #D1495B; background: rgba(209,73,91,.08);
                   padding: .8rem 1rem; border-radius: 6px; font-size: .92rem;}
      .risk-pill {display:inline-block; padding:.25rem .8rem; border-radius:999px; color:#fff;
                  font-weight:600;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_resource(show_spinner=False)
def get_explainer():
    background = pd.read_csv(config.SHAP_BACKGROUND_PATH)
    explainer, _ = explain.build_explainer(get_model(), background)
    return explainer


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_FILE)


@st.cache_data(show_spinner=False)
def get_json(path_str: str) -> dict:
    from pathlib import Path
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def disclaimer() -> None:
    st.markdown(f'<div class="disclaimer">⚠️ <b>Disclaimer.</b> {config.DISCLAIMER}</div>',
                unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def risk_pill(label: str) -> str:
    return f'<span class="risk-pill" style="background:{RISK_COLORS[label]}">{label} risk</span>'


def gauge(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Predicted probability of heart disease"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#264653"},
            "steps": [
                {"range": [lo * 100, min(hi, 1) * 100], "color": RISK_COLORS[lab]}
                for lo, hi, lab in config.RISK_BANDS
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home(meta: dict) -> None:
    st.title("🩺 Disease Prediction & Risk Assessment System")
    st.caption("An end-to-end, explainable machine-learning demonstration for heart disease risk estimation.")
    disclaimer()
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("🎯 Purpose", "Estimate the probability that a patient record belongs to the "
                          "<i>heart disease</i> class, using 11 routinely collected clinical attributes.")
    with c2:
        card("⚙️ How it works", "Inputs are validated, passed through the same preprocessing pipeline "
                               "used during training, scored by the trained model and explained with SHAP.")
    with c3:
        card("🔍 Explainable", "Every prediction shows which inputs pushed the model's output up or down "
                              "(model contributions, not medical causes).")

    st.subheader("How the system works")
    steps = ["Patient inputs", "Input validation", "Preprocessing pipeline", "Trained model",
             "Probability + risk band", "SHAP explanation", "Result dashboard"]
    st.markdown(" → ".join(f"**{s}**" for s in steps))

    st.subheader("Dataset")
    ds = meta.get("dataset", {})
    st.markdown(
        f"- **Source:** {ds.get('source', 'Kaggle Heart Failure Prediction Dataset')} "
        f"([link]({ds.get('url', 'https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction')}))\n"
        f"- **License:** {ds.get('license', 'see Kaggle page')}\n"
        f"- **Labelled records used:** {ds.get('labelled_records', '—')} "
        f"(train {ds.get('train_records', '—')} / test {ds.get('test_records', '—')})\n"
        f"- **Target:** `HeartDisease` (1 = heart disease, 0 = normal)"
    )
    st.subheader("Machine-learning approach")
    st.markdown(
        "Eight classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, "
        "XGBoost, LightGBM) were compared using stratified 5-fold cross-validation. The three strongest were "
        "tuned with RandomizedSearchCV and the final model was chosen on **cross-validated ROC-AUC with "
        "recall as tie-breaker** — not on accuracy alone — then evaluated once on a held-out 20 % test split."
    )
    if meta:
        st.info(f"Deployed model: **{meta['model_name']}** · version {meta['version']} · "
                f"trained {meta['training_date_utc'][:10]}")


def patient_form() -> dict | None:
    """Render the input form; return the record when submitted."""
    st.markdown("Load an example profile or enter values manually. All fields except cholesterol are required.")
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.9, 2])
    for col, name in zip((b1, b2), EXAMPLES):
        if col.button(name, width="stretch"):
            for k, v in EXAMPLES[name].items():
                st.session_state[f"in_{k}"] = v
    if b3.button("Reset", width="stretch"):
        for k, v in DEFAULTS.items():
            st.session_state[f"in_{k}"] = v
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"in_{k}", v)

    with st.form("patient_form"):
        st.markdown("##### 👤 Demographics")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18, 100, step=1, key="in_Age",
                              help="Training data covers ages 28–77; predictions outside that range are extrapolations.")
        sex = c2.radio("Sex", ["M", "F"], horizontal=True, key="in_Sex",
                       format_func=lambda v: "Male" if v == "M" else "Female")

        st.markdown("##### 🫀 Symptoms & vitals")
        c1, c2, c3 = st.columns(3)
        cp = c1.selectbox("Chest pain type", list(CHEST_PAIN), key="in_ChestPainType",
                          format_func=CHEST_PAIN.get)
        rbp = c2.number_input("Resting blood pressure (mm Hg)", 60, 220, step=1, key="in_RestingBP")
        angina = c3.radio("Exercise-induced angina", ["N", "Y"], horizontal=True, key="in_ExerciseAngina",
                          format_func=lambda v: "Yes" if v == "Y" else "No")

        st.markdown("##### 🧪 Laboratory")
        c1, c2, c3 = st.columns(3)
        chol = c1.number_input("Serum cholesterol (mg/dl)", 80, 650, step=1, key="in_Cholesterol")
        chol_missing = c2.checkbox("Cholesterol not measured", key="in_chol_missing",
                                   help="The model will impute the training-set median.")
        fbs = c3.radio("Fasting blood sugar > 120 mg/dl", [0, 1], horizontal=True, key="in_FastingBS",
                       format_func=lambda v: "Yes" if v == 1 else "No")

        st.markdown("##### 📈 ECG & exercise test")
        c1, c2, c3, c4 = st.columns(4)
        ecg = c1.selectbox("Resting ECG", list(RESTING_ECG), key="in_RestingECG", format_func=RESTING_ECG.get)
        maxhr = c2.number_input("Max heart rate (bpm)", 50, 220, step=1, key="in_MaxHR")
        oldpeak = c3.number_input("Oldpeak (ST depression, mm)", -3.0, 7.0, step=0.1, format="%.1f",
                                  key="in_Oldpeak")
        slope = c4.selectbox("ST slope", list(ST_SLOPE), key="in_ST_Slope", format_func=ST_SLOPE.get)

        submitted = st.form_submit_button("🔍 Predict Disease Risk", type="primary", width="stretch")

    if not submitted:
        return None
    return {
        "Age": int(age), "Sex": sex, "ChestPainType": cp, "RestingBP": int(rbp),
        "Cholesterol": None if chol_missing else int(chol), "FastingBS": int(fbs),
        "RestingECG": ecg, "MaxHR": int(maxhr), "ExerciseAngina": angina,
        "Oldpeak": float(oldpeak), "ST_Slope": slope,
    }


def show_result_summary(result: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted class", result["label"])
    c2.metric("Probability of heart disease", f"{result['probability']:.1%}")
    c3.metric("Risk category", result["risk_category"])
    c4.metric("Model confidence", f"{result['confidence']:.1%}",
              help="Probability the model assigns to its predicted class. High confidence ≠ clinical certainty.")


def page_prediction() -> None:
    st.title("🔍 Patient Risk Prediction")
    disclaimer()
    st.write("")
    record = patient_form()
    if record is not None:
        try:
            result = predict_record(record)
        except ValueError as exc:
            st.error(f"Input validation failed: {exc}")
            return
        st.session_state["last_record"] = record
        st.session_state["last_result"] = result
        st.session_state.pop("last_explanation", None)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.divider()
        st.subheader("Result")
        st.markdown(risk_pill(result["risk_category"]), unsafe_allow_html=True)
        st.write("")
        show_result_summary(result)
        used = st.session_state["last_record"]
        st.caption("Values used for this prediction: " + " · ".join(
            f"{k} = {'not measured' if v is None else v}" for k, v in used.items())
            + ". If a field showed a range error, its last valid value was used.")
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(gauge(result["probability"]), width="stretch")
        with c2:
            st.markdown("**Risk bands used by this demo**")
            st.markdown("\n".join(f"- {lab}: {lo:.0%} – {min(hi, 1):.0%}" for lo, hi, lab in config.RISK_BANDS))
            st.caption("Bands are illustrative cut-offs on the model's probability, not clinical risk "
                       "categories. The class decision uses a 0.5 threshold.")
            st.info("See **Risk Assessment** and **Explainable AI** in the sidebar for details on this prediction.")


def page_risk_assessment() -> None:
    st.title("📊 Risk Assessment")
    disclaimer()
    if "last_result" not in st.session_state:
        st.warning("No prediction yet. Go to **Prediction**, enter patient values and click *Predict Disease Risk*.")
        return
    record, result = st.session_state["last_record"], st.session_state["last_result"]
    st.write("")
    show_result_summary(result)
    st.progress(result["probability"], text=f"Model probability: {result['probability']:.1%}")

    st.subheader("Entered values")
    shown = {k: ("not measured" if v is None else v) for k, v in record.items()}
    shown["MaxHR as % of age-predicted max"] = f"{record['MaxHR'] / (220 - record['Age']):.0%}"
    st.dataframe(pd.DataFrame([shown]).T.rename(columns={0: "Value"}).astype(str), width="stretch")

    st.subheader("Where does this patient sit in the training data?")
    st.caption("Distributions of each numeric feature by outcome in the labelled dataset; the black line marks "
               "the entered value. This is descriptive context, not a diagnosis.")
    df = get_dataset().copy()
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = np.nan
    df["Outcome"] = df[config.TARGET].map(config.TARGET_LABELS)
    cols = st.columns(2)
    for i, feat in enumerate(config.NUMERIC_FEATURES):
        fig = px.histogram(df, x=feat, color="Outcome", barmode="overlay", opacity=0.6, nbins=30,
                           color_discrete_map={"No Heart Disease": "#2E86AB", "Heart Disease": "#D1495B"})
        if record.get(feat) is not None:
            fig.add_vline(x=record[feat], line_width=3, line_color="black")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title=feat,
                          legend=dict(orientation="h", y=-0.3), yaxis_title="Patients")
        cols[i % 2].plotly_chart(fig, width="stretch")


def page_model_info(meta: dict) -> None:
    st.title("🧠 Model Information")
    if not meta:
        st.error("Model metadata not found. Run `python train.py`.")
        return
    test = meta["evaluation_metrics"]["held_out_test"]
    cv = meta["evaluation_metrics"]["cross_validation"]
    st.subheader(f"Selected model: {meta['model_name']}")
    c = st.columns(6)
    for col, (label, key) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                     ("Specificity", "specificity"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{test[key]:.3f}")
    st.caption(f"Held-out stratified test split (n={meta['dataset']['test_records']}), threshold 0.5. "
               f"5-fold CV on the training split: ROC-AUC {cv['cv_roc_auc']:.3f} ± {cv['cv_roc_auc_std']:.3f}, "
               f"F1 {cv['cv_f1']:.3f} ± {cv['cv_f1_std']:.3f}, recall {cv['cv_recall']:.3f} ± {cv['cv_recall_std']:.3f}.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Model comparison", "📉 Evaluation plots", "⚙️ Hyperparameters",
                                      "🗂️ Training info"])
    with tab1:
        if config.COMPARISON_PATH.exists():
            comp = pd.read_csv(config.COMPARISON_PATH)
            show = comp[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC",
                         "CV ROC-AUC (mean)", "CV ROC-AUC (std)", "CV F1 (mean)", "Training time (s)"]]
            st.dataframe(show.style.format(precision=3).highlight_max(
                subset=["CV ROC-AUC (mean)", "Recall", "F1 Score"], color="#d4efdf"), width="stretch")
            st.caption("Baseline (untuned) models. Held-out columns use the 20 % test split; CV columns use "
                       "5-fold stratified CV on the 80 % training split.")
        tuning = config.REPORTS_DIR / "tuning_results.csv"
        if tuning.exists():
            st.markdown("**After hyperparameter tuning (top-3 candidates)**")
            st.dataframe(pd.read_csv(tuning).style.format(precision=4), width="stretch")
    with tab2:
        figs = [("13_confusion_matrix.png", "Confusion matrix"), ("14_roc_curve.png", "ROC curve"),
                ("15_precision_recall_curve.png", "Precision-Recall curve"),
                ("16_threshold_analysis.png", "Threshold trade-off")]
        cols = st.columns(2)
        for i, (fname, cap) in enumerate(figs):
            path = config.FIGURES_DIR / fname
            if path.exists():
                cols[i % 2].image(str(path), caption=cap, width="stretch")
    with tab3:
        st.json(meta["hyperparameters"])
    with tab4:
        st.markdown("**Preprocessing steps**")
        st.markdown("\n".join(f"{i + 1}. {s}" for i, s in enumerate(meta["preprocessing_steps"])))
        st.markdown(f"**Engineered features:** {', '.join(meta['engineered_features'])}")
        st.markdown(f"**Version:** {meta['version']} · **Trained (UTC):** {meta['training_date_utc']} · "
                    f"**random_state:** {meta['random_state']}")
        st.json(meta["library_versions"])


def page_explainability(meta: dict) -> None:
    st.title("💡 Explainable AI")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) splits a prediction into additive **contributions from each "
        "input feature**, relative to the model's average output. \n\n"
        "> ⚠️ These are **model feature contributions** — they describe how *this model* uses the data. "
        "They are **not medical causes** and must not be interpreted as such."
    )
    tab1, tab2 = st.tabs(["🧍 This prediction", "🌍 Global importance"])
    with tab1:
        if "last_record" not in st.session_state:
            st.warning("Make a prediction first on the **Prediction** page.")
        else:
            if "last_explanation" not in st.session_state:
                with st.spinner("Computing SHAP explanation…"):
                    frame = record_to_frame(st.session_state["last_record"])
                    st.session_state["last_explanation"] = explain.explain_single(get_model(), get_explainer(), frame)
            exp = st.session_state["last_explanation"]
            result = st.session_state["last_result"]
            st.markdown(f"**Why did the model predict _{result['label']}_ "
                        f"(probability {result['probability']:.1%})?**")
            top = exp.head(5)
            st.markdown("Top contributing features:\n" + "\n".join(
                f"{i + 1}. **{r.feature}** = {r.value} → {r.direction}" for i, r in top.iterrows()))
            plot_df = exp.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=plot_df["contribution"], y=[f"{f} = {v}" for f, v in zip(plot_df["feature"], plot_df["value"])],
                orientation="h",
                marker_color=["#D1495B" if c > 0 else "#2E86AB" for c in plot_df["contribution"]],
                text=[f"{c:+.3f}" for c in plot_df["contribution"]], textposition="outside",
            ))
            unit = "probability" if meta.get("shap_explainer") in ("tree", "kernel") else "log-odds"
            fig.update_layout(height=430, margin=dict(l=10, r=40, t=30, b=10),
                              xaxis_title=f"SHAP contribution ({unit} units); red ↑ risk, blue ↓ risk",
                              title="Feature contributions for this patient")
            st.plotly_chart(fig, width="stretch")
            st.caption("One-hot encoded columns are summed back to their original feature. "
                       "MaxHR_pct_predicted is the engineered ratio MaxHR / (220 − Age).")
    with tab2:
        imp_path = config.REPORTS_DIR / "shap_global_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path).sort_values("mean_abs_shap")
            fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                         labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
                         title="Average impact on the model output (held-out test split)")
            fig.update_traces(marker_color="#D1495B")
            fig.update_layout(height=430)
            st.plotly_chart(fig, width="stretch")
        bees = config.FIGURES_DIR / "17_shap_summary_beeswarm.png"
        if bees.exists():
            st.image(str(bees), caption="SHAP summary plot: each dot is one patient; colour = feature value.",
                     width="stretch")


def page_dataset(meta: dict) -> None:
    st.title("🗃️ Dataset")
    df = get_dataset()
    c = st.columns(4)
    c[0].metric("Labelled records", len(df))
    c[1].metric("Input features", len(config.RAW_FEATURES))
    c[2].metric("Target", config.TARGET)
    c[3].metric("Positive class share", f"{df[config.TARGET].mean():.1%}")

    st.markdown(
        "The **Heart Failure Prediction Dataset** (fedesoriano, Kaggle, Sept 2021) combines five heart disease "
        "cohorts (Cleveland, Hungarian, Switzerland, Long Beach VA and Statlog) on 11 common features. "
        "The copy used here provides a labelled file of 734 records and an unlabelled file of 184 records "
        "(no target, therefore used only for demonstration scoring)."
    )
    c1, c2 = st.columns([1, 1.4])
    counts = df[config.TARGET].map(config.TARGET_LABELS).value_counts().reset_index()
    counts.columns = ["Outcome", "Patients"]
    fig = px.pie(counts, names="Outcome", values="Patients", hole=0.45, title="Class distribution",
                 color="Outcome", color_discrete_map={"No Heart Disease": "#2E86AB", "Heart Disease": "#D1495B"})
    c1.plotly_chart(fig, width="stretch")
    feat = c2.selectbox("Explore a feature", config.RAW_FEATURES, index=10)
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df[config.TARGET].map(config.TARGET_LABELS)
    if feat in config.NUMERIC_FEATURES:
        fig = px.box(plot_df, x="Outcome", y=feat, color="Outcome", points="outliers",
                     color_discrete_map={"No Heart Disease": "#2E86AB", "Heart Disease": "#D1495B"},
                     title=f"{feat} by outcome (raw values)")
    else:
        fig = px.histogram(plot_df, x=feat, color="Outcome", barmode="group", text_auto=True,
                           color_discrete_map={"No Heart Disease": "#2E86AB", "Heart Disease": "#D1495B"},
                           title=f"{feat} by outcome")
    c2.plotly_chart(fig, width="stretch")
    if feat == "Cholesterol":
        c2.warning("129 records have Cholesterol = 0, which is physiologically impossible and encodes "
                   "'not recorded'. The pipeline treats these as missing.")

    with st.expander("📖 Data dictionary", expanded=False):
        rows = []
        for col in df.columns:
            rows.append({"Column": col, "Description": FEATURE_DESCRIPTIONS.get(col, ""),
                         "Type": ("Numeric" if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2
                                  else "Categorical / binary"),
                         "Example": str(df[col].iloc[0]),
                         "Role": "Target" if col == config.TARGET else "Feature"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("🔎 Sample records"):
        st.dataframe(df.head(20), width="stretch")


def page_about() -> None:
    st.title("ℹ️ About the Project")
    tabs = st.tabs(["Problem", "Objectives", "Methodology", "Technologies", "Limitations", "Future scope",
                    "Privacy"])
    with tabs[0]:
        st.markdown(
            "Cardiovascular diseases are the leading cause of death globally. Clinicians routinely collect "
            "simple measurements (age, blood pressure, cholesterol, ECG and exercise-test results). This project "
            "investigates how well machine-learning models can separate records with and without heart disease "
            "using only these attributes, and how their predictions can be explained transparently.")
    with tabs[1]:
        st.markdown(
            "- Build a reproducible, leakage-free ML pipeline on a public dataset\n"
            "- Compare multiple classifiers using metrics suited to healthcare screening\n"
            "- Explain predictions with SHAP\n"
            "- Deliver an interactive, privacy-respecting demo deployable on Hugging Face Spaces")
    with tabs[2]:
        st.markdown(
            "1. Dataset inspection & profiling → 2. Cleaning (zero-coded missing values) → 3. EDA → "
            "4. Feature engineering (MaxHR % of age-predicted maximum) → 5. Stratified 80/20 split → "
            "6. Pipeline (imputation, scaling, one-hot) → 7. 8 baseline models with 5-fold CV → "
            "8. RandomizedSearchCV on the top 3 → 9. CV-based model selection → 10. Held-out evaluation → "
            "11. SHAP → 12. Serialisation → 13. Streamlit app → 14. Hugging Face deployment")
    with tabs[3]:
        st.markdown("Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib · "
                    "Seaborn · Plotly · Streamlit · joblib · Hugging Face Spaces")
    with tabs[4]:
        st.markdown(
            "- Small dataset (734 labelled records) from 1980s–90s cohorts; not representative of any current "
            "population\n- Class prevalence (55 %) is far higher than in the general population, so probabilities "
            "are not calibrated to real-world risk\n- Missing cholesterol is concentrated in specific source "
            "cohorts\n- No external validation; performance on other hospitals is unknown\n"
            "- Feature contributions are associations, not causes")
    with tabs[5]:
        st.markdown("- External validation on independent data\n- Probability calibration for a target "
                    "population\n- Fairness analysis across sex and age groups\n- Additional clinical "
                    "variables and longitudinal outcomes\n- Clinician-in-the-loop usability studies")
    with tabs[6]:
        st.markdown("- No names, contact details or other identifiers are collected\n- Inputs are kept only in "
                    "the current session's memory and are never stored or logged by the app\n- The dataset is "
                    "public and de-identified\n- No secrets or tokens are included in the source code")
    disclaimer()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def apply_deep_link(page_labels: list[str]) -> None:
    """Support demo links such as ``?page=explainable&demo=high``.

    ``page`` selects a sidebar page by a word in its label; ``demo`` (``high`` or
    ``low``) pre-runs the matching example profile. Applied once per session.
    """
    if st.session_state.get("_deep_link_done"):
        return
    st.session_state["_deep_link_done"] = True
    params = st.query_params
    wanted = params.get("page", "").lower()
    match = next((p for p in page_labels if wanted and wanted in p.lower()), None)
    if match:
        st.session_state["nav"] = match
    demo = {"high": "Higher-risk example", "low": "Lower-risk example"}.get(params.get("demo", "").lower())
    if demo:
        example = EXAMPLES[demo]
        for k, v in example.items():
            st.session_state[f"in_{k}"] = v
        record = {k: v for k, v in example.items() if k != "chol_missing"}
        if example["chol_missing"]:
            record["Cholesterol"] = None
        st.session_state["last_record"] = record
        st.session_state["last_result"] = predict_record(record)


def main() -> None:
    try:
        get_model()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    meta = load_metadata()

    pages = {
        "🏠 Home": lambda: page_home(meta),
        "🔍 Prediction": page_prediction,
        "📊 Risk Assessment": page_risk_assessment,
        "🧠 Model Information": lambda: page_model_info(meta),
        "💡 Explainable AI": lambda: page_explainability(meta),
        "🗃️ Dataset": lambda: page_dataset(meta),
        "ℹ️ About Project": page_about,
    }
    apply_deep_link(list(pages))
    with st.sidebar:
        st.markdown("## 🩺 Heart Disease Risk")
        choice = st.radio("Navigate", list(pages), label_visibility="collapsed", key="nav")
        st.divider()
        if meta:
            st.caption(f"Model: **{meta['model_name']}** v{meta['version']}")
            st.caption(f"Held-out ROC-AUC: {meta['evaluation_metrics']['held_out_test']['roc_auc']:.3f}")
        st.caption("Academic demonstration — not a medical device.")
    pages[choice]()


main()

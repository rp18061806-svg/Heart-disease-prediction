---
title: Disease Prediction & Risk Assessment
emoji: 🩺
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Explainable heart disease risk demo (academic, not medical)
---

# Disease Prediction & Risk Assessment System

> ⚠️ **Disclaimer.** This application is an academic machine-learning demonstration and is not a medical
> diagnostic system. Predictions should not be used for diagnosis, treatment, or medical decision-making.
> Always consult a qualified healthcare professional for medical advice.

## 1. Project Overview
An end-to-end, explainable machine-learning project that estimates the probability of **heart disease**
from 11 routinely collected clinical attributes. It covers dataset profiling, leakage-free preprocessing,
EDA, feature engineering, comparison of 8 classifiers with cross-validation, hyperparameter tuning, held-out
evaluation, SHAP explanations, and an interactive Streamlit dashboard deployable on Hugging Face Spaces.

## 2. Demo
* **Live app:** `https://<your-username>-disease-risk-assessment.hf.space` *(replace after deploying — see §14)*
* **Locally:** `streamlit run app.py` → http://localhost:8501
* Try **Prediction → "Higher-risk example" → Predict Disease Risk**, then open **Explainable AI**.

## 3. Features
* 🔍 Patient form auto-built from the dataset schema, with validation and an optional "cholesterol not measured"
* 📈 Predicted class, probability gauge, Low / Moderate / High band, model confidence
* 📊 Risk-assessment view comparing the patient with the training distributions
* 💡 Per-prediction SHAP explanation ("why did the model predict this?") + global importance
* 🧠 Model information: comparison table, tuning results, confusion matrix, ROC, PR and threshold curves
* 🗃️ Dataset explorer and data dictionary
* 🔒 No personal data collected or stored; prominent disclaimer

## 4. Dataset
| | |
|---|---|
| Name | Heart Failure Prediction Dataset |
| Records used | 734 labelled (`heart_train.csv`) + 184 unlabelled (`heart_test.csv`, demo scoring only) |
| Features | Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope |
| Target | `HeartDisease` (1 = heart disease 55.3 %, 0 = normal 44.7 %) |
| Key quality issue | 129 records (17.6 %) have `Cholesterol = 0`, meaning *not recorded* → treated as missing |

The Kaggle dataset combines five UCI heart disease cohorts (Cleveland, Hungarian, Switzerland,
Long Beach VA, Statlog), totalling 918 records after de-duplication. Details:
[documentation/04_Dataset_Documentation.md](documentation/04_Dataset_Documentation.md) and
[05_Data_Dictionary.md](documentation/05_Data_Dictionary.md).

## 5. Dataset Source
fedesoriano. (September 2021). *Heart Failure Prediction Dataset*. Kaggle.
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## 6. Dataset License
As listed on the Kaggle dataset page: **"Database: Open Database, Contents: © Original Authors"** — i.e. the
database is under the Open Database License (ODbL) and the contents remain © their original authors.
The MIT licence in `LICENSE` covers this project's **source code only**, not the dataset.

## 7. Technologies
Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib · Seaborn · Plotly ·
Streamlit · joblib · pytest · Docker · Hugging Face Spaces

## 8. Machine Learning Models
Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, SVM (RBF), Gradient Boosting,
XGBoost, LightGBM — all in the same `Pipeline` (median imputation, scaling, one-hot encoding, engineered
`MaxHR_pct_predicted = MaxHR / (220 − Age)`), compared with stratified 5-fold CV. The top 3 were tuned with
`RandomizedSearchCV`. **Selection rule:** highest CV ROC-AUC, ties (≤ 0.005) broken by CV recall → **Random Forest**.

| Model (baseline) | CV ROC-AUC | Held-out ROC-AUC | Held-out Recall | Held-out F1 |
|---|---|---|---|---|
| Random Forest | 0.928 ± 0.020 | 0.905 | 0.877 | 0.845 |
| Gradient Boosting | 0.921 ± 0.012 | 0.905 | 0.889 | 0.862 |
| LightGBM | 0.921 ± 0.021 | 0.916 | 0.926 | 0.888 |
| Logistic Regression | 0.920 ± 0.019 | 0.897 | 0.815 | 0.810 |
| XGBoost | 0.918 ± 0.017 | 0.907 | 0.889 | 0.852 |
| SVM | 0.916 ± 0.028 | 0.891 | 0.877 | 0.845 |
| KNN | 0.892 ± 0.027 | 0.881 | 0.889 | 0.857 |
| Decision Tree | 0.778 ± 0.042 | 0.786 | 0.815 | 0.810 |

## 9. Model Evaluation (final tuned Random Forest, held-out test split, n = 147)
| Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.830 | 0.826 | **0.877** | 0.773 | 0.850 | **0.906** (95 % CI 0.849–0.952) |

Confusion matrix: TN 51 · FP 15 · FN 10 · TP 71. 5-fold CV on the training split: ROC-AUC 0.931 ± 0.021.
These numbers describe performance on this historical dataset only — **not clinical validity**.

## 10. Explainable AI
SHAP `TreeExplainer` (probability units), with one-hot columns summed back to original features.
Global ranking: **ST_Slope** (0.193) ≫ ChestPainType (0.074) > ExerciseAngina (0.057) > Oldpeak > Sex > MaxHR …
The app explains every individual prediction. *These are model feature contributions, not medical causes.*

## 11. Application Workflow
```
User → Streamlit UI → Input validation → Cleaning (0 → missing) → Feature engineering + encoding
     → Random Forest → Probability → Risk band → SHAP explanation → Result dashboard
```
![Architecture](reports/architecture.png)

## 12. Installation
```bash
git clone <your-repo-url> Disease-Prediction-Risk-Assessment
cd Disease-Prediction-Risk-Assessment
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt    # full dev/training environment
# or: pip install -r requirements.txt  # app only
```

## 13. Local Execution
```bash
python -m src.profile_dataset     # Phase 1 dataset profile → reports/dataset_profile.json
python -m src.eda                 # EDA figures → reports/figures/
python train.py                   # full training (~5–15 min; --quick for a smoke test)
python -m src.build_notebooks     # regenerate + execute the 4 notebooks
python -m src.diagrams            # architecture/workflow diagrams
pytest -q                         # 45 automated tests
streamlit run app.py              # launch the app
python predict.py --csv data/raw/heart_test.csv --out predictions.csv   # batch CLI
```
The trained model is already included, so `streamlit run app.py` works immediately after installing.

## 14. Hugging Face Deployment
Create a **Docker** Space (Streamlit template), push this repository (the YAML header above configures
`sdk: docker`, `app_port: 8501`), wait for *Running*. The `Dockerfile` runs `streamlit run app.py`.
Full step-by-step guide and troubleshooting: [documentation/15_HuggingFace_Deployment.md](documentation/15_HuggingFace_Deployment.md).

## 15. Project Structure
```
Disease-Prediction-Risk-Assessment/
├── app.py  train.py  predict.py  config.py
├── requirements.txt  requirements-dev.txt  packages.txt  Dockerfile  .dockerignore
├── README.md  LICENSE  .gitignore  .streamlit/config.toml
├── src/            preprocessing · eda · modeling · explain · profile_dataset · diagrams · build_notebooks
├── data/raw/       heart_train.csv · heart_test.csv
├── data/processed/ train_split.csv · test_split.csv · unlabeled_predictions.csv
├── notebooks/      01_data_analysis · 02_eda · 03_model_training · 04_model_evaluation (executed)
├── models/         final_model.joblib · model_metadata.json · best_params.json · shap_background.csv
├── reports/        figures/ (18 PNG + 1 HTML) · architecture.png · workflow.png · model_metrics.json
│                   model_comparison.csv · tuning_results.csv · feature_*.csv · dataset_profile.json …
├── documentation/  01_Project_Overview.md … 21_Conclusion.md
├── tests/          test_pipeline.py
├── assets/screenshots/
├── PROJECT_REPORT.md  PRESENTATION_OUTLINE.md  VIVA_QUESTIONS.md  PROJECT_EXPLANATION.md
```

## 16. Limitations
Small (734 labelled) historical referral cohort with 55 % prevalence, 79 % male; no external validation or
probability calibration; fixed 0.5 threshold; limited feature set. See
[18_Limitations.md](documentation/18_Limitations.md).

## 17. Ethical Considerations
No personal data collected or stored; input validation; explanations labelled as non-causal; known biases
disclosed; no secrets in code. See [19_Ethical_Considerations.md](documentation/19_Ethical_Considerations.md).

## 18. Future Scope
External validation, calibration, clinically chosen thresholds, fairness audit, richer features, CI/CD.
See [20_Future_Scope.md](documentation/20_Future_Scope.md).

## 19. Disclaimer
This application is an academic machine-learning demonstration and is not a medical diagnostic system.
Predictions should not be used for diagnosis, treatment, or medical decision-making. Always consult a
qualified healthcare professional for medical advice.

## 20. Author
**[Rupesh Ajay Patil]** — [B.Tech/ Artificial Intelligence], [G H Raisoni collage ], [2027]
Contact: [your GitHub profile URL]

---
### GitHub quick start
```bash
git init
git add .
git commit -m "Disease Prediction & Risk Assessment System"
git branch -M main
git remote add origin https://github.com/<your-username>/Disease-Prediction-Risk-Assessment.git
git push -u origin main
```
Authenticate with your browser/credential manager or a personal access token entered at the prompt —
never commit passwords or tokens.

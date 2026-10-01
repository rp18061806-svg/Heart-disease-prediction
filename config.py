"""Central configuration for the Disease Prediction & Risk Assessment System.

All paths are relative to the project root so the project works after cloning
on any operating system (no hard-coded local paths).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_FILE = RAW_DATA_DIR / "heart_train.csv"
UNLABELED_FILE = RAW_DATA_DIR / "heart_test.csv"  # no target column

MODEL_PATH = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
PROFILE_PATH = REPORTS_DIR / "dataset_profile.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.csv"

# --------------------------------------------------------------------------
# Dataset schema (verified during Phase 1 inspection)
# --------------------------------------------------------------------------
TARGET = "HeartDisease"
TARGET_LABELS = {0: "No Heart Disease", 1: "Heart Disease"}

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
BINARY_NUMERIC_FEATURES = ["FastingBS"]
CATEGORICAL_FEATURES = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
RAW_FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# Features the user may leave blank; they are imputed by the pipeline.
OPTIONAL_FEATURES = ["Cholesterol"]

CATEGORY_LEVELS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ASY", "NAP", "ATA", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"],
}

# Physiologically plausible input ranges used for validation in the app.
# These are sanity bounds (not clinical reference ranges).
VALID_RANGES = {
    "Age": (18, 100),
    "RestingBP": (60, 220),       # mm Hg; 0 is physiologically impossible
    "Cholesterol": (80, 650),     # mg/dl; 0 encodes "not measured"
    "MaxHR": (50, 220),           # beats per minute
    "Oldpeak": (-3.0, 7.0),       # mm ST depression
}

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MODEL_VERSION = "1.0.0"

# Risk bands applied to the predicted probability of heart disease.
RISK_BANDS = [
    (0.00, 0.30, "Low"),
    (0.30, 0.60, "Moderate"),
    (0.60, 1.01, "High"),
]

DISCLAIMER = (
    "This application is an academic machine-learning demonstration and is "
    "not a medical diagnostic system. Predictions should not be used for "
    "diagnosis, treatment, or medical decision-making. Always consult a "
    "qualified healthcare professional for medical advice."
)

"""Inference utilities and command-line interface.

Examples
--------
Single patient (JSON):
    python predict.py --json "{\"Age\": 54, \"Sex\": \"M\", \"ChestPainType\": \"ASY\",
        \"RestingBP\": 140, \"Cholesterol\": 239, \"FastingBS\": 0, \"RestingECG\": \"Normal\",
        \"MaxHR\": 118, \"ExerciseAngina\": \"Y\", \"Oldpeak\": 1.5, \"ST_Slope\": \"Flat\"}"

Batch (CSV with the 11 feature columns):
    python predict.py --csv data/raw/heart_test.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import config
from src.preprocessing import clean_raw_data, validate_patient_input


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model has not been generated yet."""


@lru_cache(maxsize=1)
def load_model():
    """Load the serialised preprocessing + model pipeline (cached)."""
    if not config.MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Run `python train.py` first.")
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not config.METADATA_PATH.exists():
        return {}
    with open(config.METADATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def risk_category(probability: float) -> str:
    """Map P(heart disease) to a Low / Moderate / High band."""
    for low, high, label in config.RISK_BANDS:
        if low <= probability < high:
            return label
    return config.RISK_BANDS[-1][2]


def record_to_frame(record: dict) -> pd.DataFrame:
    """Convert a validated input dict to a one-row, cleaned DataFrame."""
    row = {col: record.get(col, np.nan) for col in config.RAW_FEATURES}
    if row["Cholesterol"] in (None, ""):
        row["Cholesterol"] = np.nan
    frame = pd.DataFrame([row])
    return clean_raw_data(frame, drop_duplicates=False)[config.RAW_FEATURES]


def predict_record(record: dict, threshold: float = 0.5) -> dict:
    """Predict for one patient.

    Returns:
        dict with predicted_class, label, probability, risk_category and
        confidence (probability of the predicted class).

    Raises:
        ValueError: if the input fails validation.
    """
    errors = validate_patient_input(record)
    if errors:
        raise ValueError("; ".join(errors))
    model = load_model()
    frame = record_to_frame(record)
    proba = float(model.predict_proba(frame)[0, 1])
    predicted = int(proba >= threshold)
    return {
        "predicted_class": predicted,
        "label": config.TARGET_LABELS[predicted],
        "probability": proba,
        "risk_category": risk_category(proba),
        "confidence": proba if predicted == 1 else 1 - proba,
        "threshold": threshold,
    }


def predict_frame(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Batch prediction for a DataFrame containing the 11 feature columns."""
    missing = [c for c in config.RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    clean = clean_raw_data(df[config.RAW_FEATURES], drop_duplicates=False)
    proba = load_model().predict_proba(clean)[:, 1]
    out = df.copy()
    out["probability"] = proba.round(4)
    out["predicted_class"] = (proba >= threshold).astype(int)
    out["risk_category"] = [risk_category(p) for p in proba]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart disease risk prediction (academic demo).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="single patient record as a JSON string")
    group.add_argument("--csv", help="CSV file with the 11 feature columns")
    parser.add_argument("--out", help="output CSV path for batch mode")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(config.DISCLAIMER + "\n")
    try:
        if args.json:
            print(json.dumps(predict_record(json.loads(args.json), args.threshold), indent=2))
            return
        result = predict_frame(pd.read_csv(args.csv), args.threshold)
    except (ValueError, ModelNotFoundError) as exc:  # json.JSONDecodeError is a ValueError
        raise SystemExit(f"Error: {exc}")
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote {len(result)} predictions to {args.out}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()

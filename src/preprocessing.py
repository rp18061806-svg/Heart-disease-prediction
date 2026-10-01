"""Data cleaning, feature engineering and the scikit-learn preprocessing pipeline.

Design principles
-----------------
* ``clean_raw_data`` only applies *row-wise, statistics-free* rules
  (whitespace stripping, type coercion, marking impossible values as missing).
  It therefore cannot leak information and is safe to run before splitting.
* Everything that *learns* from data (imputation medians, scaling statistics,
  one-hot vocabularies) lives inside a scikit-learn ``Pipeline`` and is fitted
  on the training split only.
* The same pipeline object is serialised and reused by the Streamlit app, so
  training-time and inference-time preprocessing can never diverge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Values that are physiologically impossible and are known to encode
# "not recorded" in the source cohorts of this dataset.
ZERO_MEANS_MISSING = ["RestingBP", "Cholesterol"]

ENGINEERED_NUMERIC = ["MaxHR_pct_predicted"]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_raw_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply deterministic, row-wise cleaning rules.

    Steps
    1. Strip leading/trailing whitespace from text columns and normalise case
       for known categorical levels (e.g. ``" flat "`` -> ``"Flat"``).
    2. Coerce numeric columns to numbers (invalid strings -> NaN).
    3. Replace physiologically impossible zeros in RestingBP and Cholesterol
       with NaN so they are imputed inside the pipeline instead of being
       treated as real measurements.
    4. Optionally drop exact duplicate rows.

    Args:
        df: Raw data frame (with or without the target column).
        drop_duplicates: Remove exact duplicate records.

    Returns:
        A cleaned copy of ``df``.
    """
    out = df.copy()

    for col in config.CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            lookup = {lvl.lower(): lvl for lvl in config.CATEGORY_LEVELS[col]}
            out[col] = out[col].map(lambda v: lookup.get(str(v).lower(), v) if pd.notna(v) else v)
            out[col] = out[col].astype(object).where(out[col].notna(), np.nan)

    for col in config.NUMERIC_FEATURES + config.BINARY_NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ZERO_MEANS_MISSING:
        if col in out.columns:
            out.loc[out[col] == 0, col] = np.nan

    if config.TARGET in out.columns:
        out[config.TARGET] = pd.to_numeric(out[config.TARGET], errors="coerce")
        out = out.dropna(subset=[config.TARGET])
        out[config.TARGET] = out[config.TARGET].astype(int)

    if drop_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add documented, clinically interpretable engineered features.

    ``MaxHR_pct_predicted``
        Peak heart rate achieved as a fraction of the age-predicted maximum
        (220 - age, the Fox formula). Raw MaxHR is strongly age-dependent;
        this ratio expresses exercise capacity relative to age. It is a
        deterministic row-wise transform, so it cannot leak target information.

    Optionally (ablation only) ``Cholesterol_missing`` - a 0/1 flag for an
    unrecorded cholesterol value. It is **disabled by default** because in
    this dataset missingness is an artefact of which source cohort a record
    came from, not a physiological signal.
    """

    def __init__(self, add_hr_ratio: bool = True, add_chol_missing_flag: bool = False):
        self.add_hr_ratio = add_hr_ratio
        self.add_chol_missing_flag = add_chol_missing_flag

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        """Stateless transformer; nothing is learned."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with engineered columns appended."""
        X = X.copy()
        if self.add_hr_ratio:
            X["MaxHR_pct_predicted"] = X["MaxHR"] / (220.0 - X["Age"])
        if self.add_chol_missing_flag:
            X["Cholesterol_missing"] = X["Cholesterol"].isna().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else list(config.RAW_FEATURES)
        if self.add_hr_ratio:
            names.append("MaxHR_pct_predicted")
        if self.add_chol_missing_flag:
            names.append("Cholesterol_missing")
        return np.asarray(names, dtype=object)


def numeric_columns(add_hr_ratio: bool = True, add_chol_missing_flag: bool = False) -> list:
    """Numeric columns entering the ColumnTransformer."""
    cols = config.NUMERIC_FEATURES + config.BINARY_NUMERIC_FEATURES
    if add_hr_ratio:
        cols = cols + ["MaxHR_pct_predicted"]
    if add_chol_missing_flag:
        cols = cols + ["Cholesterol_missing"]
    return cols


def build_preprocessor(scale_numeric: bool = True, add_hr_ratio: bool = True,
                       add_chol_missing_flag: bool = False) -> Pipeline:
    """Build the feature-engineering + ColumnTransformer preprocessing pipeline.

    Numeric: median imputation (robust to skew) -> optional standard scaling.
    Categorical: most-frequent imputation -> one-hot encoding with
    ``handle_unknown='ignore'`` so unseen categories never crash inference.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_tf = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_columns(add_hr_ratio, add_chol_missing_flag)),
            ("cat", cat_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("features", FeatureEngineer(add_hr_ratio, add_chol_missing_flag)),
        ("columns", column_tf),
    ])


def split_features_target(df: pd.DataFrame):
    """Separate the modelling features from the target column."""
    return df[config.RAW_FEATURES].copy(), df[config.TARGET].copy()


def validate_patient_input(record: dict) -> list[str]:
    """Validate a single patient record from the UI or API.

    Returns a list of human-readable error messages (empty when valid).
    """
    errors = []
    for col in config.RAW_FEATURES:
        if col in config.OPTIONAL_FEATURES:
            continue
        if col not in record or record[col] is None or record[col] == "":
            errors.append(f"'{col}' is required.")
    for col, (low, high) in config.VALID_RANGES.items():
        value = record.get(col)
        if value is None or value == "":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be numeric.")
            continue
        if not (low <= value <= high):
            errors.append(f"'{col}'={value:g} is outside the accepted range [{low}, {high}].")
    for col, levels in config.CATEGORY_LEVELS.items():
        value = record.get(col)
        if value not in (None, "") and value not in levels:
            errors.append(f"'{col}' must be one of {levels}.")
    if record.get("FastingBS") not in (None, "") and record.get("FastingBS") not in (0, 1):
        errors.append("'FastingBS' must be 0 or 1.")
    return errors

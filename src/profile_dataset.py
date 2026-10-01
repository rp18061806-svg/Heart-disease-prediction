"""Phase 1 - Dataset inspection.

Generates a schema-agnostic profile of the raw CSV files (no assumptions about
column names or target) and writes it to reports/dataset_profile.json.

Usage:
    python -m src.profile_dataset
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CORRELATION_THRESHOLD = 0.7
NEAR_CONSTANT_THRESHOLD = 0.95  # share of the most frequent value


def iqr_outliers(series: pd.Series) -> dict:
    """Return Tukey (1.5 x IQR) fence information for a numeric series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < low) | (series > high)
    return {
        "lower_fence": round(float(low), 3),
        "upper_fence": round(float(high), 3),
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(100 * float(mask.mean()), 2),
    }


def profile_frame(df: pd.DataFrame, name: str) -> dict:
    """Build a profile dictionary for one DataFrame."""
    profile = {
        "file_name": name,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns_detail": {},
    }
    for col in df.columns:
        s = df[col]
        top_share = float(s.value_counts(normalize=True, dropna=False).iloc[0])
        info = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * float(s.isna().mean()), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "top_value_share": round(top_share, 4),
            "constant": bool(s.nunique(dropna=False) <= 1),
            "near_constant": bool(top_share >= NEAR_CONSTANT_THRESHOLD),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            info.update({
                "min": float(desc["min"]), "q1": float(desc["25%"]),
                "median": float(desc["50%"]), "mean": round(float(desc["mean"]), 3),
                "q3": float(desc["75%"]), "max": float(desc["max"]),
                "std": round(float(desc["std"]), 3),
                "skew": round(float(s.skew()), 3),
                "n_zeros": int((s == 0).sum()),
                "n_negative": int((s < 0).sum()),
                "iqr_outliers": iqr_outliers(s),
            })
            if s.nunique() <= 10:
                info["value_counts"] = {str(k): int(v) for k, v in s.value_counts().sort_index().items()}
        else:
            stripped = s.astype(str).str.strip()
            info.update({
                "cardinality": int(s.nunique()),
                "value_counts": {str(k): int(v) for k, v in s.value_counts().items()},
                "has_leading_trailing_spaces": bool((stripped != s.astype(str)).any()),
            })
        profile["columns_detail"][col] = info
    return profile


def target_candidates(train: pd.DataFrame, unlabeled: pd.DataFrame) -> list:
    """Columns present in the labelled file but absent from the unlabelled file,
    or low-cardinality columns - both are target candidates."""
    missing_in_test = [c for c in train.columns if c not in unlabeled.columns]
    return missing_in_test


def correlations(df: pd.DataFrame) -> dict:
    """Pearson correlations between numeric columns; flag |r| >= threshold."""
    corr = df.select_dtypes(include=np.number).corr()
    high = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= CORRELATION_THRESHOLD:
                high.append({"a": a, "b": b, "r": round(float(corr.loc[a, b]), 3)})
    return {"matrix": corr.round(3).to_dict(), "high_pairs": high}


def main() -> None:
    train = pd.read_csv(config.TRAIN_FILE)
    unlabeled = pd.read_csv(config.UNLABELED_FILE)

    report = {
        "labelled_file": profile_frame(train, config.TRAIN_FILE.name),
        "unlabelled_file": profile_frame(unlabeled, config.UNLABELED_FILE.name),
        "target_candidates": target_candidates(train, unlabeled),
        "correlations_labelled": correlations(train),
        "column_sets_match": list(unlabeled.columns) == [c for c in train.columns if c in unlabeled.columns],
        "overlap_rows_between_files": int(
            pd.merge(train.drop(columns=target_candidates(train, unlabeled)),
                     unlabeled, how="inner").shape[0]
        ),
    }
    for cand in report["target_candidates"]:
        vc = train[cand].value_counts()
        report[f"target_distribution_{cand}"] = {
            "counts": {str(k): int(v) for k, v in vc.items()},
            "pct": {str(k): round(100 * v / len(train), 2) for k, v in vc.items()},
            "imbalance_ratio_major_to_minor": round(float(vc.max() / vc.min()), 3),
        }
        report[f"target_correlation_{cand}"] = (
            train.select_dtypes(include=np.number).corr()[cand].drop(cand).round(3).to_dict()
        )

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROFILE_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

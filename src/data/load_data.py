#!/usr/bin/env python3
"""
Load and validate raw interview data.

Reads data/raw/interview_data.parquet, validates schema and constraints,
and writes validated data to data/interim/loaded_data.parquet.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def load_params(params_path: Path) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def validate_schema(df: pd.DataFrame, params: dict) -> None:
    """Validate DataFrame schema and constraints."""
    required_columns = {
        "question_id": "object",
        "question": "object",
        "reference_answer": "object",
        "candidate_answer": "object",
        "score_grader1": "float64",
        "score_grader2": "float64",
        "score_avg": "float64",
        "quality_label": "int64",
    }

    # Check all required columns exist
    missing = set(required_columns.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check dtypes
    for col, expected_dtype in required_columns.items():
        actual_dtype = str(df[col].dtype)
        if expected_dtype in ("object", "str") and actual_dtype not in ("object", "string", "str"):
            raise ValueError(f"Column '{col}' expected string type, got {actual_dtype}")
        elif expected_dtype == "float64" and actual_dtype not in ("float64", "float32"):
            raise ValueError(f"Column '{col}' expected float64/float32, got {actual_dtype}")
        elif expected_dtype == "int64" and actual_dtype != "int64":
            raise ValueError(f"Column '{col}' expected int64, got {actual_dtype}")

    # No nulls in text columns
    text_cols = ["question_id", "question", "reference_answer", "candidate_answer"]
    null_counts = df[text_cols].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in text columns:\n{null_counts[null_counts > 0]}")

    # Score ranges [0, 5]
    score_cols = ["score_grader1", "score_grader2", "score_avg"]
    for col in score_cols:
        if (df[col] < 0).any() or (df[col] > 5).any():
            raise ValueError(f"Column '{col}' has values outside [0, 5] range")

    # quality_label in {0, 1, 2, 3}
    invalid_labels = df[~df["quality_label"].isin([0, 1, 2, 3])]
    if len(invalid_labels) > 0:
        raise ValueError(f"quality_label has invalid values: {invalid_labels['quality_label'].unique()}")

    logger.info("Schema validation passed")


def log_summary(df: pd.DataFrame, params: dict) -> None:
    """Log dataset summary statistics."""
    logger.info("Row count: %d", len(df))
    logger.info("Unique question_ids: %d", df["question_id"].nunique())

    class_dist = df["quality_label"].value_counts().sort_index()
    logger.info("Class distribution:")
    for cls, count in class_dist.items():
        label = params["data"]["target_classes"][cls]
        logger.info("  Class %d (%s): %d (%.1f%%)", cls, label, count, 100 * count / len(df))

    logger.info("score_avg: mean=%.3f, median=%.3f, std=%.3f",
                df["score_avg"].mean(), df["score_avg"].median(), df["score_avg"].std())


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    params = load_params(project_root / "params.yaml")

    raw_path = project_root / params["data"]["raw_data_path"]
    out_path = project_root / "data/interim/loaded_data.parquet"

    logger.info("Reading raw data from %s", raw_path)
    df = pd.read_parquet(raw_path)

    logger.info("Validating schema...")
    validate_schema(df, params)

    log_summary(df, params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Written validated data to %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
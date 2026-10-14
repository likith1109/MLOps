#!/usr/bin/env python3
"""
Preprocess interview data: clean text and create grouped train/val/test splits.

Uses GroupShuffleSplit to split by question_id to prevent question leakage.
"""

import logging
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def load_params(params_path: Path) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Clean text: strip, normalize unicode, collapse repeated spaces. Keep case."""
    if pd.isna(text):
        return ""
    # Normalize unicode (NFC)
    text = text.strip()
    # Collapse repeated whitespace (including newlines, tabs) to single space
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text_columns(df: pd.DataFrame, text_columns: dict) -> pd.DataFrame:
    """Apply text cleaning to specified columns."""
    df = df.copy()
    for logical_name, col_name in text_columns.items():
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(clean_text)
            logger.debug("Cleaned text column: %s", col_name)
    return df


def grouped_split(
    df: pd.DataFrame,
    group_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by group_column using GroupShuffleSplit in two steps.

    Step 1: Split off test set (test_size fraction of groups)
    Step 2: Split remaining into train and validation (validation_size / (1 - test_size) fraction of remaining groups)

    Returns: (train_df, val_df, test_df)
    """
    groups = df[group_column].values

    # Step 1: test vs (train + val)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: validation vs train from train_val
    # validation fraction of the remaining data
    val_frac = validation_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_val_groups = train_val_df[group_column].values
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def assert_disjoint_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, group_column: str
) -> None:
    """Assert that question_id sets are pairwise disjoint across splits."""
    train_groups = set(train_df[group_column].unique())
    val_groups = set(val_df[group_column].unique())
    test_groups = set(test_df[group_column].unique())

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    if train_val_overlap:
        raise AssertionError(f"Train/Val group overlap: {train_val_overlap}")
    if train_test_overlap:
        raise AssertionError(f"Train/Test group overlap: {train_test_overlap}")
    if val_test_overlap:
        raise AssertionError(f"Val/Test group overlap: {val_test_overlap}")

    logger.info("Split assertions passed: all group sets are disjoint")


def log_split_summary(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    group_column: str, target_column: str, params: dict
) -> None:
    """Log row counts, unique groups, and class distribution for each split."""
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, split_df in splits.items():
        n_rows = len(split_df)
        n_groups = split_df[group_column].nunique()
        class_dist = split_df[target_column].value_counts().sort_index()
        logger.info("%s: %d rows, %d unique %s", name, n_rows, n_groups, group_column)
        for cls, count in class_dist.items():
            label = params["data"]["target_classes"].get(cls, str(cls))
            logger.info("  Class %d (%s): %d (%.1f%%)", cls, label, count, 100 * count / n_rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    params = load_params(project_root / "params.yaml")

    input_path = project_root / "data/interim/loaded_data.parquet"
    processed_dir = project_root / params["data"]["processed_data_dir"]

    logger.info("Reading validated data from %s", input_path)
    df = pd.read_parquet(input_path)

    # Clean text columns
    logger.info("Cleaning text columns...")
    df = clean_text_columns(df, params["data"]["text_columns"])

    # Grouped split
    split_params = params["split"]
    logger.info("Performing grouped split (test_size=%.2f, validation_size=%.2f, random_state=%d)...",
                split_params["test_size"], split_params["validation_size"], split_params["random_state"])

    train_df, val_df, test_df = grouped_split(
        df=df,
        group_column=split_params["group_column"],
        test_size=split_params["test_size"],
        validation_size=split_params["validation_size"],
        random_state=split_params["random_state"],
    )

    # Assertions
    assert_disjoint_splits(train_df, val_df, test_df, split_params["group_column"])

    # Verify all rows preserved
    total_split = len(train_df) + len(val_df) + len(test_df)
    if total_split != len(df):
        raise AssertionError(f"Row count mismatch: input={len(df)}, splits sum={total_split}")
    logger.info("Row count preserved: %d = %d + %d + %d", len(df), len(train_df), len(val_df), len(test_df))

    # Log summaries
    log_split_summary(train_df, val_df, test_df, split_params["group_column"],
                      params["data"]["target_column"], params)

    # Write outputs
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    logger.info("Written splits to %s", processed_dir)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Prepare Mohler dataset for interview answer quality classification.

Reads data/external/mohler/open_ended.parquet, normalizes grader scores to 0-5 scale,
uses source score_avg (0-5) for quality_label, and writes
data/raw/interview_data.parquet and data/raw/interview_data.csv.
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


def extract_question_id(raw_id: str) -> str:
    """Remove the final .Axx part from id (e.g., E01.Q01.A00 -> E01.Q01)."""
    return raw_id.rsplit(".A", 1)[0]


def normalize_grader_scores(
    g1: float, g2: float, score_avg: float
) -> tuple[float, float, str]:
    """
    Normalize grader scores to 0-5 scale per row.

    Returns:
        (norm_g1, norm_g2, category)
    Categories:
        - both_0_5: both graders already 0-5
        - both_0_10: both graders 0-10, divided by 2
        - mixed_g1_10: g1 was 0-10, g2 was 0-5
        - mixed_g2_10: g2 was 0-10, g1 was 0-5
        - default_mixed_g1_10: g1 > 5, g2 <= 5, no formula match, assume g1 0-10
        - default_mixed_g2_10: g2 > 5, g1 <= 5, no formula match, assume g2 0-10
    """
    if g1 <= 5 and g2 <= 5:
        return g1, g2, "both_0_5"

    if g1 > 5 and g2 > 5:
        # Both > 5: check if score_avg matches (g1+g2)/4
        if abs((g1 + g2) / 4 - score_avg) < 0.01:
            return g1 / 2, g2 / 2, "both_0_10"
        # Fallback: assume both 0-10
        return g1 / 2, g2 / 2, "both_0_10"

    if g1 > 5 and g2 <= 5:
        # g1 > 5, g2 <= 5: check mixed formula (g1/2 + g2)/2
        if abs((g1 / 2 + g2) / 2 - score_avg) < 0.01:
            return g1 / 2, g2, "mixed_g1_10"
        # Check both_0_10 formula
        if abs((g1 + g2) / 4 - score_avg) < 0.01:
            return g1 / 2, g2 / 2, "both_0_10_g2_le5"
        # Default: assume g1 is 0-10, g2 is 0-5
        return g1 / 2, g2, "default_mixed_g1_10"

    if g2 > 5 and g1 <= 5:
        # g2 > 5, g1 <= 5: symmetric
        if abs((g1 + g2 / 2) / 2 - score_avg) < 0.01:
            return g1, g2 / 2, "mixed_g2_10"
        if abs((g1 + g2) / 4 - score_avg) < 0.01:
            return g1 / 2, g2 / 2, "both_0_10_g1_le5"
        return g1, g2 / 2, "default_mixed_g2_10"

    # Should not reach here
    return g1, g2, "unknown"


def compute_quality_label(score_avg: float, thresholds: dict) -> int:
    """Map score_avg (0-5) to quality_label using thresholds."""
    if score_avg < thresholds["poor_max"]:
        return 0
    elif score_avg < thresholds["needs_improvement_max"]:
        return 1
    elif score_avg < thresholds["good_max"]:
        return 2
    else:
        return 3


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    params = load_params(project_root / "params.yaml")

    raw_path = project_root / "data/external/mohler/open_ended.parquet"
    out_parquet = project_root / params["data"]["raw_data_path"]
    out_csv = project_root / params["data"]["raw_data_csv_path"]
    thresholds = params["data"]["label_thresholds"]

    logger.info("Reading raw data from %s", raw_path)
    df = pd.read_parquet(raw_path)
    logger.info("Raw rows: %d", len(df))
    logger.info("Columns: %s", df.columns.tolist())

    # Verify score ranges
    logger.info(
        "Score ranges - g1: [%.2f, %.2f], g2: [%.2f, %.2f], score_avg: [%.2f, %.2f]",
        df["score_grader_1"].min(),
        df["score_grader_1"].max(),
        df["score_grader_2"].min(),
        df["score_grader_2"].max(),
        df["score_avg"].min(),
        df["score_avg"].max(),
    )

    # Keep original score_avg and id for assertion later
    original_score_avg = df["score_avg"].copy()
    original_id = df["id"].copy()

    # Strip stray <br> tags from student_answer
    df["student_answer"] = df["student_answer"].str.replace(r"<br\s*/?>", "", regex=True)

    # Normalize grader scores (for agreement statistics only)
    logger.info("Normalizing grader scores...")
    norm_results = df.apply(
        lambda row: normalize_grader_scores(
            row["score_grader_1"], row["score_grader_2"], row["score_avg"]
        ),
        axis=1,
        result_type="expand",
    )
    df["score_grader1"] = norm_results[0]
    df["score_grader2"] = norm_results[1]
    df["norm_category"] = norm_results[2]

    # Log normalization categories
    cat_counts = df["norm_category"].value_counts()
    logger.info("Normalization categories:")
    for cat, count in cat_counts.items():
        logger.info("  %s: %d", cat, count)

    # Count how many were actually changed (normalized from 0-10 to 0-5)
    changed = ((df["score_grader1"] != df["score_grader_1"]) | (df["score_grader2"] != df["score_grader_2"])).sum()
    logger.info("Rows with grader scores normalized (0-10 -> 0-5): %d", changed)

    # Compute quality_label from ORIGINAL score_avg (not recomputed)
    df["quality_label"] = df["score_avg"].apply(lambda x: compute_quality_label(x, thresholds))

    # Compute what the labels would have been with recomputed score_avg (for comparison)
    recomputed_score_avg = (df["score_grader1"] + df["score_grader2"]) / 2
    labels_recomputed = recomputed_score_avg.apply(lambda x: compute_quality_label(x, thresholds))
    labels_original = df["quality_label"]
    label_changes = (labels_recomputed != labels_original).sum()
    logger.info("Rows with different quality_label (recomputed vs original score_avg): %d", label_changes)

    # Extract question_id
    df["question_id"] = df["id"].apply(extract_question_id)

    # Select and rename columns for output
    out_df = pd.DataFrame({
        "question_id": df["question_id"],
        "question": df["question"],
        "reference_answer": df["instructor_answer"],
        "candidate_answer": df["student_answer"],
        "score_grader1": df["score_grader1"],
        "score_grader2": df["score_grader2"],
        "score_avg": df["score_avg"],  # Use original score_avg unchanged
        "quality_label": df["quality_label"],
    })

    # Assertion: output score_avg equals source score_avg for every row (matched by original id)
    # We need to match by id since out_df doesn't have id column anymore
    # Create a mapping from id to original score_avg
    id_to_original_avg = dict(zip(original_id, original_score_avg))
    out_df["_source_id"] = original_id.values  # temporary for assertion
    out_df["_expected_avg"] = out_df["_source_id"].map(id_to_original_avg)
    mismatch = (out_df["score_avg"] - out_df["_expected_avg"]).abs() > 1e-10
    if mismatch.any():
        raise AssertionError(f"score_avg mismatch for {mismatch.sum()} rows")
    logger.info("Assertion passed: output score_avg equals source score_avg for all rows")
    out_df = out_df.drop(columns=["_source_id", "_expected_avg"])

    # Drop rows with empty candidate_answer or missing scores
    initial_rows = len(out_df)
    out_df = out_df.dropna(subset=["candidate_answer", "score_grader1", "score_grader2", "score_avg"])
    out_df = out_df[out_df["candidate_answer"].str.strip() != ""]
    dropped = initial_rows - len(out_df)
    if dropped > 0:
        logger.info("Dropped %d rows with empty candidate_answer or missing scores", dropped)

    # Ensure output directory exists
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)
    logger.info("Written %d rows to %s and %s", len(out_df), out_parquet, out_csv)

    # Print summary statistics
    logger.info("=== Dataset Summary ===")
    logger.info("Row count: %d", len(out_df))
    logger.info("Unique questions: %d", out_df["question_id"].nunique())
    logger.info("Class distribution:")
    class_dist = out_df["quality_label"].value_counts().sort_index()
    for cls, count in class_dist.items():
        label = params["data"]["target_classes"][cls]
        logger.info("  Class %d (%s): %d (%.1f%%)", cls, label, count, 100 * count / len(out_df))
    logger.info("score_avg mean: %.3f", out_df["score_avg"].mean())
    logger.info("score_avg median: %.3f", out_df["score_avg"].median())


if __name__ == "__main__":
    main()
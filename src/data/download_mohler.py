#!/usr/bin/env python3
"""
Download Mohler ASAG dataset from Hugging Face (nkazi/MohlerASAG, subset "cleaned", split "open_ended")
and save as Parquet. Columns match the existing file: id, question, instructor_answer, student_answer,
score_grader_1, score_grader_2, score_avg.
"""

import argparse
from pathlib import Path

from datasets import load_dataset
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mohler ASAG open_ended split from HF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/mohler/open_ended.parquet"),
        help="Output Parquet path (default: data/external/mohler/open_ended.parquet)",
    )
    args = parser.parse_args()

    print(f"Loading nkazi/MohlerASAG (subset: cleaned) from Hugging Face...")
    ds = load_dataset("nkazi/MohlerASAG", "cleaned", split="open_ended")
    df = pd.DataFrame(ds)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Ensure column order matches existing file
    expected_cols = [
        "id",
        "question",
        "instructor_answer",
        "student_answer",
        "score_grader_1",
        "score_grader_2",
        "score_avg",
    ]
    df = df[expected_cols]

    # Ensure dtypes match
    df["id"] = df["id"].astype(str)
    df["question"] = df["question"].astype(str)
    df["instructor_answer"] = df["instructor_answer"].astype(str)
    df["student_answer"] = df["student_answer"].astype(str)
    df["score_grader_1"] = df["score_grader_1"].astype("float32")
    df["score_grader_2"] = df["score_grader_2"].astype("float32")
    df["score_avg"] = df["score_avg"].astype("float32")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
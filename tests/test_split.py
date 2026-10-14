#!/usr/bin/env python3
"""
Tests for grouped split logic in src/data/preprocess.py
"""

import pandas as pd
import pytest

from src.data.preprocess import grouped_split, clean_text, assert_disjoint_splits


def make_synthetic_data(n_groups: int = 10, rows_per_group: int = 5, n_classes: int = 4, seed: int = 42) -> pd.DataFrame:
    """Create synthetic DataFrame for testing splits."""
    import numpy as np
    np.random.seed(seed)

    data = []
    for g in range(n_groups):
        group_id = f"Q{g:03d}"
        for r in range(rows_per_group):
            data.append({
                "question_id": group_id,
                "question": f"Question {group_id} text",
                "reference_answer": f"Reference answer for {group_id}",
                "candidate_answer": f"Candidate answer {r} for {group_id}",
                "score_grader1": float(np.random.randint(0, 6)),
                "score_grader2": float(np.random.randint(0, 6)),
                "score_avg": float(np.random.randint(0, 6)),
                "quality_label": np.random.randint(0, n_classes),
            })
    return pd.DataFrame(data)


class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert clean_text("hello    world") == "hello world"

    def test_collapses_newlines_tabs(self):
        assert clean_text("hello\n\tworld") == "hello world"

    def test_keeps_case(self):
        assert clean_text("HeLLo WoRLD") == "HeLLo WoRLD"

    def test_handles_nan(self):
        assert clean_text(float("nan")) == ""

    def test_handles_empty(self):
        assert clean_text("") == ""


class TestGroupedSplit:
    def test_zero_group_overlap(self):
        """Verify no question_id appears in more than one split."""
        df = make_synthetic_data(n_groups=20, rows_per_group=5)
        train, val, test = grouped_split(
            df=df,
            group_column="question_id",
            test_size=0.20,
            validation_size=0.10,
            random_state=42,
        )

        train_groups = set(train["question_id"].unique())
        val_groups = set(val["question_id"].unique())
        test_groups = set(test["question_id"].unique())

        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)

    def test_all_rows_preserved(self):
        """Sum of split lengths equals input length."""
        df = make_synthetic_data(n_groups=20, rows_per_group=5)
        train, val, test = grouped_split(
            df=df,
            group_column="question_id",
            test_size=0.20,
            validation_size=0.10,
            random_state=42,
        )

        assert len(train) + len(val) + len(test) == len(df)

    def test_reproducibility(self):
        """Same seed produces identical splits."""
        df = make_synthetic_data(n_groups=20, rows_per_group=5)

        train1, val1, test1 = grouped_split(
            df=df, group_column="question_id",
            test_size=0.20, validation_size=0.10, random_state=42,
        )
        train2, val2, test2 = grouped_split(
            df=df, group_column="question_id",
            test_size=0.20, validation_size=0.10, random_state=42,
        )

        # Compare question_id sets
        assert set(train1["question_id"]) == set(train2["question_id"])
        assert set(val1["question_id"]) == set(val2["question_id"])
        assert set(test1["question_id"]) == set(test2["question_id"])

    def test_approximate_split_ratios(self):
        """Split ratios are approximately correct (within tolerance due to grouping)."""
        df = make_synthetic_data(n_groups=100, rows_per_group=10)
        train, val, test = grouped_split(
            df=df,
            group_column="question_id",
            test_size=0.20,
            validation_size=0.10,
            random_state=42,
        )

        total = len(df)
        test_ratio = len(test) / total
        val_ratio = len(val) / total
        train_ratio = len(train) / total

        # With grouping, ratios are approximate (groups have varying sizes)
        assert 0.15 <= test_ratio <= 0.25
        assert 0.07 <= val_ratio <= 0.13
        assert 0.65 <= train_ratio <= 0.75

    def test_different_seeds_different_splits(self):
        """Different seeds produce different splits (with high probability)."""
        df = make_synthetic_data(n_groups=20, rows_per_group=5)

        train1, val1, test1 = grouped_split(
            df=df, group_column="question_id",
            test_size=0.20, validation_size=0.10, random_state=42,
        )
        train2, val2, test2 = grouped_split(
            df=df, group_column="question_id",
            test_size=0.20, validation_size=0.10, random_state=123,
        )

        # At least one split should differ
        same = (set(train1["question_id"]) == set(train2["question_id"]) and
                set(val1["question_id"]) == set(val2["question_id"]) and
                set(test1["question_id"]) == set(test2["question_id"]))
        assert not same, "Different seeds produced identical splits (unlikely)"


class TestAssertDisjointSplits:
    def test_passes_for_disjoint(self):
        df = make_synthetic_data(n_groups=10, rows_per_group=5)
        train, val, test = grouped_split(
            df=df, group_column="question_id",
            test_size=0.20, validation_size=0.10, random_state=42,
        )
        # Should not raise
        assert_disjoint_splits(train, val, test, "question_id")

    def test_raises_for_overlap(self):
        df = make_synthetic_data(n_groups=10, rows_per_group=5)
        train, val, test = grouped_split(
            df=df, group_column="question_id",
            test_size=0.20, validation_size=0.10, random_state=42,
        )
        # Create overlap by adding a row from train to val
        overlap_row = train.iloc[[0]].copy()
        val_with_overlap = pd.concat([val, overlap_row], ignore_index=True)

        with pytest.raises(AssertionError, match="Train/Val group overlap"):
            assert_disjoint_splits(train, val_with_overlap, test, "question_id")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
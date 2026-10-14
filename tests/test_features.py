#!/usr/bin/env python3
"""
Tests for feature pipeline in src/features/build_features.py
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.features.build_features import (
    build_feature_pipeline,
    transform_data,
    TextLengthFeatures,
    ReferenceSimilarityFeatures,
)


def make_test_params():
    """Create test params matching the structure in params.yaml."""
    return {
        "features": {
            "tfidf": {
                "max_features": 100,
                "ngram_range": [1, 2],
                "stop_words": "english",
                "min_df": 1,
                "sublinear_tf": True,
            },
            "engineered_features": {
                "include_length": True,
                "include_reference_similarity": True,
            },
        }
    }


def make_sample_data(n: int = 50) -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    data = []
    for i in range(n):
        data.append({
            "question_id": f"Q{i // 5:03d}",
            "question": f"Question {i} text",
            "reference_answer": f"Reference answer for question {i} with some words",
            "candidate_answer": f"Candidate answer {i} with different words and length",
            "score_grader1": float(np.random.randint(0, 6)),
            "score_grader2": float(np.random.randint(0, 6)),
            "score_avg": float(np.random.randint(0, 6)),
            "quality_label": np.random.randint(0, 4),
        })
    return pd.DataFrame(data)


class TestTextLengthFeatures:
    def test_basic_extraction(self):
        transformer = TextLengthFeatures(include_length=True)
        texts = pd.Series(["hello world", "this is a test", ""])
        X = pd.DataFrame({"candidate_answer": texts})
        result = transformer.fit_transform(X)
        assert result.shape == (3, 5)
        # Check feature names
        names = transformer.get_feature_names_out()
        assert list(names) == [
            "char_count", "token_count", "sentence_count",
            "unique_token_ratio", "mean_word_length"
        ]

    def test_disabled(self):
        transformer = TextLengthFeatures(include_length=False)
        texts = pd.Series(["hello world", "foo bar", "baz qux"])
        X = pd.DataFrame({"candidate_answer": texts})
        result = transformer.fit_transform(X)
        assert result.shape == (3, 0)

    def test_values_reasonable(self):
        transformer = TextLengthFeatures(include_length=True)
        texts = pd.Series(["hello world", "a b c d e"])
        X = pd.DataFrame({"candidate_answer": texts})
        result = transformer.fit_transform(X)
        # char_count
        assert result[0, 0] == 11  # "hello world"
        assert result[1, 0] == 9   # "a b c d e"
        # token_count
        assert result[0, 1] == 2
        assert result[1, 1] == 5
        # unique_token_ratio
        assert result[0, 3] == 1.0  # both unique
        assert result[1, 3] == 1.0  # all unique


class TestReferenceSimilarityFeatures:
    def test_basic_extraction(self):
        transformer = ReferenceSimilarityFeatures(include_reference_similarity=True)
        df = pd.DataFrame({
            "candidate_answer": ["hello world", "foo bar"],
            "reference_answer": ["hello world", "baz qux"],
        })
        result = transformer.fit_transform(df)
        assert result.shape == (2, 1)
        names = transformer.get_feature_names_out()
        assert list(names) == ["candidate_reference_cosine_sim"]
        # First pair identical -> similarity 1.0
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        # Second pair different -> similarity < 1.0
        assert 0.0 <= result[1, 0] < 1.0

    def test_disabled(self):
        transformer = ReferenceSimilarityFeatures(include_reference_similarity=False)
        df = pd.DataFrame({
            "candidate_answer": ["hello"],
            "reference_answer": ["world"],
        })
        result = transformer.fit_transform(df)
        assert result.shape == (1, 0)

    def test_range_clamped(self):
        transformer = ReferenceSimilarityFeatures(include_reference_similarity=True)
        # Use longer texts so TF-IDF with min_df=1 works
        df = pd.DataFrame({
            "candidate_answer": ["hello world", "hello world"],
            "reference_answer": ["hello world", "different words"],
        })
        result = transformer.fit_transform(df)
        assert 0.0 <= result[0, 0] <= 1.0
        assert 0.0 <= result[1, 0] <= 1.0


class TestFeaturePipeline:
    def test_pipeline_fitted_on_train_only(self):
        """Pipeline should be fitted on train only, not on val/test."""
        params = make_test_params()
        train_df = make_sample_data(30)
        val_df = make_sample_data(10)
        test_df = make_sample_data(10)

        pipeline = build_feature_pipeline(params)
        X_train = pipeline.fit_transform(train_df)
        X_val = pipeline.transform(val_df)
        X_test = pipeline.transform(test_df)

        # All should have same number of features
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

    def test_output_shapes_match_across_splits(self):
        """Output feature dimension should be identical for train/val/test."""
        params = make_test_params()
        train_df = make_sample_data(30)
        val_df = make_sample_data(10)
        test_df = make_sample_data(10)

        pipeline = build_feature_pipeline(params)
        pipeline.fit(train_df)

        X_train = pipeline.transform(train_df)
        X_val = pipeline.transform(val_df)
        X_test = pipeline.transform(test_df)

        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        assert X_train.shape[0] == 30
        assert X_val.shape[0] == 10
        assert X_test.shape[0] == 10

    def test_cosine_similarity_in_range(self):
        """Cosine similarity feature should be in [0, 1]."""
        params = make_test_params()
        train_df = make_sample_data(30)

        pipeline = build_feature_pipeline(params)
        X_train = pipeline.fit_transform(train_df)

        # Find the cosine similarity column index (last if reference similarity enabled)
        feature_names = pipeline.get_feature_names_out()
        ref_sim_idx = None
        for i, name in enumerate(feature_names):
            if "cosine_sim" in name:
                ref_sim_idx = i
                break

        assert ref_sim_idx is not None, "Cosine similarity feature not found"

        # Extract cosine similarity values
        if hasattr(X_train, "toarray"):
            sim_values = X_train[:, ref_sim_idx].toarray().flatten()
        else:
            sim_values = X_train[:, ref_sim_idx]

        assert np.all(sim_values >= 0.0), f"Found negative cosine similarity: {sim_values.min()}"
        assert np.all(sim_values <= 1.0), f"Found cosine similarity > 1: {sim_values.max()}"

    def test_transform_helper(self):
        """Test the transform_data helper function."""
        params = make_test_params()
        train_df = make_sample_data(20)
        test_df = make_sample_data(10)

        pipeline = build_feature_pipeline(params)
        pipeline.fit(train_df)

        X_test = transform_data(pipeline, test_df)
        assert X_test.shape[0] == 10
        assert X_test.shape[1] == pipeline.transform(train_df).shape[1]

    def test_sparse_output(self):
        """Output should be sparse matrix (due to TF-IDF) or dense array."""
        params = make_test_params()
        train_df = make_sample_data(20)

        pipeline = build_feature_pipeline(params)
        X_train = pipeline.fit_transform(train_df)

        # Should be sparse matrix or dense array (both have toarray method)
        # ColumnTransformer with sparse_threshold may return dense if dense features dominate
        assert hasattr(X_train, "toarray") or hasattr(X_train, "shape")
        assert X_train.shape[0] == 20
        assert X_train.shape[1] > 0

    def test_feature_names_present(self):
        """Pipeline should produce expected feature names."""
        params = make_test_params()
        train_df = make_sample_data(20)

        pipeline = build_feature_pipeline(params)
        pipeline.fit(train_df)

        feature_names = pipeline.get_feature_names_out()
        # Should have TF-IDF features + length features + cosine similarity
        assert len(feature_names) > 0
        # Check for TF-IDF features (they have format like "tfidf__word")
        tfidf_features = [n for n in feature_names if n.startswith("tfidf__")]
        assert len(tfidf_features) > 0
        # Check for length features
        length_features = [n for n in feature_names if n.startswith("length_features__")]
        assert len(length_features) == 5
        # Check for cosine similarity
        ref_sim_features = [n for n in feature_names if "cosine_sim" in n]
        assert len(ref_sim_features) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
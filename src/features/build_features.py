#!/usr/bin/env python3
"""
Build feature pipeline for interview answer quality classification.

Creates a scikit-learn ColumnTransformer combining:
- TF-IDF on candidate_answer (fitted on train only)
- Engineered numeric features: length stats, cosine similarity to reference_answer

Saves fitted pipeline to models/feature_pipeline.joblib.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def load_params(params_path: Path) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


class TextLengthFeatures(BaseEstimator, TransformerMixin):
    """Extract text length features from candidate_answer."""

    def __init__(self, include_length: bool = True):
        self.include_length = include_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.include_length:
            return np.empty((len(X), 0))

        # Handle both Series and DataFrame input
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = X.fillna("").astype(str).tolist()

        features = []
        for text in texts:
            # Character count
            char_count = len(text)

            # Token count (split on whitespace)
            tokens = text.split()
            token_count = len(tokens)

            # Sentence count (split on .!?)
            import re
            sentences = re.split(r'[.!?]+', text)
            sentence_count = len([s for s in sentences if s.strip()])

            # Unique token ratio
            unique_tokens = len(set(tokens)) if tokens else 0
            unique_token_ratio = unique_tokens / token_count if token_count > 0 else 0.0

            # Mean word length
            mean_word_length = np.mean([len(t) for t in tokens]) if tokens else 0.0

            features.append([
                char_count,
                token_count,
                sentence_count,
                unique_token_ratio,
                mean_word_length,
            ])

        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        if not self.include_length:
            return np.array([], dtype=object)
        return np.array([
            "char_count",
            "token_count",
            "sentence_count",
            "unique_token_ratio",
            "mean_word_length",
        ], dtype=object)


class ReferenceSimilarityFeatures(BaseEstimator, TransformerMixin):
    """Extract cosine similarity between candidate_answer and reference_answer using TF-IDF."""

    def __init__(self, include_reference_similarity: bool = True):
        self.include_reference_similarity = include_reference_similarity
        self.tfidf_vectorizer_ = None

    def fit(self, X, y=None):
        if not self.include_reference_similarity:
            return self

        # X is a DataFrame with 'candidate_answer' and 'reference_answer' columns
        candidate_texts = X['candidate_answer'].fillna("").astype(str).tolist()
        reference_texts = X['reference_answer'].fillna("").astype(str).tolist()

        # Fit TF-IDF on both candidate and reference texts combined
        all_texts = candidate_texts + reference_texts
        self.tfidf_vectorizer_ = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )
        self.tfidf_vectorizer_.fit(all_texts)
        return self

    def transform(self, X):
        if not self.include_reference_similarity:
            return np.empty((len(X), 0))

        # Handle both Series and DataFrame input
        if isinstance(X, pd.DataFrame):
            candidate_texts = X['candidate_answer'].fillna("").astype(str).tolist()
            reference_texts = X['reference_answer'].fillna("").astype(str).tolist()
        else:
            # Fallback - should not happen with ColumnTransformer
            candidate_texts = X.fillna("").astype(str).tolist()
            reference_texts = []

        # Transform both
        candidate_tfidf = self.tfidf_vectorizer_.transform(candidate_texts)
        reference_tfidf = self.tfidf_vectorizer_.transform(reference_texts)

        # Compute cosine similarity row-wise
        similarities = []
        for i in range(len(candidate_texts)):
            cand_vec = candidate_tfidf[i]
            ref_vec = reference_tfidf[i]

            # Cosine similarity = (A·B) / (||A|| * ||B||)
            dot_product = cand_vec.dot(ref_vec.T).toarray()[0, 0]
            cand_norm = np.sqrt(cand_vec.dot(cand_vec.T).toarray()[0, 0])
            ref_norm = np.sqrt(ref_vec.dot(ref_vec.T).toarray()[0, 0])

            if cand_norm > 0 and ref_norm > 0:
                sim = dot_product / (cand_norm * ref_norm)
            else:
                sim = 0.0

            # Clamp to [0, 1]
            sim = max(0.0, min(1.0, sim))
            similarities.append(sim)

        return np.array(similarities, dtype=np.float32).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if not self.include_reference_similarity:
            return np.array([], dtype=object)
        return np.array(["candidate_reference_cosine_sim"], dtype=object)


def build_feature_pipeline(params: dict) -> ColumnTransformer:
    """Build the feature extraction pipeline."""
    tfidf_params = params["features"]["tfidf"]
    eng_params = params["features"]["engineered_features"]

    # TF-IDF on candidate_answer only
    tfidf = TfidfVectorizer(
        max_features=tfidf_params["max_features"],
        ngram_range=tuple(tfidf_params["ngram_range"]),
        stop_words=tfidf_params["stop_words"],
        min_df=tfidf_params["min_df"],
        sublinear_tf=tfidf_params["sublinear_tf"],
    )

    # Engineered features
    length_features = TextLengthFeatures(include_length=eng_params["include_length"])
    ref_sim_features = ReferenceSimilarityFeatures(
        include_reference_similarity=eng_params["include_reference_similarity"]
    )

    # ColumnTransformer: specify which columns go to which transformer
    # We pass the full DataFrame to each transformer
    transformers = [
        ("tfidf", tfidf, "candidate_answer"),
        ("length_features", length_features, ["candidate_answer"]),
        ("ref_sim_features", ref_sim_features, ["candidate_answer", "reference_answer"]),
    ]

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,  # Keep sparse for TF-IDF, dense for others
        n_jobs=None,
    )

    return pipeline


def transform_data(pipeline: ColumnTransformer, df: pd.DataFrame) -> csr_matrix:
    """Transform data using fitted pipeline."""
    return pipeline.transform(df)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    params = load_params(project_root / "params.yaml")

    processed_dir = project_root / params["data"]["processed_data_dir"]
    models_dir = project_root / "models"

    logger.info("Loading train/val/test data...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info("Train: %d rows, Val: %d rows, Test: %d rows",
                len(train_df), len(val_df), len(test_df))

    # Build pipeline
    logger.info("Building feature pipeline...")
    pipeline = build_feature_pipeline(params)

    # Fit on train only
    logger.info("Fitting pipeline on train data...")
    X_train = pipeline.fit_transform(train_df)

    # Transform val and test
    logger.info("Transforming val and test data...")
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)

    logger.info("Feature shapes - Train: %s, Val: %s, Test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    # Verify shapes match (same number of features)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        "Feature dimension mismatch across splits"

    # Save pipeline
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved feature pipeline to %s", pipeline_path)

    # Log feature names
    feature_names = pipeline.get_feature_names_out()
    logger.info("Total features: %d", len(feature_names))
    logger.info("First 20 feature names: %s", feature_names[:20])

    # Quick sanity check on cosine similarity range
    # The reference similarity is the last feature if included
    if params["features"]["engineered_features"]["include_reference_similarity"]:
        ref_sim_idx = -1
        train_ref_sim = X_train[:, ref_sim_idx].toarray().flatten() if hasattr(X_train, "toarray") else X_train[:, ref_sim_idx]
        logger.info("Train cosine similarity range: [%.4f, %.4f]",
                    train_ref_sim.min(), train_ref_sim.max())
        assert 0.0 <= train_ref_sim.min() <= train_ref_sim.max() <= 1.0, \
            "Cosine similarity out of [0, 1] range"

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    main()
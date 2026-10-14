#!/usr/bin/env python3
"""
Train baseline LogisticRegression model with MLflow tracking.

Loads feature pipeline, trains on train split, evaluates on validation,
performs grouped cross-validation, and logs everything to MLflow.
"""

import logging
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.features.build_features import (
    build_feature_pipeline,
    transform_data,
    TextLengthFeatures,
    ReferenceSimilarityFeatures,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def load_params(params_path: Path) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_dataset_version(dvc_lock_path: Path) -> str:
    """Extract md5 hash of raw data from dvc.lock."""
    with open(dvc_lock_path, "r") as f:
        lock_data = yaml.safe_load(f)
    # Find prepare_dataset stage output for interview_data.parquet
    for out in lock_data["stages"]["prepare_dataset"]["outs"]:
        if out["path"] == "data/raw/interview_data.parquet":
            return out["md5"]
    return "unknown"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    # Load configs
    params = load_params(project_root / "params.yaml")
    config = load_config(project_root / "configs/config.yaml")

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Model parameters
    model_params = params["model"]["baseline"]
    random_state = model_params["random_state"]
    c_param = model_params["c_param"]
    solver = model_params["solver"]
    max_iter = model_params["max_iter"]

    # Data paths
    processed_dir = project_root / params["data"]["processed_data_dir"]
    models_dir = project_root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading train and validation data...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    X_train = train_df.drop(columns=[params["data"]["target_column"]])
    y_train = train_df[params["data"]["target_column"]].values
    X_val = val_df.drop(columns=[params["data"]["target_column"]])
    y_val = val_df[params["data"]["target_column"]].values

    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    # Load feature pipeline
    logger.info("Loading feature pipeline...")
    feature_pipeline = joblib.load(models_dir / "feature_pipeline.joblib")

    # Transform data
    logger.info("Transforming data with feature pipeline...")
    X_train_feat = transform_data(feature_pipeline, X_train)
    X_val_feat = transform_data(feature_pipeline, X_val)

    # Train LogisticRegression
    logger.info("Training LogisticRegression...")
    clf = LogisticRegression(
        C=c_param,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=None,
    )
    clf.fit(X_train_feat, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train_feat)
    y_val_pred = clf.predict(X_val_feat)

    # Metrics on train and validation
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)

    # Most frequent class baseline on validation
    most_frequent_class = np.bincount(y_train).argmax()
    y_val_baseline = np.full_like(y_val, most_frequent_class)
    baseline_metrics = compute_metrics(y_val, y_val_baseline)
    logger.info("Most frequent class in train: %d", most_frequent_class)

    # Grouped Cross-Validation on train only
    logger.info("Performing grouped cross-validation...")
    groups = X_train[params["data"]["group_column"]].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_f1_macro = []
    cv_accuracy = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
        logger.info("Fold %d: train=%d, val=%d", fold_idx + 1, len(train_idx), len(val_idx))

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]

        # Build and fit a NEW pipeline + classifier for this fold (no data leakage)
        fold_pipeline = build_feature_pipeline(params)
        fold_clf = LogisticRegression(
            C=c_param,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=None,
        )

        # Fit pipeline on fold training data
        X_fold_train_feat = fold_pipeline.fit_transform(X_fold_train)
        fold_clf.fit(X_fold_train_feat, y_fold_train)

        # Transform fold validation data
        X_fold_val_feat = transform_data(fold_pipeline, X_fold_val)
        y_fold_pred = fold_clf.predict(X_fold_val_feat)

        fold_metrics = compute_metrics(y_fold_val, y_fold_pred)
        cv_f1_macro.append(fold_metrics["f1_macro"])
        cv_accuracy.append(fold_metrics["accuracy"])

        logger.info("  Fold %d: f1_macro=%.4f, accuracy=%.4f",
                    fold_idx + 1, fold_metrics["f1_macro"], fold_metrics["accuracy"])

    cv_f1_macro_mean = np.mean(cv_f1_macro)
    cv_f1_macro_std = np.std(cv_f1_macro)
    cv_accuracy_mean = np.mean(cv_accuracy)
    cv_accuracy_std = np.std(cv_accuracy)

    logger.info("CV f1_macro: %.4f (+/- %.4f)", cv_f1_macro_mean, cv_f1_macro_std)
    logger.info("CV accuracy: %.4f (+/- %.4f)", cv_accuracy_mean, cv_accuracy_std)

    # MLflow logging
    run_name = "logistic_regression_baseline"
    git_commit = get_git_commit_hash()
    dataset_version = get_dataset_version(project_root / "dvc.lock")

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "model_algorithm": model_params["algorithm"],
            "C": c_param,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": random_state,
            "class_weight": "balanced",
            "n_splits_cv": 5,
            "cv_strategy": "StratifiedGroupKFold",
            "group_column": params["data"]["group_column"],
        })

        # Log git commit and dataset version
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dataset_version", dataset_version)

        # Log train metrics
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        # Log validation metrics
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        # Log baseline metrics
        for k, v in baseline_metrics.items():
            mlflow.log_metric(f"val_baseline_{k}", v)

        # Log CV metrics
        mlflow.log_metric("cv_f1_macro_mean", cv_f1_macro_mean)
        mlflow.log_metric("cv_f1_macro_std", cv_f1_macro_std)
        mlflow.log_metric("cv_accuracy_mean", cv_accuracy_mean)
        mlflow.log_metric("cv_accuracy_std", cv_accuracy_std)

        # Log individual fold metrics
        for i, (f1, acc) in enumerate(zip(cv_f1_macro, cv_accuracy)):
            mlflow.log_metric(f"cv_fold_{i}_f1_macro", f1)
            mlflow.log_metric(f"cv_fold_{i}_accuracy", acc)

        # Log model and feature pipeline
        mlflow.sklearn.log_model(clf, "model")
        mlflow.sklearn.log_model(
            feature_pipeline,
            "feature_pipeline",
            skops_trusted_types=[
                "src.features.build_features.TextLengthFeatures",
                "src.features.build_features.ReferenceSimilarityFeatures",
            ],
        )

        # Confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        cm_path = project_root / config["paths"]["figures_dir"] / "confusion_matrix.png"
        cm_path.parent.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        class_names = [params["data"]["target_classes"][i] for i in range(4)]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()

        mlflow.log_artifact(str(cm_path), "plots")

        # Save model locally
        model_path = models_dir / "baseline_model.joblib"
        joblib.dump(clf, model_path)
        logger.info("Saved model to %s", model_path)

        # Print all metrics
        logger.info("=== TRAIN METRICS ===")
        for k, v in train_metrics.items():
            logger.info("  %s: %.4f", k, v)

        logger.info("=== VALIDATION METRICS ===")
        for k, v in val_metrics.items():
            logger.info("  %s: %.4f", k, v)

        logger.info("=== BASELINE (always predict class %d) ===", most_frequent_class)
        for k, v in baseline_metrics.items():
            logger.info("  %s: %.4f", k, v)

        logger.info("=== GROUPED CV (5-fold StratifiedGroupKFold) ===")
        logger.info("  f1_macro: %.4f (+/- %.4f)", cv_f1_macro_mean, cv_f1_macro_std)
        logger.info("  accuracy: %.4f (+/- %.4f)", cv_accuracy_mean, cv_accuracy_std)

        logger.info("MLflow run ID: %s", run.info.run_id)
        logger.info("Training complete")


if __name__ == "__main__":
    main()
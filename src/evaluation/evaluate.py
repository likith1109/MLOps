#!/usr/bin/env python3
"""
Evaluate both models on test data, apply selection rule, and log results.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)

from src.features.build_features import (
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


def get_latest_mlflow_metrics(run_name: str, tracking_uri: str, experiment_name: str) -> dict:
    """Fetch the latest run metrics for a given run name."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found for run_name='{run_name}'")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_classes: dict) -> dict:
    """Compute per-class precision, recall, F1 with support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3]
    )
    # target_classes uses integer keys in params.yaml
    return {
        target_classes[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(4)
    }


def get_majority_baseline_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute baseline metrics (always predict most frequent class from train)."""
    most_frequent = int(np.bincount(y_train).argmax())
    y_baseline = np.full_like(y_test, most_frequent)
    return {
        "most_frequent_class": most_frequent,
        "accuracy": float(accuracy_score(y_test, y_baseline)),
        "f1_macro": float(f1_score(y_test, y_baseline, average="macro", zero_division=0)),
    }


def apply_selection_rule(baseline_cv: dict, rf_cv: dict, baseline_val_f1_macro: float, rf_val_f1_macro: float) -> tuple:
    """
    Apply selection rule:
    - Choose higher grouped-CV f1_macro mean
    - If gap < max(CV f1_macro stds), tie-break by higher validation f1_macro
    Returns: (primary_model_name, reason)
    """
    baseline_mean = baseline_cv["cv_f1_macro_mean"]
    rf_mean = rf_cv["cv_f1_macro_mean"]
    baseline_std = baseline_cv["cv_f1_macro_std"]
    rf_std = rf_cv["cv_f1_macro_std"]

    gap = abs(baseline_mean - rf_mean)
    max_std = max(baseline_std, rf_std)

    logger.info(f"Baseline CV f1_macro: {baseline_mean:.4f} (±{baseline_std:.4f})")
    logger.info(f"RandomForest CV f1_macro: {rf_mean:.4f} (±{rf_std:.4f})")
    logger.info(f"Gap: {gap:.4f}, Max std: {max_std:.4f}")

    if gap < max_std:
        # Tie - use validation f1_macro
        if rf_val_f1_macro > baseline_val_f1_macro:
            return "random_forest", f"Tie on CV f1_macro (gap={gap:.4f} < max_std={max_std:.4f}); RandomForest has higher val f1_macro ({rf_val_f1_macro:.4f} vs {baseline_val_f1_macro:.4f})"
        else:
            return "logistic_regression", f"Tie on CV f1_macro (gap={gap:.4f} < max_std={max_std:.4f}); LogisticRegression has higher val f1_macro ({baseline_val_f1_macro:.4f} vs {rf_val_f1_macro:.4f})"
    else:
        # Clear winner on CV
        if rf_mean > baseline_mean:
            return "random_forest", f"RandomForest has higher CV f1_macro ({rf_mean:.4f} vs {baseline_mean:.4f})"
        else:
            return "logistic_regression", f"LogisticRegression has higher CV f1_macro ({baseline_mean:.4f} vs {rf_mean:.4f})"


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, title: str, save_path: Path):
    """Plot and save confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                            target_column: str, target_classes: dict, save_path: Path):
    """Plot class distribution across splits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = {
        "train": train_df[target_column].value_counts().sort_index(),
        "val": val_df[target_column].value_counts().sort_index(),
        "test": test_df[target_column].value_counts().sort_index(),
    }

    x = np.arange(4)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (split_name, counts) in enumerate(splits.items()):
        values = [counts.get(j, 0) for j in range(4)]
        ax.bar(x + i * width, values, width, label=split_name.capitalize())

    ax.set_xlabel("Quality Label")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution Across Splits")
    ax.set_xticks(x + width)
    ax.set_xticklabels([target_classes[i] for i in range(4)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_answer_length_by_class(df: pd.DataFrame, target_column: str, target_classes: dict, save_path: Path):
    """Plot answer length (characters) by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    df["answer_length"] = df["candidate_answer"].str.len()

    class_order = [target_classes[i] for i in range(4)]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=target_column, y="answer_length", order=range(4))
    plt.xticks(range(4), class_order)
    plt.xlabel("Quality Label")
    plt.ylabel("Answer Length (characters)")
    plt.title("Candidate Answer Length by Quality Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    params = load_params(project_root / "params.yaml")
    config = load_config(project_root / "configs/config.yaml")

    # Paths
    processed_dir = project_root / params["data"]["processed_data_dir"]
    models_dir = project_root / config["paths"]["models_dir"]
    reports_dir = project_root / config["paths"]["reports_dir"]
    figures_dir = project_root / config["paths"]["figures_dir"]
    metrics_file = project_root / config["paths"]["metrics_file"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    X_test = test_df.drop(columns=[params["data"]["target_column"]])
    y_test = test_df[params["data"]["target_column"]].values

    # Also load train for majority baseline
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    y_train = train_df[params["data"]["target_column"]].values
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    # Load feature pipeline
    logger.info("Loading feature pipeline...")
    feature_pipeline = joblib.load(models_dir / "feature_pipeline.joblib")

    # Transform test data
    X_test_feat = transform_data(feature_pipeline, X_test)

    # Load both models
    logger.info("Loading models...")
    baseline_model = joblib.load(models_dir / "baseline_model.joblib")
    rf_model = joblib.load(models_dir / "rf_model.joblib")

    # Get CV and validation metrics from MLflow
    logger.info("Fetching MLflow metrics...")
    baseline_mlflow = get_latest_mlflow_metrics(
        "logistic_regression_baseline",
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["experiment_name"],
    )
    rf_mlflow = get_latest_mlflow_metrics(
        "random_forest",
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["experiment_name"],
    )

    # Extract metrics for selection
    baseline_cv = {
        "cv_f1_macro_mean": baseline_mlflow["metrics"]["cv_f1_macro_mean"],
        "cv_f1_macro_std": baseline_mlflow["metrics"]["cv_f1_macro_std"],
    }
    rf_cv = {
        "cv_f1_macro_mean": rf_mlflow["metrics"]["cv_f1_macro_mean"],
        "cv_f1_macro_std": rf_mlflow["metrics"]["cv_f1_macro_std"],
    }
    baseline_val_f1_macro = baseline_mlflow["metrics"]["val_f1_macro"]
    rf_val_f1_macro = rf_mlflow["metrics"]["val_f1_macro"]

    # Apply selection rule
    primary_model_name, selection_reason = apply_selection_rule(
        baseline_cv, rf_cv, baseline_val_f1_macro, rf_val_f1_macro
    )
    logger.info(f"Primary model: {primary_model_name} - {selection_reason}")

    # Evaluate both models on test
    models = {
        "logistic_regression": baseline_model,
        "random_forest": rf_model,
    }

    test_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_feat)
        test_results[name] = {
            "metrics": compute_metrics(y_test, y_pred),
            "per_class": compute_per_class_metrics(y_test, y_pred, params["data"]["target_classes"]),
            "y_pred": y_pred,
        }

    # Majority baseline on test
    baseline_metrics = get_majority_baseline_metrics(y_train, y_test)

    # Test set info
    test_rows = len(test_df)
    test_questions = test_df["question_id"].nunique()

    # Prepare metrics.json
    metrics_json = {
        "test_evaluation": {
            "logistic_regression": {
                "test_metrics": test_results["logistic_regression"]["metrics"],
                "per_class_metrics": test_results["logistic_regression"]["per_class"],
            },
            "random_forest": {
                "test_metrics": test_results["random_forest"]["metrics"],
                "per_class_metrics": test_results["random_forest"]["per_class"],
            },
            "majority_class_baseline": baseline_metrics,
            "test_rows": test_rows,
            "test_unique_questions": test_questions,
        },
        "selection": {
            "cv_metrics": {
                "logistic_regression": baseline_cv,
                "random_forest": rf_cv,
            },
            "validation_f1_macro": {
                "logistic_regression": baseline_val_f1_macro,
                "random_forest": rf_val_f1_macro,
            },
            "primary_model": primary_model_name,
            "selection_reason": selection_reason,
            "selection_rule": params["evaluation"]["selection_rule"],
        },
        "notes": {
            "poor_class_test_rows": int(test_results["logistic_regression"]["per_class"]["Poor"]["support"]),
        },
    }

    # Save metrics.json
    with open(metrics_file, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Generate plots
    class_names = [params["data"]["target_classes"][i] for i in range(4)]

    # Confusion matrix for logistic_regression
    lr_cm_path = figures_dir / "confusion_matrix_logistic_regression.png"
    plot_confusion_matrix(
        y_test, test_results["logistic_regression"]["y_pred"], class_names,
        "Test Confusion Matrix - LogisticRegression", lr_cm_path
    )

    # Confusion matrix for random_forest
    rf_cm_path = figures_dir / "confusion_matrix_random_forest.png"
    plot_confusion_matrix(
        y_test, test_results["random_forest"]["y_pred"], class_names,
        "Test Confusion Matrix - RandomForest", rf_cm_path
    )

    # Primary model confusion matrix (copy for primary)
    primary_cm_path = figures_dir / "confusion_matrix.png"
    import shutil
    shutil.copy(
        lr_cm_path if primary_model_name == "logistic_regression" else rf_cm_path,
        primary_cm_path
    )

    # Class distribution plot
    dist_path = figures_dir / "class_distribution.png"
    plot_class_distribution(train_df, val_df, test_df, params["data"]["target_column"],
                            params["data"]["target_classes"], dist_path)

    # Answer length by class plot
    length_path = figures_dir / "answer_length_by_class.png"
    plot_answer_length_by_class(train_df, params["data"]["target_column"],
                                 params["data"]["target_classes"], length_path)

    # MLflow logging
    with mlflow.start_run(run_name="final_test_evaluation") as run:
        # Log test metrics for both models
        for name in ["logistic_regression", "random_forest"]:
            for k, v in test_results[name]["metrics"].items():
                mlflow.log_metric(f"test_{name}_{k}", v)

        # Log majority baseline
        for k, v in baseline_metrics.items():
            if k != "most_frequent_class":
                mlflow.log_metric(f"test_baseline_{k}", v)

        # Log CV and validation metrics used for selection
        mlflow.log_metric("cv_f1_macro_mean_baseline", baseline_cv["cv_f1_macro_mean"])
        mlflow.log_metric("cv_f1_macro_std_baseline", baseline_cv["cv_f1_macro_std"])
        mlflow.log_metric("cv_f1_macro_mean_rf", rf_cv["cv_f1_macro_mean"])
        mlflow.log_metric("cv_f1_macro_std_rf", rf_cv["cv_f1_macro_std"])
        mlflow.log_metric("val_f1_macro_baseline", baseline_val_f1_macro)
        mlflow.log_metric("val_f1_macro_rf", rf_val_f1_macro)

        # Log selection info
        mlflow.set_tag("primary_model", primary_model_name)
        mlflow.set_tag("selection_reason", selection_reason)

        # Log figures as artifacts
        mlflow.log_artifact(str(primary_cm_path), "plots")
        mlflow.log_artifact(str(lr_cm_path), "plots")
        mlflow.log_artifact(str(rf_cm_path), "plots")
        mlflow.log_artifact(str(dist_path), "plots")
        mlflow.log_artifact(str(length_path), "plots")
        mlflow.log_artifact(str(metrics_file), "metrics")

        logger.info(f"MLflow run ID: {run.info.run_id}")

    # Print summary
    logger.info("=== TEST METRICS (LogisticRegression) ===")
    for k, v in test_results["logistic_regression"]["metrics"].items():
        logger.info("  %s: %.4f", k, v)

    logger.info("=== TEST METRICS (RandomForest) ===")
    for k, v in test_results["random_forest"]["metrics"].items():
        logger.info("  %s: %.4f", k, v)

    logger.info("=== MAJORITY BASELINE (class %d) ===", baseline_metrics["most_frequent_class"])
    logger.info("  accuracy: %.4f", baseline_metrics["accuracy"])
    logger.info("  f1_macro: %.4f", baseline_metrics["f1_macro"])

    logger.info("=== SELECTION ===")
    logger.info("  Primary model: %s", primary_model_name)
    logger.info("  Reason: %s", selection_reason)


if __name__ == "__main__":
    main()
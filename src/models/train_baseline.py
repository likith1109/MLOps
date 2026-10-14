#!/usr/bin/env python3
"""
Train baseline LogisticRegression model with MLflow tracking.

Loads feature pipeline, trains on train split, evaluates on validation,
performs grouped cross-validation, and logs everything to MLflow.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.utils import (
    load_params,
    load_config,
    load_train_val_data,
    load_feature_pipeline,
    transform_features,
    get_most_frequent_class_baseline,
    run_grouped_cv,
    setup_mlflow,
    log_common_mlflow,
    save_model_locally,
    print_metrics,
    compute_metrics,
)
# Import custom classes so joblib can unpickle the feature pipeline
from src.features.build_features import TextLengthFeatures, ReferenceSimilarityFeatures  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def build_logistic_regression(random_state: int):
    """Build LogisticRegression classifier with params from outer scope."""
    # Params will be captured from outer scope via closure
    return LogisticRegression(
        C=build_logistic_regression.c_param,
        solver=build_logistic_regression.solver,
        max_iter=build_logistic_regression.max_iter,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=None,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    # Load configs
    params = load_params(project_root / "params.yaml")
    config = load_config(project_root / "configs/config.yaml")

    # Model parameters
    model_params = params["model"]["baseline"]
    random_state = model_params["random_state"]
    c_param = model_params["c_param"]
    solver = model_params["solver"]
    max_iter = model_params["max_iter"]

    # Attach params to build function for closure
    build_logistic_regression.c_param = c_param
    build_logistic_regression.solver = solver
    build_logistic_regression.max_iter = max_iter

    # Load data
    X_train, y_train, X_val, y_val = load_train_val_data(params, config)

    # Load feature pipeline
    feature_pipeline = load_feature_pipeline(params, config)

    # Transform data
    X_train_feat, X_val_feat = transform_features(feature_pipeline, X_train, X_val)

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
    most_frequent_class, baseline_metrics = get_most_frequent_class_baseline(y_train, y_val)

    # Grouped Cross-Validation on train only
    cv_results = run_grouped_cv(
        X_train, y_train, params, build_logistic_regression, random_state
    )
    cv_f1_macro_mean, cv_f1_macro_std, cv_accuracy_mean, cv_accuracy_std, cv_f1_macro_list, cv_accuracy_list = cv_results

    # MLflow logging
    with setup_mlflow(config, "logistic_regression_baseline") as run:
        log_common_mlflow(
            run, params, config, model_params,
            train_metrics, val_metrics, baseline_metrics,
            cv_f1_macro_mean, cv_f1_macro_std, cv_accuracy_mean, cv_accuracy_std,
            cv_f1_macro_list, cv_accuracy_list,
            clf, feature_pipeline, y_val, y_val_pred,
        )
        save_model_locally(clf, config, "baseline_model.joblib")
        print_metrics(train_metrics, val_metrics, baseline_metrics,
                      most_frequent_class, cv_f1_macro_mean, cv_f1_macro_std,
                      cv_accuracy_mean, cv_accuracy_std)
        logger.info("Training complete")


if __name__ == "__main__":
    main()
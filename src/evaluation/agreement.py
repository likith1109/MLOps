#!/usr/bin/env python3
"""
Compute inter-annotator agreement between score_grader1 and score_grader2.

Calculates percent agreement, Cohen's kappa (unweighted), and quadratic-weighted kappa
on the normalized 0-5 grader scores (rounded to integers).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray, min_rating: int = 0, max_rating: int = 5) -> float:
    """
    Compute quadratic-weighted Cohen's kappa.

    Based on: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py
    """
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)

    n = len(y1)
    if n == 0:
        return 0.0

    # Confusion matrix
    k = max_rating - min_rating + 1
    conf_mat = np.zeros((k, k), dtype=int)
    for i in range(n):
        conf_mat[y1[i] - min_rating, y2[i] - min_rating] += 1

    # Expected matrix
    hist1 = conf_mat.sum(axis=1)
    hist2 = conf_mat.sum(axis=0)
    expected = np.outer(hist1, hist2) / n

    # Weight matrix (quadratic)
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    observed = np.sum(weights * conf_mat)
    expected_weighted = np.sum(weights * expected)

    if expected_weighted == 0:
        return 1.0 if observed == 0 else 0.0

    kappa = 1.0 - observed / expected_weighted
    return float(kappa)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data/raw/interview_data.parquet"
    out_path = project_root / "reports/annotation_agreement.json"

    logger.info("Reading prepared data from %s", data_path)
    df = pd.read_parquet(data_path)

    # Round grader scores to integers 0-5
    g1 = np.round(df["score_grader1"].values).astype(int)
    g2 = np.round(df["score_grader2"].values).astype(int)

    # Clip to 0-5 range
    g1 = np.clip(g1, 0, 5)
    g2 = np.clip(g2, 0, 5)

    n = len(g1)
    logger.info("Computing agreement on %d samples", n)

    # Percent agreement
    percent_agree = float(np.mean(g1 == g2) * 100)

    # Cohen's kappa (unweighted)
    kappa = float(cohen_kappa_score(g1, g2, labels=list(range(6))))

    # Quadratic-weighted kappa
    qwk = quadratic_weighted_kappa(g1, g2)

    # Also compute Pearson correlation
    pearson_r, pearson_p = pearsonr(g1, g2)

    # Per-class agreement
    class_agreement = {}
    for cls in range(6):
        mask = (g1 == cls) | (g2 == cls)
        if mask.sum() > 0:
            class_agreement[str(cls)] = float(np.mean(g1[mask] == g2[mask]) * 100)

    results = {
        "n_samples": int(n),
        "percent_agreement": round(percent_agree, 2),
        "cohen_kappa": round(kappa, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "per_class_agreement_percent": {k: round(v, 2) for k, v in class_agreement.items()},
        "grader1_distribution": {str(i): int((g1 == i).sum()) for i in range(6)},
        "grader2_distribution": {str(i): int((g2 == i).sum()) for i in range(6)},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Agreement results saved to %s", out_path)
    logger.info("Percent agreement: %.2f%%", percent_agree)
    logger.info("Cohen's kappa: %.4f", kappa)
    logger.info("Quadratic-weighted kappa: %.4f", qwk)
    logger.info("Pearson r: %.4f (p=%.2e)", pearson_r, pearson_p)


if __name__ == "__main__":
    main()
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-10-15

### Added — Phase 1: Development & Reproducibility
- **Dataset switch to Mohler ASAG**: Replaced synthetic data plan with Mohler Short-Answer Grading corpus (`nkazi/MohlerASAG`, `open_ended` split, 2,273 answers, 81 questions, CC BY 4.0). Real human grader scores (`score_grader1`, `score_grader2`, `score_avg`) used to derive 4-class quality labels via thresholds in `params.yaml`.
- **Grouped split (GroupShuffleSplit)**: Implemented in `src/data/preprocess.py` — splits by `question_id` (56 train / 8 val / 17 test questions) to prevent question leakage.
- **Feature engineering pipeline**: TF-IDF (max 5,000 features, 1-2-grams, sublinear_tf; 2,818 kept after fitting) + 5 length features + 1 reference-answer cosine similarity = 2,824 total features. Fitted on train only (`src/features/build_features.py`).
- **Two baseline models**: 
  - Logistic Regression (C=1.0, lbfgs, max_iter=1000, class_weight=balanced)
  - Random Forest (n_estimators=300, max_depth=None, class_weight=balanced)
- **MLflow experiment tracking**: Experiment `interview-answer-quality-baseline` with runs for `logistic_regression_baseline`, `random_forest`, and `final_test_evaluation`. Hyperparameters, metrics, and confusion matrices logged.
- **DVC pipeline**: 7 stages plus the tracked `data/external/mohler.dvc` (`prepare_dataset`, `load_data`, `preprocess`, `feature_engineering`, `train_baseline`, `train_compare`, `evaluate`) with `dvc.lock` for reproducibility.
- **Model selection rule**: Higher grouped-CV f1_macro mean; if gap < max(CV std), tie-break by validation f1_macro. Result: Random Forest selected (CV gap 0.0027 < max std 0.0456; val f1_macro 0.3778 > 0.3418).
- **Test evaluation**: Logistic Regression macro-F1 0.3669, Random Forest macro-F1 0.3925, majority baseline 0.1868. Both beat majority on macro-F1 but lose on accuracy (0.5611, 0.5901 vs 0.5963).
- **Executed notebooks**: `01_data_exploration.ipynb`, `02_data_preprocessing.ipynb`, `03_feature_engineering.ipynb`, `04_baseline_models.ipynb`.
- **Phase 1 report**: `reports/phase1/phase1_report.md` with metrics, grader agreement, splits, features, models, selection, test results, data quality, limitations, reproducibility commands, MLflow run IDs, Phase 2 plan, and git log.

### Fixed
- **Bug: `score_avg` recomputed**: an early version of `src/data/prepare_dataset.py` recomputed `score_avg` from the normalized grader columns, which changed 114 labels. It now uses the dataset's own `score_avg` (asserted equal for all rows).
- **Bug: feature pipeline pickled under `__main__`**: `models/feature_pipeline.joblib` was saved with its custom classes under `__main__`, so loading it from another script failed (`AttributeError: TextLengthFeatures`). The `feature_engineering` stage now runs `python -m src.features.build_features`.

---

## [0.1.0] — 2026-09-27

### Added
- Professional MLOps project foundation and modular architecture.
- Full MLOps lifecycle design (Data, DVC, EDA, Features, Baseline, MLflow, FastAPI, Docker, CI/CD, Evidently, Prometheus, Grafana).
- Academic alignment details for Course AI252TA (Semester V) covering UN SDGs 4 & 8.
- Configuration foundation with `params.yaml` and `configs/config.yaml`.
- DVC pipeline specification scaffold (`dvc.yaml`).
- Phase 1 scaffold notebooks (EDA, Preprocessing, Feature Engineering, Baseline Modeling).
- Packaging, linting, formatting, and testing configuration (`pyproject.toml`, `requirements.txt`, `requirements-dev.txt`).
- Developer automation tools (`Makefile`) for environment setup, linting, and tests.
- Comprehensive documentation hub (`docs/`) for architecture, dataset, experiments, mlops, and governance.
- Project status tracker (`docs/PROJECT_STATUS.md`).
- Community governance and collaboration guidelines (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.github/`).
- Futuristic Orange/Dark themed documentation visuals and system diagrams.
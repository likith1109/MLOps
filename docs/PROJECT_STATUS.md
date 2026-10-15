# Project Status Dashboard

> **Project:** AI-Based Interview Preparation Assistant  
> **Course:** AI252TA – Machine Learning Operations  
> **Semester:** V  
> **Repository:** [https://github.com/Tejzraj/MLOps.git](https://github.com/Tejzraj/MLOps.git)  
> **Last Updated:** 2026-10-15  

---

## Current Status Overview

| Metric | Status |
| :--- | :--- |
| **Current Stage** | **PHASE 1 — DEVELOPMENT & REPRODUCIBILITY (Complete through Evaluation)** |
| **Foundation Readiness** | `██████████` **100%** |
| **Phase 1 (Development & Reproducibility)** | `████████████████░░` **80%** |
| **Phase 2 (Productionization & Deployment)** | `░░░░░░░░░░` **0% (Planned)** |
| **Phase 3 (Monitoring, Feedback & Governance)** | `░░░░░░░░░░` **0% (Planned)** |

---

## Milestone Breakdown

### Phase 0 — Foundation & Repository Architecture
- [x] Git repository configured with clean remote tracking
- [x] Standardized MLOps directory hierarchy initialized
- [x] Orange/Dark futuristic UI visual theme established
- [x] Configuration systems initialized (`params.yaml`, `configs/config.yaml`)
- [x] Dependency management defined (`requirements.txt`, `requirements-dev.txt`, `pyproject.toml`)
- [x] DVC initialized and pipeline blueprint created (`dvc.yaml`)
- [x] Developer workflows and linting automated (`Makefile`, `ruff`, `black`, `pytest`)
- [x] Comprehensive documentation framework created (`docs/`, `data/README.md`)
- [x] Open-source and academic governance added (`LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`)

### Phase 1 — Development & Reproducibility (Complete through Evaluation)
- [x] Dataset specification, labeling guidelines, and data collection protocol authored (`docs/dataset/`)
- [x] Faculty review dossier prepared (`docs/dataset/DATASET_APPROVAL.md`)
- [ ] Faculty committee review and formal approval (pending)
- [x] DVC initialized with pipeline blueprint (`dvc.yaml`, `dvc.lock`)
- [x] Dataset acquired: Mohler ASAG (`nkazi/MohlerASAG`), `open_ended` split, 2,273 answers, 81 questions, CC BY 4.0
- [x] Dataset versioned with DVC (`data/raw/interview_data.parquet` tracked in `dvc.lock`)
- [x] Data quality checks and schema verification implemented (`src/data/load_data.py`, `src/data/preprocess.py`)
- [x] Exploratory Data Analysis completed (`notebooks/01_data_exploration.ipynb`, executed)
- [x] Data preprocessing and cleaning pipeline implemented (`src/data/preprocess.py`)
- [x] Grouped (GroupShuffleSplit) train / validation / test partitioning executed (grouped by `question_id`, 56/8/17 questions)
- [x] Feature engineering pipeline constructed: TF-IDF (up to 5,000 terms, 1-2-grams; 2,818 kept) + 5 length features + 1 reference similarity = 2,824 features (`src/features/build_features.py`)
- [x] Baseline Logistic Regression model trained and logged (`src/models/train_baseline.py`, `models/baseline_model.joblib`)
- [x] Comparison Random Forest model trained and evaluated (`src/models/train_compare.py`, `models/rf_model.joblib`)
- [x] MLflow experiment tracking integrated (hyperparameters, metrics, confusion matrices in `mlruns/`)
- [x] Model selection rule implemented and applied (CV f1_macro tie → validation f1_macro tie-break → Random Forest selected)
- [x] Test evaluation completed (Logistic Regression macro-F1 0.3669, Random Forest macro-F1 0.3925, majority baseline 0.1868)
- [x] Full pipeline reproducibility verified using `dvc repro` (two consecutive runs skip all stages)
- [x] Phase 1 synthesis report authored in `reports/phase1/phase1_report.md`

### Phase 2 — Productionization & Deployment (Planned)
- [ ] PyTest test suite implemented (unit tests, integration tests, schema tests)
- [ ] Model behavioral and linguistic invariance tests added
- [ ] Model risk, edge-case, and latency profiling conducted
- [ ] MLflow Model Registry integration with stage transitions (`Staging`, `Production`)
- [ ] FastAPI REST inference service built with Pydantic payload validation
- [ ] Multi-stage Docker container built and optimized for inference
- [ ] GitHub Actions CI pipeline configured (lint, test, build)
- [ ] Continuous Deployment workflow configured to cloud environment

### Phase 3 — Observability & Continuous Monitoring (Planned)
- [ ] Structured request/response logging framework deployed
- [ ] Evidently AI integrated for data drift and vocabulary shift detection
- [ ] Prometheus metrics exporter configured for inference latency and class counts
- [ ] Grafana / Power BI operational dashboard configured
- [ ] Simulated production data generation script created
- [ ] Data and concept drift experiment simulated and documented

---

## Dataset Switch Summary

**Previous Plan:** Synthetic interview dataset across 10 technical domains with 4 LLM-generated quality tiers, human annotation via rubric.

**Actual (Phase 1):** Mohler Short-Answer Grading (ASAG) corpus from Hugging Face (`nkazi/MohlerASAG`, subset `cleaned`, split `open_ended`).
- 2,273 candidate answers across 81 questions
- Single introductory data-structures course (University of North Texas)
- Human grader scores already present (`score_grader1`, `score_grader2`, `score_avg`)
- Quality labels derived from `score_avg` using thresholds in `params.yaml` (0–5 scale → 4 classes)
- CC BY 4.0 license (Mohler, Bunescu & Mihalcea 2011)
- Replaced synthetic data plan because it provides authentic student answers with real human grader scores

**Known Limitations** (see `reports/phase1/phase1_report.md` for details):
1. Single course only — generalization unknown
2. Highly skewed labels (Excellent 62.8%, Poor 3.5%)
3. Tiny Poor test class (only 10 rows)
4. Only 8 validation and 17 test questions → high variance
5. Moderate grader agreement (quadratic-weighted κ = 0.4875)
6. Duplicates and cross-split text overlap in raw data
7. DVC remote is local-only — fresh clone requires re-download from Hugging Face

See full report: [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md)
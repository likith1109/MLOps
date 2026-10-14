<div align="center">

<img src="docs/assets/banner.svg" alt="AI-Based Interview Preparation Assistant MLOps Banner" width="100%" />

# AI-Based Interview Preparation Assistant
### Production-Grade MLOps Pipeline & Continuous Delivery Platform

<p align="center">
  <strong>Data</strong> ➔ 
  <strong>Version</strong> ➔ 
  <strong>EDA</strong> ➔ 
  <strong>Features</strong> ➔ 
  <strong>Train</strong> ➔ 
  <strong>Track</strong> ➔ 
  <strong>Register</strong> ➔ 
  <strong>Serve</strong> ➔ 
  <strong>Monitor</strong> ➔ 
  <strong>Retrain</strong>
</p>

<!-- Technology Badges -->
[![Python 3.11+](https://img.shields.io/badge/Python-3.11-FF6B00?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-v3.40_Data_Ops-FF8C00?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking_&_Registry-FFA726?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-FF6B00?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving_(Phase_2)-1A1A1A?style=for-the-badge&logo=fastapi&logoColor=FF8C00)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containers_(Phase_2)-1A1A1A?style=for-the-badge&logo=docker&logoColor=FF8C00)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Actions_(Phase_2)-1A1A1A?style=for-the-badge&logo=github-actions&logoColor=FF8C00)](https://github.com/features/actions)
[![Evidently](https://img.shields.io/badge/Evidently-Drift_Ops_(Phase_3)-1A1A1A?style=for-the-badge&logo=datadog&logoColor=FF8C00)](https://www.evidentlyai.com/)

<!-- Project & Academic Badges -->
[![Course](https://img.shields.io/badge/Course-AI252TA%3A%20MLOps-FF6B00?style=flat-square&logo=bookstack&logoColor=white)](https://github.com/Tejzraj/MLOps)
[![Semester](https://img.shields.io/badge/Semester-V-FF8C00?style=flat-square)](https://github.com/Tejzraj/MLOps)
[![Status](https://img.shields.io/badge/Project_Stage-Phase_0%20(Foundation)-FFA726?style=flat-square)](docs/PROJECT_STATUS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-0D1117?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![SDG 4](https://img.shields.io/badge/UN%20SDG-4%20Quality%20Education-FF6B00?style=flat-square)](https://sdgs.un.org/goals/goal4)
[![SDG 8](https://img.shields.io/badge/UN%20SDG-8%20Decent%20Work%20%26%20Economic%20Growth-FF8C00?style=flat-square)](https://sdgs.un.org/goals/goal8)

</div>

---

## Executive Summary

The **AI-Based Interview Preparation Assistant** is an end-to-end Machine Learning Operations (MLOps) platform designed to evaluate and coach candidates on technical and behavioral interview responses. The system analyzes interview question-answer pairs, assesses answer depth and correctness, and classifies response quality into a 4-tier standardized grading rubric to provide actionable, automated feedback.

Engineered for academic course **AI252TA (Machine Learning Operations - Semester V)**, this repository implements industry-standard MLOps practices: **versioned data pipelines with DVC**, **reproducible experiment tracking with MLflow**, **automated quality validation**, **containerized inference serving**, and **drift-triggered continuous retraining**.

> [!NOTE]
> **Foundation Status:** This project is at **Phase 0 (Foundation)**. No model weights, production deployments, or fabricated benchmark results are claimed. All milestones follow a structured, phased academic implementation plan.

---

## Quick Start (Phase 1)

**Prerequisites:** Python 3.11, Git, `make`.

```bash
# 1. Clone and enter the repository
git clone https://github.com/Tejzraj/MLOps.git
cd MLOps

# 2. Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Allow MLflow file store (required for local tracking)
export MLFLOW_ALLOW_FILE_STORE=true

# 4. Download the Mohler ASAG dataset (open_ended split) from Hugging Face
python src/data/download_mohler.py

# 5. Run the full DVC pipeline (data → split → features → train → evaluate)
dvc repro

# 6. Run tests
python -m pytest -q
```

**Dataset:** Mohler ASAG (Short Answer Grading) — `nkazi/MohlerASAG` on Hugging Face, subset `cleaned`, split `open_ended`, 2273 rows, CC BY 4.0 license (Mohler, Bunescu & Mihalcea 2011). This replaces the earlier synthetic interview data specification. The `data/external/mohler/open_ended.parquet` is DVC-tracked; teammates must run the download command above because the DVC remote (`~/dvc_storage`) is local-only and `dvc pull` will not work for them.

**Files not in Git (enforced by `.gitignore`):** `data/`, `models/*.joblib`, `mlruns/`, `.env`. Commit only code, config, and `dvc.lock`/`*.dvc` pointer files.

> [!NOTE]
> **Phase 1 is in progress.** This quickstart runs the current pipeline through evaluation; it does not imply Phase 1 completion or production readiness.

---

## Real-Time Milestone Dashboard

```text
========================================================================================
                                 PROJECT PROGRESSION
========================================================================================
Phase 0: Foundation & Environment    [████████████████████] 100% (Completed)
Phase 1: Development & Reproduce     [░░░░░░░░░░░░░░░░░░░░]   0% (In Preparation)
Phase 2: Serving & CI/CD Pipeline    [░░░░░░░░░░░░░░░░░░░░]   0% (Planned)
Phase 3: Observability & Governance  [░░░░░░░░░░░░░░░░░░░░]   0% (Planned)
========================================================================================
CURRENT ACTIVE PHASE : PHASE 1 — DATASET SPECIFICATION & ACADEMIC DEFENSE
NEXT MILESTONE       : FACULTY APPROVAL & DVC DATASET INGESTION
========================================================================================
```

---

## Phase 1 Lifecycle & Dataset Execution Status

```text
Phase 1 Execution Flow:
Dataset Specification ──► Dataset Acquisition ──► EDA ──► Feature Engineering ──► Baseline Model ──► MLflow Tracking
    [IN PROGRESS]             [NOT STARTED]     [NOT STARTED]    [NOT STARTED]        [NOT STARTED]       [NOT STARTED]
```

| Lifecycle Component | Status | Artifact / Documentation Reference |
| :--- | :---: | :--- |
| **Dataset Specification** | `[~] IN PROGRESS` | [`docs/dataset/DATASET_SPECIFICATION.md`](docs/dataset/DATASET_SPECIFICATION.md) |
| **Labeling Guidelines & Rubric** | `[~] IN PROGRESS` | [`docs/dataset/LABELING_GUIDELINES.md`](docs/dataset/LABELING_GUIDELINES.md) |
| **Data Collection Protocol** | `[~] IN PROGRESS` | [`docs/dataset/DATA_COLLECTION.md`](docs/dataset/DATA_COLLECTION.md) |
| **Faculty Approval Dossier** | `[ ] PENDING REVIEW` | [`docs/dataset/DATASET_APPROVAL.md`](docs/dataset/DATASET_APPROVAL.md) |
| **Dataset Acquisition & DVC Tracking** | `[ ] NOT STARTED` | Planned under `data/raw/` tracked via DVC |
| **Exploratory Data Analysis (EDA)** | `[ ] NOT STARTED` | [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) |
| **Feature Engineering Pipeline** | `[ ] NOT STARTED` | [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb), `src/features/` |
| **Baseline Model Training** | `[ ] NOT STARTED` | [`notebooks/04_baseline_models.ipynb`](notebooks/04_baseline_models.ipynb), `src/models/` |
| **MLflow Experiment Tracking** | `[ ] NOT STARTED` | Local & remote tracking configured in `mlruns/` |

---

## Machine Learning Problem Formulation

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       Interview Question        │   +   │        Candidate Answer         │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    ML Feature Extraction      │
                      │  (TF-IDF / Linguistic / Embed) │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ Answer Quality Classification │
                      └───────────────┬───────────────┘
                                      │
        ┌──────────────┬──────────────┴──────────────┬──────────────┐
        ▼              ▼                             ▼              ▼
     Class 0        Class 1                       Class 2        Class 3
    [ Poor ]  [ Needs Improvement ]              [ Good ]     [ Excellent ]
```

### Problem Definition
- **Input:** Multi-turn technical or behavioral interview prompt (`interview_question`) and raw candidate textual response (`candidate_answer`).
- **Target Output:** 4-class ordinal quality classification score (`quality_label` $\in \{0, 1, 2, 3\}$).
- **Evaluation Criteria:**
  - **Class 0 (Poor):** Irrelevant, incoherent, factually erroneous, or fewer than 5 tokens.
  - **Class 1 (Needs Improvement):** Superficial answer missing critical concepts or terminology.
  - **Class 2 (Good):** Factually accurate, technically sound, and clearly articulated.
  - **Class 3 (Excellent):** Comprehensive explanation including trade-offs, architecture patterns, and edge cases.

---

## End-to-End MLOps Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["Data Engineering & Versioning (Phase 1)"]
        Raw["Raw QA Dataset (data/raw)"] --> DVC["DVC Data Tracking"]
        DVC --> Preproc["Preprocessing & Splitting (src/data)"]
        Preproc --> Splits["Train / Val / Test Splits (data/processed)"]
    end

    subgraph Experimentation["Modeling & Tracking (Phase 1)"]
        Splits --> Features["Feature Engineering (src/features)"]
        Features --> Train["Baseline & Comparison Training (src/models)"]
        Train <--> MLflow["MLflow Experiment Tracking (Params, Metrics, Artifacts)"]
        Train --> Eval["Model Evaluation & Error Profiling (src/evaluation)"]
    end

    subgraph CI_CD_Deploy["Productionization & Deployment (Phase 2 - Planned)"]
        Eval --> Registry["MLflow Model Registry"]
        Registry --> API["FastAPI REST Inference Service"]
        API --> Docker["Docker Containerization"]
        Docker --> Actions["GitHub Actions CI/CD"]
        Actions --> Prod["Production API Endpoint"]
    end

    subgraph Monitoring_Loop["Observability & Continuous ML (Phase 3 - Planned)"]
        Prod --> Logs["Prediction & Telemetry Logging"]
        Logs --> Evidently["Evidently AI Drift Analysis"]
        Evidently --> Grafana["Prometheus & Grafana Dashboards"]
        Grafana --> DriftCheck{"Data/Concept Drift Detected?"}
        DriftCheck -- "Yes: Automated Trigger" --> Train
        DriftCheck -- "No: Stable" --> MonitorState["Continuous Monitoring"]
    end

    classDef orange fill:#FF6B00,stroke:#0D1117,stroke-width:2px,color:#FFFFFF;
    classDef dark fill:#161B22,stroke:#FF8C00,stroke-width:1.5px,color:#FFFFFF;
    classDef planned fill:#1F242C,stroke:#484F58,stroke-width:1px,stroke-dasharray: 4 4,color:#8B949E;

    class Raw,Splits,Features,Train,Eval orange;
    class DVC,Preproc,MLflow dark;
    class Registry,API,Docker,Actions,Prod,Logs,Evidently,Grafana,DriftCheck,MonitorState planned;
```

---

## Technology Stack

| Domain | Tool / Technology | Status | Role in Pipeline |
| :--- | :--- | :---: | :--- |
| **Language** | **Python 3.11+** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Core runtime environment across entire pipeline |
| **Data Ops** | **Pandas, NumPy** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Numerical transformations, dataframes, and sanitization |
| **Data Versioning** | **DVC (Data Version Control)** | ![Planned P1](https://img.shields.io/badge/Phase_1-FF8C00?style=flat-square) | Git-independent dataset versioning and pipeline orchestration |
| **Machine Learning** | **Scikit-learn** | ![Planned P1](https://img.shields.io/badge/Phase_1-FF8C00?style=flat-square) | Baseline classifiers, TF-IDF vectorization, evaluation metrics |
| **Prototyping** | **Jupyter Lab / Notebooks** | ![Active](https://img.shields.io/badge/Active-FF6B00?style=flat-square) | Exploratory data analysis, error inspection, visualization |
| **Experimentation** | **MLflow** | ![Planned P1](https://img.shields.io/badge/Phase_1-FF8C00?style=flat-square) | Experiment tracking, hyperparameter logging, model registry |
| **Testing** | **PyTest & Coverage** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | Unit, integration, and model invariance testing |
| **Serving API** | **FastAPI & Uvicorn** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | High-performance REST service for real-time inference |
| **Containerization** | **Docker** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | Multi-stage, reproducible microservice container builds |
| **CI/CD** | **GitHub Actions** | ![Planned P2](https://img.shields.io/badge/Phase_2-1A1A1A?style=flat-square) | Automated linting, test execution, container push |
| **Observability** | **Evidently AI** | ![Planned P3](https://img.shields.io/badge/Phase_3-1A1A1A?style=flat-square) | Text drift detection, data quality, and model degradation |
| **Metrics & UI** | **Prometheus & Grafana** | ![Planned P3](https://img.shields.io/badge/Phase_3-1A1A1A?style=flat-square) | System metric scraping and real-time operations dashboard |

---

## Detailed Multi-Phase Roadmap

### Legend
- `[x]` Completed
- `[~]` In Progress
- `[ ]` Not Started / Planned

```
Phase 0 ───► Phase 1 ───► Phase 2 ───► Phase 3 ───► Phase 4 ───► Phase 5
Foundation   Dev & Repro  Production   Observability Continuous ML Governance
 [100%]         [0%]         [0%]         [0%]         [0%]         [0%]
```

<details open>
<summary><strong>Phase 0 — Project Foundation & Architecture (COMPLETED)</strong></summary>

- [x] Repository initialized with clean remote tracking on `main`
- [x] Full MLOps directory hierarchy initialized (`src/`, `data/`, `notebooks/`, `docs/`, `tests/`)
- [x] Central parameter management configured (`params.yaml`, `configs/config.yaml`)
- [x] Dependency management configured (`requirements.txt`, `requirements-dev.txt`, `pyproject.toml`)
- [x] DVC pipeline blueprint initialized (`dvc.yaml`)
- [x] Developer automation scripts added (`Makefile`)
- [x] Project governance and licensing created (`LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`)
- [x] Custom SVG asset and architecture documentation authored
</details>

<details open>
<summary><strong>Phase 1 — Development & Reproducibility (CURRENT FOCUS)</strong></summary>

- [x] Dataset specification, 19-field schema, and multi-dimensional rubric authored
- [x] Faculty review dossier prepared (`docs/dataset/DATASET_APPROVAL.md`)
- [ ] Faculty committee review and formal approval (Pending)
- [x] DVC initialized with pipeline blueprint (`dvc.yaml`)
- [ ] Question-bank acquisition & candidate response collection/creation
- [ ] Human / rubric-based annotation and team dual-audit validation
- [ ] DVC raw dataset versioning (`data/raw/interview_data.parquet.dvc`)
- [ ] Data quality checks and schema verification implemented
- [ ] Exploratory Data Analysis completed (`notebooks/01_data_exploration.ipynb`)
- [ ] Data preprocessing and cleaning pipeline implemented (`src/data/`)
- [ ] Stratified grouped train / validation / test partitioning executed
- [ ] Feature engineering pipeline (TF-IDF, linguistic metrics) constructed (`src/features/`)
- [ ] Baseline Logistic Regression model trained and logged (`src/models/`)
- [ ] Comparison Random Forest model trained and evaluated
- [ ] MLflow experiment tracking integrated (hyperparameters, metrics, confusion matrices)
- [ ] Full pipeline reproducibility verified using `dvc repro`
- [ ] Phase 1 synthesis report authored in `reports/phase1/README.md`
</details>

<details>
<summary><strong>Phase 2 — Productionization & Deployment (PLANNED)</strong></summary>

- [ ] PyTest test suite implemented (unit tests, integration tests, schema tests)
- [ ] Model behavioral and linguistic invariance tests added
- [ ] Model risk, edge-case, and latency profiling conducted
- [ ] MLflow Model Registry integration with stage transitions (`Staging`, `Production`)
- [ ] FastAPI REST inference service built with Pydantic payload validation
- [ ] Multi-stage Docker container built and optimized for inference
- [ ] GitHub Actions CI pipeline configured (lint, test, build)
- [ ] Continuous Deployment workflow configured to cloud environment
</details>

<details>
<summary><strong>Phase 3 — Observability & Continuous Monitoring (PLANNED)</strong></summary>

- [ ] Structured request/response logging framework deployed
- [ ] Evidently AI integrated for data drift and vocabulary shift detection
- [ ] Prometheus metrics exporter configured for inference latency and class counts
- [ ] Grafana / Power BI operational dashboard configured
- [ ] Simulated production data generation script created
- [ ] Data and concept drift experiment simulated and documented
</details>

<details>
<summary><strong>Phase 4 — Continuous ML & Automated Retraining (PLANNED)</strong></summary>

- [ ] Automated drift-triggered retraining orchestrator designed
- [ ] Shadow model deployment and automated candidate comparison evaluation
- [ ] Automated rollback mechanism on performance degradation
- [ ] Continuous model artifact promotion into Model Registry
</details>

<details>
<summary><strong>Phase 5 — Governance, Compliance & Auditability (PLANNED)</strong></summary>

- [ ] Mitchell et al. Model Card authored (`docs/governance/model_card.md`)
- [ ] Dataset provenance and ethical assessment document completed
- [ ] Fairness and dialectal bias evaluation report published
- [ ] Comprehensive academic viva demonstration and reproducible run documentation
</details>

---

## Repository Structure

```text
MLOps/
│
├── README.md                          # Project documentation and architectural overview
├── LICENSE                            # Open-source MIT License
├── CONTRIBUTING.md                    # Developer guidelines and Git workflows
├── CODE_OF_CONDUCT.md                 # Contributor Covenant Code of Conduct
├── CHANGELOG.md                       # Release notes and milestone log
├── .gitignore                         # Comprehensive Python, DVC, and artifact ignores
├── .gitattributes                     # LF normalization and binary file tracking
├── requirements.txt                   # Production ML & MLOps dependencies
├── requirements-dev.txt               # Development, linting, and testing dependencies
├── pyproject.toml                     # Modern build system, ruff, black, and pytest configuration
├── Makefile                           # Developer automation commands
├── dvc.yaml                           # DVC reproducible pipeline stages
├── params.yaml                        # Centralized pipeline hyperparameters
│
├── configs/
│   └── config.yaml                    # Global system and logging configurations
│
├── data/
│   ├── raw/                           # Raw immutable QA datasets (DVC-tracked)
│   ├── interim/                       # Cleaned intermediate QA data
│   ├── processed/                     # Final train/val/test splits
│   └── README.md                      # Dataset specification and governance
│
├── notebooks/                         # Interactive Jupyter scaffolds for Phase 1
│   ├── 01_data_exploration.ipynb      # EDA and quality distribution analysis
│   ├── 02_data_preprocessing.ipynb    # Text sanitization and stratified splitting
│   ├── 03_feature_engineering.ipynb   # TF-IDF and syntactic feature extraction
│   └── 04_baseline_models.ipynb       # Baseline models & MLflow experiment runs
│
├── src/                               # Modular source package
│   ├── __init__.py
│   ├── data/                          # Ingestion, validation, and cleaning logic
│   │   └── __init__.py
│   ├── features/                      # Feature extractors and text transformers
│   │   └── __init__.py
│   ├── models/                        # Baseline modeling and training scripts
│   │   └── __init__.py
│   ├── evaluation/                    # Metric computation and evaluation plots
│   │   └── __init__.py
│   └── utils/                         # Shared utilities, YAML loaders, logging
│       └── __init__.py
│
├── models/                            # Serialized model weights (.joblib, DVC-tracked)
│   └── .gitkeep
│
├── reports/                           # Generated evaluation outputs and documentation
│   ├── figures/                       # Confusion matrices and distribution plots
│   │   └── .gitkeep
│   └── phase1/
│       └── README.md                  # Phase 1 evaluation report placeholder
│
├── tests/                             # PyTest test suite
│   └── README.md                      # Testing architecture and execution guide
│
├── mlruns/                            # Local MLflow tracking experiments (ignored by Git)
│   └── .gitkeep
│
├── docs/                              # Technical documentation hub
│   ├── PROJECT_STATUS.md              # Live milestone status dashboard
│   ├── assets/
│   │   └── banner.svg                 # Neon orange futuristic control center banner
│   ├── architecture/README.md         # System design and component interactions
│   ├── dataset/README.md              # Dataset approval and annotation rubrics
│   ├── experiments/README.md          # Experiment tracking and MLflow protocol
│   ├── mlops/README.md                # CI/CD, Docker, and reproducibility specifications
│   └── governance/README.md           # Model cards, ethics, and bias analysis
│
└── .github/                           # GitHub repository configuration
    ├── workflows/                     # GitHub Actions CI/CD workflows
    │   └── .gitkeep
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md              # Structured bug report template
    │   └── feature_request.md         # Structured feature proposal template
    └── PULL_REQUEST_TEMPLATE.md       # Pull request checklist
```

---

## Academic Alignment & Sustainable Development Goals

### Course Context
- **Course Title:** Machine Learning Operations
- **Course Code:** `AI252TA`
- **Academic Semester:** Semester V

### Demonstrable MLOps Core Competencies
1. **Data Lifecycle Management:** Decoupled storage, versioning, and provenance via DVC.
2. **Reproducible Experimentation:** Controlled hyperparameter tracking and model metrics via MLflow.
3. **Engineering Rigor:** Automated linting (`ruff`), formatting (`black`), testing (`pytest`), and environment isolation.
4. **Production Architecture:** Microservice inference (`FastAPI`) inside standardized container images (`Docker`).
5. **Observability & Continuous Learning:** Automated drift surveillance (`Evidently`) and metric dashboards (`Prometheus`/`Grafana`).
6. **Governance & Responsible AI:** Detailed Model Cards, candidate data privacy enforcement, and evaluation fairness.

### United Nations Sustainable Development Goals
- **SDG 4 (Quality Education):** Provides students and job candidates with objective, automated, and accessible assessment of interview communication quality, closing access gaps to elite interview coaching.
- **SDG 8 (Decent Work & Economic Growth):** Equips job seekers with personalized feedback to identify weak knowledge areas, improving employability and economic mobility.

---

## Planned Viva & Live Demonstration Milestones

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT DEMONSTRATION MATRIX                           │
├─────────┬──────────────────────────────────────────┬────────────────────────────┤
│ Phase   │ Demonstration Workflow                   │ Expected Artifacts         │
├─────────┼──────────────────────────────────────────┼────────────────────────────┤
│ Phase 1 │ Raw Data ➔ DVC Repro ➔ EDA ➔ MLflow Runs │ DVC DAG, MLflow UI Metrics │
│ Phase 2 │ Model Registry ➔ FastAPI ➔ Docker ➔ CI/CD│ OpenAPI Docs, Docker Image │
│ Phase 3 │ Prod Simulation ➔ Drift Detection ➔ Train│ Evidently Drift Reports    │
└─────────┴──────────────────────────────────────────┴────────────────────────────┘
```

> [!IMPORTANT]
> The demonstration workflows above represent formal academic deliverables. Each will be made demonstratable sequentially as project milestones are completed.

---

## Data Privacy & Security Governance

1. **Zero Personally Identifiable Information (PII):** Candidate answers and question datasets must be fully scrubbed of names, personal identifiers, contact details, and proprietary organizational data.
2. **Secret Hygiene:** No API keys, database credentials, or environment secrets may ever be committed to Git. Local secrets are stored in `.env` (strictly enforced by `.gitignore`).
3. **Model & Data Integrity:** Large dataset files and model binaries are managed exclusively via DVC and MLflow artifacts, ensuring the Git repository remains lightweight and auditable.
4. **Fairness & Dialectal Neutrality:** Language models will be assessed for stylistic bias to ensure non-native phrasing is not unfairly penalized when technical content is correct.

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Git & Git LFS
- Make (optional, for developer shortcut commands)

### 2. Clone Repository
```bash
git clone https://github.com/Tejzraj/MLOps.git
cd MLOps
```

### 3. Environment Initialization
```bash
# Create virtual environment
make setup-env
source .venv/bin/activate

# Install core and development dependencies
make install-dev
```

### 4. Verify Development Tooling
```bash
# Check formatting and linting
make lint

# Run initial test suite verification
make test
```

### 5. Launch Local Services (Scaffolded)
```bash
# Launch MLflow tracking dashboard
make mlflow-ui

# Inspect DVC pipeline status
make dvc-status
```

---

## Maintainer & Contributions

- **Repository Maintainer:** Tejzraj ([@Tejzraj](https://github.com/Tejzraj))
- **Course:** AI252TA – Machine Learning Operations, Semester V
- **Contributions:** Please read [CONTRIBUTING.md](CONTRIBUTING.md) and adhere to the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

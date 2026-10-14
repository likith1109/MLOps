# MLOps Engineering, CI/CD & Operations

> **Project:** AI-Based Interview Preparation Assistant  
> **Course:** AI252TA – Machine Learning Operations | Semester V  
> **Status:** Phase 0 / Phase 1 Inception

---

## 1. Directory Purpose

This directory contains MLOps engineering specifications covering CI/CD automation, Docker containerization, DVC pipelines, testing standards, and observability architecture.

---

## 2. DVC Remote Setup

The project uses a local DVC remote for storing data and model artifacts. The remote is configured at `../dvc_storage` (outside the repo).

### Initial setup (run once):

```bash
# Initialize DVC (if not already initialized)
dvc init

# Remove data/external/ from .gitignore (DVC refuses to track Git-ignored paths)
# Then track the external dataset
dvc add data/external/mohler

# Add default local remote
dvc remote add -d localstore ../dvc_storage
```

### Reproduce pipeline up to feature engineering:

```bash
# Run pipeline stages up to feature_engineering
dvc repro feature_engineering

# Push artifacts to remote storage
dvc push
```

### Pull data from remote (on another machine/CI):

```bash
# Pull all DVC-tracked files
dvc pull
```

### Check status:

```bash
# Show DVC pipeline status
dvc status

# Show Git status (which files to commit)
git status --short
```

---

## 3. Planned Operational Documentation

- **`reproducibility_protocol.md`:** Step-by-step instructions to reproduce data pipelines and model runs end-to-end via `dvc repro`.
- **`ci_cd_architecture.md`:** GitHub Actions workflows detailing pull request checks (linting, testing) and automated deployment triggers.
- **`containerization_guide.md`:** Docker image build instructions, security scanning, and multi-stage container optimization.
- **`monitoring_and_drift.md`:** Specification for Evidently data drift reports and Prometheus/Grafana metrics scraping.

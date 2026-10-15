# Dataset Documentation & Governance

> **Project:** AI-Based Interview Preparation Assistant<br>
> **Course:** AI252TA – Machine Learning Operations | Semester V<br>
> **Status:** Phase 1 — Mohler ASAG Dataset Adopted & Evaluation Complete

---

## 1. Directory Purpose

This directory serves as the centralized technical, legal, and pedagogical governance hub for all data used in the **AI-Based Interview Preparation Assistant**.

To ensure full academic defensibility and production-grade MLOps rigor, all dataset specifications, collection pipelines, scoring rubrics, and faculty review dossiers are maintained as version-controlled engineering documents.

---

## 2. Core Dataset Documentation Suite

| Document | Primary Focus | Status |
| :--- | :--- | :---: |
| **[`DATASET_SPECIFICATION.md`](DATASET_SPECIFICATION.md)** | Mohler ASAG dataset specification: ML formulation, 8-field schema, label derivation from human grader scores, grouped leakage prevention, data quality notes, limitations. | `[x] FINALIZED` |
| **[`DATA_COLLECTION.md`](DATA_COLLECTION.md)** | Actual data acquisition pipeline: Mohler download, preparation, label derivation, grouped splitting, feature engineering, model training, evaluation. | `[x] COMPLETED` |
| **[`LABELING_GUIDELINES.md`](LABELING_GUIDELINES.md)** | Original proposed annotation rubric (3 dimensions × 0–5) for reference; actual Phase 1 labels derived from Mohler's pre-existing human grader scores (`score_avg`). | `[x] REFERENCE` |
| **[`DATASET_APPROVAL.md`](DATASET_APPROVAL.md)** | Faculty-facing review dossier with Mohler ASAG provenance, actual schema, label derivation, grader agreement, test results, and committee acknowledgment form. | `[x] PREPARED (faculty review pending)` |

---

## 3. Current Dataset Status

**Dataset:** Mohler Short-Answer Grading (ASAG) — `nkazi/MohlerASAG` on Hugging Face, subset `cleaned`, split `open_ended`.

**Key Facts:**
- 2,273 candidate answers across 81 questions
- Single introductory data-structures course (University of North Texas)
- Two human grader scores per answer (0–5 scale) → `score_avg` → 4-class quality labels via `params.yaml` thresholds
- CC BY 4.0 license (Mohler, Bunescu & Mihalcea 2011)
- Replaces earlier synthetic data plan (10 domains, 4 LLM-generated tiers)

**Files (DVC-tracked, not in Git):**
- `data/raw/interview_data.parquet` (137 KB)
- `data/raw/interview_data.csv` (695 KB)
- `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- `models/feature_pipeline.joblib`, `baseline_model.joblib`, `rf_model.joblib`

**Reproducibility:**
```bash
python src/data/download_mohler.py
dvc repro
python -m pytest -q
```

**Phase 1 Report:** [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md)

---

## 4. What Changed from Original Plan

| Aspect | Original Plan (Synthetic) | Actual Phase 1 (Mohler ASAG) |
| :--- | :--- | :--- |
| **Data Source** | 10 domains, synthetic Q&A | 1 domain (Data Structures), real student answers |
| **Annotation** | Student team rubric (3 dims × 0–5) | Pre-existing 2 human graders (0–5 each) |
| **Label Derivation** | 0–15 total → 4 tiers | `score_avg` (0–5) → 4 tiers via thresholds |
| **Class Balance** | Target 25% ± 3% per tier | Highly skewed (Excellent 62.8%) |
| **Grader Agreement** | To be measured (dual-audit) | QWK = 0.4875 (original graders) |
| **Faculty Approval** | Required before annotation | Dossier prepared; review pending |

---

## 5. Physical Storage Conventions

See [`data/README.md`](../../data/README.md) for DVC remote configuration and storage paths.
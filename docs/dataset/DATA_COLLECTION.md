# Data Collection & Annotation Protocol

> **Project:** AI-Based Interview Preparation Assistant  
> **Course:** AI252TA – Machine Learning Operations | Semester V  
> **Document Status:** COMPLETED — Mohler ASAG Dataset Adopted  
> **Last Updated:** 2026-10-15  

---

## 1. Overview & Core Methodological Principle

This document outlines the end-to-end data acquisition protocol for the **AI-Based Interview Preparation Assistant**. The project adopted the **Mohler Short-Answer Grading (ASAG)** corpus from Hugging Face (`nkazi/MohlerASAG`, subset `cleaned`, split `open_ended`) as its Phase 1 dataset, replacing the earlier plan for synthetic interview data creation and human annotation.

> [!IMPORTANT]
> **Defensibility & Scientific Integrity:**
> 1. **Real Human Grader Scores:** The Mohler dataset contains scores from two human graders (`score_grader1`, `score_grader2`) on a 0–5 scale for each candidate answer. The average (`score_avg`) is used to derive quality labels via thresholds in `params.yaml`.
> 2. **No Automated Ground Truth:** Quality labels are derived from human scores, not from LLM-generated judgments.
> 3. **Reference Answer Role:** The reference answer is an expert answer from the dataset, used to assist evaluation models by illustrating required depth. It is an evaluation support tool, **not** automatically ground truth for candidate answer quality.
> 4. **Honest Project Status:** The synthetic data collection and annotation pipeline described in earlier versions was **NOT EXECUTED**. The Mohler ASAG dataset was adopted instead.

---

## 2. Dataset Acquisition Summary

### What Was Done (Phase 1 Actual)
1. **Dataset Identification:** Mohler ASAG corpus identified on Hugging Face (`nkazi/MohlerASAG`).
2. **Subset Selection:** `cleaned` subset, `open_ended` split selected (2,273 answers, 81 questions).
3. **Download & Versioning:** `src/data/download_mohler.py` downloads the dataset; `src/data/prepare_dataset.py` saves as Parquet and CSV to `data/raw/`, tracked by DVC (`dvc.lock`).
4. **Label Derivation:** the dataset's own `score_avg` (mean of the two human graders; it is not recomputed) is mapped to 4 classes with the thresholds in `params.yaml` (Poor < 2.0, Needs Improvement < 3.5, Good < 4.5, Excellent ≥ 4.5).
5. **Grouped Splitting:** `src/data/preprocess.py` splits by `question_id` using a grouped split (GroupShuffleSplit) (56 train / 8 val / 17 test questions) to prevent question leakage.

### What Was NOT Done (Original Plan — Superseded)
- Question-bank acquisition from multiple sources (`raghu298`, academic curricula, etc.)
- Candidate response creation/collection across 4 proficiency tiers
- Human/rubric-based annotation by student team members
- Dual-audit subset review and dispute resolution
- Class parity balancing (25% ± 3% per tier)

The original plan assumed synthetic data generation and human annotation. The Mohler dataset already contains human grader scores, so the annotation step was inherently completed by the dataset creators (Mohler et al., 2011).

---

## 3. Mohler ASAG Dataset Details

| Property | Value |
| :--- | :--- |
| **Source** | Hugging Face: `nkazi/MohlerASAG` |
| **Subset** | `cleaned` |
| **Split Used** | `open_ended` (2,273 rows, 81 questions) |
| **Split Not Used** | `close_ended` (constrained answer formats) |
| **Course** | Introductory Data Structures (University of North Texas) |
| **License** | CC BY 4.0 |
| **Citation** | Mohler, M., Bunescu, R., & Mihalcea, R. (2011). Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments. *Proceedings of ACL-HLT 2011*. |

### Raw Data Columns (from `data/raw/interview_data.parquet`)
- `question_id`: Unique question identifier
- `question`: Question text
- `reference_answer`: Expert reference answer
- `candidate_answer`: Student candidate answer
- `score_grader1`: First grader score (0–5)
- `score_grader2`: Second grader score (0–5)
- `score_avg`: Mean of two grader scores (0–5)
- `quality_label`: Derived class (0=Poor, 1=Needs Improvement, 2=Good, 3=Excellent)

### Grader Agreement (from `reports/annotation_agreement.json`)
- Quadratic-weighted kappa: 0.4875
- Percent agreement: 58.69%
- Cohen's kappa: 0.2988
- Pearson r: 0.5832
- 89 rows have ambiguous grader scale (affects agreement stats only, not labels)

---

## 4. Data Pipeline Implementation

The following scripts implement the data pipeline (all runnable via `python src/...`):

| Script | Purpose |
| :--- | :--- |
| `src/data/download_mohler.py` | Downloads Mohler ASAG from Hugging Face, saves to `data/external/mohler/` |
| `src/data/prepare_dataset.py` | Converts to Parquet/CSV, saves to `data/raw/`, DVC-tracked |
| `src/data/load_data.py` | Loads raw data, computes `score_avg`, derives `quality_label` using `params.yaml` thresholds |
| `src/data/preprocess.py` | Grouped split (GroupShuffleSplit) by `question_id` → `data/processed/train.parquet`, `val.parquet`, `test.parquet` |
| `src/features/build_features.py` | TF-IDF + length + reference similarity features, fitted on train only |

### DVC Pipeline Stages
1. `prepare_dataset` → `data/raw/interview_data.parquet`, `data/raw/interview_data.csv`
2. `load_data` → `data/interim/loaded_data.parquet`
3. `preprocess` → `data/processed/train.parquet`, `val.parquet`, `test.parquet`
4. `feature_engineering` → `models/feature_pipeline.joblib`
5. `train_baseline` → `models/baseline_model.joblib` (Logistic Regression)
6. `train_compare` → `models/rf_model.joblib` (Random Forest)
7. `evaluate` → `reports/metrics.json`, confusion matrices in `reports/figures/`

Run full pipeline: `dvc repro`

---

## 5. Data Quality Checks (Implemented)

Programmatic checks in `src/data/load_data.py` and `src/data/preprocess.py`:
- Zero nulls in required columns
- `score_avg` computed as mean of two grader scores
- `quality_label` derived from `score_avg` using `params.yaml` thresholds
- Grouped split: zero overlapping `question_id` between train/val/test
- Class distribution logged for each split

### Known Data Quality Issues (from Phase 1 Report)
- **Duplicates:** 143 rows with duplicate `candidate_answer` (45 unique); 98 redundant copies. 105 fully duplicated rows; 68 fully redundant. 1 `(question_id, candidate_answer)` group with >1 `quality_label`.
- **Cross-split overlap:** 19 test rows (3.93%) share answer text with train; 6 unique strings overlap train/test.
- These are documented as limitations; no data was modified after seeing test results.

---

## 6. Use of AI-Assisted Synthetic Responses

> [!NOTE]
> **The original plan for AI-assisted synthetic response generation was NOT executed.** The Mohler ASAG dataset provides authentic student answers with real human grader scores. No LLM-generated candidate answers or labels were used.

---

## 7. Pipeline Summary & Status Matrix

| Pipeline Stage | Functional Role | Implementation | Status |
| :--- | :--- | :--- | :---: |
| 1. Dataset Download | Mohler ASAG from Hugging Face | `src/data/download_mohler.py` | `[x] COMPLETED` |
| 2. Dataset Preparation | Parquet/CSV export, DVC tracking | `src/data/prepare_dataset.py` | `[x] COMPLETED` |
| 3. Label Derivation | `score_avg` → `quality_label` | `src/data/prepare_dataset.py` | `[x] COMPLETED` |
| 4. Grouped Splitting | Train/Val/Test by `question_id` | `src/data/preprocess.py` | `[x] COMPLETED` |
| 5. Feature Engineering | TF-IDF + length + ref similarity | `src/features/build_features.py` | `[x] COMPLETED` |
| 6. Baseline Training | Logistic Regression | `src/models/train_baseline.py` | `[x] COMPLETED` |
| 7. Comparison Training | Random Forest | `src/models/train_compare.py` | `[x] COMPLETED` |
| 8. Evaluation | Test metrics, confusion matrices | `src/evaluation/evaluate.py` | `[x] COMPLETED` |

---

## 8. Next Immediate Action

Phase 1 is complete through evaluation. Next steps (Phase 2):
- Model serving (FastAPI)
- MLflow Model Registry integration
- Docker containerization
- GitHub Actions CI/CD
- Drift detection (Evidently AI)

See [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md) for full Phase 1 results and Phase 2 plan.
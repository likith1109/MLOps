# Dataset Approval Dossier

> **Academic Course:** AI252TA – Machine Learning Operations  
> **Academic Semester:** Semester V  
> **Project Title:** AI-Based Interview Preparation Assistant  
> **Student / Maintainer:** Tejzraj ([@Tejzraj](https://github.com/Tejzraj))  
> **Repository:** [https://github.com/Tejzraj/MLOps.git](https://github.com/Tejzraj/MLOps.git)  
> **Document Status:** DATASET ADOPTED — Mohler ASAG (`nkazi/MohlerASAG`) Used for Phase 1  
> **Submission Date:** 2026-10-15  

---

## 1. Problem Statement

Automating interview preparation requires evaluating candidate answers to structured technical questions. Existing public datasets primarily feature question-and-answer pairs designed for chatbot question answering (SFT) or knowledge retrieval, **which is why the original plan assumed graded answers would have to be created; the adopted Mohler corpus provides them**.

To train an objective, reproducible machine learning evaluator, we adopted the **Mohler Short-Answer Grading (ASAG)** corpus, which provides:
- 2,273 authentic student answers from a real university course
- Scores from two human graders on a 0–5 scale
- Expert reference answers for each question
- CC BY 4.0 license permitting academic use

**ML Formulation:**
$$\text{Input: (Question, Candidate Answer, Reference Answer)}$$
$$\text{Output: Answer Quality Class } (0, 1, 2, 3) \text{ derived from } \texttt{score\_avg} \in [0, 5]$$

---

## 2. Why Mohler ASAG Was Chosen

1. **Targeted Task Alignment:** The Mohler corpus is specifically designed for short-answer grading — the exact ML task required.
2. **Real Human Grader Scores:** Two independent human graders scored each answer, providing realistic label noise and ambiguity.
3. **Single overall score:** each answer has two human grader scores and their average (`score_avg`); the dataset has no separate correctness, completeness or communication scores.
4. **MLOps Lifecycle Defense:** Serves as the foundational versioned asset for DVC pipeline tracking, data drift monitoring, and reproducible retraining.
5. **License Compliance:** CC BY 4.0 permits unrestricted academic use with attribution.
6. **Replaces Synthetic Plan:** The original plan for synthetic data across 10 domains with 4 LLM-generated tiers was abandoned because it would require fabricating both answers and labels, introducing uncontrolled bias.

---

## 3. Data Source & Provenance

| Property | Value |
| :--- | :--- |
| **Dataset** | Mohler Short-Answer Grading (ASAG) |
| **Hugging Face** | `nkazi/MohlerASAG` |
| **Subset** | `cleaned` |
| **Split Used** | `open_ended` (2,273 rows, 81 questions) |
| **Split Not Used** | `close_ended` (constrained answer formats) |
| **Course** | Introductory Data Structures (University of North Texas) |
| **License** | CC BY 4.0 |
| **Citation** | Mohler, M., Bunescu, R., & Mihalcea, R. (2011). Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments. *Proceedings of ACL-HLT 2011*. |

**Original Candidate Sources (Not Used):**
- `raghu298/ml-interview-sft-dataset` (Apache-2.0) — only reference answers, no graded candidates
- `Max00035/ml-systems-interview-bench` — license verification required
- `Ankshi/hr-interview-dataset` — license verification required
- Academic curricula (OS, DBMS, CN, OOP, DSA) — would require synthetic candidate generation

---

## 4. Dataset Schema (Actual)

The final dataset at `data/raw/interview_data.parquet` contains 8 columns:

| Field | Type | Description |
| :--- | :--- | :--- |
| `question_id` | string | Unique question identifier |
| `question` | string | Question text prompt |
| `reference_answer` | string | Expert reference answer from dataset |
| `candidate_answer` | string | Student candidate answer |
| `score_grader1` | float64 | First grader score (0–5) |
| `score_grader2` | float64 | Second grader score (0–5) |
| `score_avg` | float64 | Mean of two grader scores (0–5) |
| `quality_label` | int64 | Target class: 0=Poor, 1=Needs Improvement, 2=Good, 3=Excellent |

**Note:** The original 19-field schema (with `domain`, `difficulty`, `question_type`, `expected_concepts`, `correctness_score`, `completeness_score`, `communication_score`, `total_score`, `source`, `source_license`, `annotator_id`, `annotation_method`, `version`, `split_group`) was designed for the synthetic data plan. The Mohler dataset provides a simpler schema that directly supports the core ML task.

---

## 5. Label Derivation (Actual)

No new human annotation was performed. Labels derived programmatically:

1. **Source:** `score_avg = (score_grader1 + score_grader2) / 2` (range 0.0–5.0)
2. **Thresholds (from `params.yaml`):**

| quality_label | Class Name | score_avg Range |
| :---: | :--- | :--- |
| 0 | Poor | [0.0, 2.0) |
| 1 | Needs Improvement | [2.0, 3.5) |
| 2 | Good | [3.5, 4.5) |
| 3 | Excellent | [4.5, 5.0] |

3. **Distribution:**

| Class | Count | Percentage |
| :--- | ---: | ---: |
| Poor | 80 | 3.52% |
| Needs Improvement | 370 | 16.28% |
| Good | 396 | 17.42% |
| Excellent | 1,427 | 62.78% |

---

## 6. Grader Agreement (Mohler Original Graders)

From `reports/annotation_agreement.json` (computed on all 2,273 rows):

| Metric | Value |
| :--- | ---: |
| Quadratic-weighted kappa | 0.4875 |
| Percent agreement | 58.69% |
| Cohen's kappa | 0.2988 |
| Pearson r | 0.5832 |

**Note:** 89 rows have an ambiguous grader scale (one grader used 0–5, the other 1–5); these affect the agreement statistics but not the labels used for modeling.

---

## 7. Quality-Control & Anti-Leakage Methodology

- **Zero Nulls:** Programmatic validation in `src/data/load_data.py` rejecting missing or incomplete entries.
- **Grouped Anti-Leakage Partitioning:** Splitting strictly by `question_id` (Train 56 questions / Val 8 questions / Test 17 questions), guaranteeing that no question present in the training set appears in validation or testing. Implemented in `src/data/preprocess.py`.
- **Feature Pipeline Fitted on Train Only:** TF-IDF and engineered features fitted on train split only, applied to val/test. Implemented in `src/features/build_features.py`.
- **Cross-Split Overlap Check:** 19 test rows (3.93%) share answer text with train; documented as limitation.

---

## 8. Privacy & Ethics

- **Zero PII:** The Mohler dataset contains no student names, institutions, USNs, contact info, or IP addresses.
- **Controlled Text:** Uses existing published dataset; no new data collection from human subjects.

---

## 9. Licensing & Provenance Compliance

- Dataset: CC BY 4.0 (Mohler, Bunescu & Mihalcea 2011)
- Attribution included in all documentation
- DVC tracking ensures provenance metadata preserved

---

## 10. Known Dataset Limitations

1. **Single course**: Only one introductory data-structures course (UNT); generalization unknown.
2. **Skewed labels**: Excellent class is 62.8% of data; Poor is 3.5%.
3. **Tiny Poor test class**: Only 10 test rows → unstable per-class estimates.
4. **Few questions**: 8 validation and 17 test questions → high variance in grouped CV.
5. **Thresholds are a design choice**: The four bins (Poor <2.0, Needs Improvement <3.5, Good <4.5, Excellent ≥4.5) are a design choice, fixed in params.yaml before training.
6. **Moderate grader agreement**: Quadratic-weighted kappa 0.4875; labels have noise.
7. **High variance across questions**: CV f1_macro std ~0.045.
8. **Duplicates and cross-split text overlap** in raw data (143 rows with duplicate candidate answers; 19 test rows overlap train).
9. **DVC remote is local-only**: Data is not on GitHub; fresh clone requires re-downloading from Hugging Face.
10. **Faculty notification**: Faculty have not yet been informed of the dataset change; this is pending.

---

## 11. Data Versioning with DVC

- All datasets tracked outside Git via Data Version Control (DVC).
- Git tracks the lightweight pointer file: `dvc.lock` (and `dvc.yaml` for pipeline).
- Remote cache is local (`~/dvc_storage`); teammates must run `python src/data/download_mohler.py` then `dvc repro` because `dvc pull` will not work for them.
- Pipeline stages: `prepare_dataset` → `load_data` → `preprocess` → `feature_engineering` → `train_baseline` → `train_compare` → `evaluate`.

---

## 12. Faculty Review & Approval Checklist

The Mohler ASAG dataset was adopted for Phase 1 based on the following criteria:

```
========================================================================================
                      FACULTY EVALUATION COMMITTEE REVIEW FORM
========================================================================================
Course Code : AI252TA – Machine Learning Operations
Semester    : V
Project     : AI-Based Interview Preparation Assistant
Review Date : ________________________

[ ] 1. Dataset Appropriateness Reviewed
       Mohler ASAG corpus (short-answer grading) is appropriate for the ML task
       of 4-class answer quality classification.

[ ] 2. Data Source & License Approved
       Hugging Face `nkazi/MohlerASAG`, subset `cleaned`, split `open_ended`,
       CC BY 4.0 license verified.

[ ] 3. Label Derivation Methodology Approved
       Quality labels derived from mean of two human grader scores (`score_avg`)
       using thresholds in `params.yaml` (0–5 → 4 classes) is academically sound.

[ ] 4. Grouped Anti-Leakage Splitting Approved
       Split by `question_id` (56/8/17 questions) prevents question leakage
       across train/val/test partitions.

[ ] 5. Feature Engineering Approved
       TF-IDF (5,000, n-gram 1-2) + 5 length features + 1 reference similarity
       (2,824 total), fitted on train only.

[ ] 6. Baseline Models Approved
       Logistic Regression and Random Forest with class_weight=balanced,
       grouped CV, MLflow tracking.

[ ] 7. Model Selection Rule Approved
       Higher grouped-CV f1_macro; if gap < max(CV std), tie-break by val f1_macro.
       Result: Random Forest selected (CV gap 0.0027 < 0.0456; val 0.3778 > 0.3418).

[ ] 8. Test Evaluation Results Reviewed
       LR macro-F1 0.3669, RF macro-F1 0.3925, majority baseline 0.1868.
       Both beat majority on macro-F1, both lose on accuracy (0.5611, 0.5901 vs 0.5963).

[ ] 9. Limitations Acknowledged
       Single course, skewed labels, tiny Poor test class, few questions,
       moderate grader agreement, duplicates, local-only DVC remote.

[ ] 10. Reproducibility Verified
       `dvc repro` ×2 skips all stages; 25 pytest tests pass; MLflow run IDs recorded.

----------------------------------------------------------------------------------------
FINAL COMMITTEE ACTION:
[ ] ACKNOWLEDGED — Phase 1 complete through evaluation
[ ] REVISIONS REQUESTED (See Comments Below)

Faculty Signature(s) : __________________________________________
Faculty Name(s)      : __________________________________________
Comments / Revisions : 
________________________________________________________________________________________
========================================================================================
```

---

## 13. Phase 1 Results Summary

| Metric | Logistic Regression | Random Forest | Majority Baseline |
| :--- | :---: | :---: | :---: |
| **Test macro-F1** | 0.3669 | **0.3925** | 0.1868 |
| **Test accuracy** | 0.5611 | 0.5901 | **0.5963** |
| **CV f1_macro (mean ± std)** | 0.3717 ± 0.0456 | 0.3690 ± 0.0434 | — |
| **Val f1_macro** | 0.3418 | **0.3778** | 0.1979 |

**Primary Model:** Random Forest (selected via tie-break rule)

**Conclusion:** Both models beat the majority baseline on macro-F1 but lose on accuracy. The macro-F1 difference (0.0256) comes mostly from the Poor class (10 test rows). The two models are not clearly different.

See full report: [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md)
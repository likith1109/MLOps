# Dataset Specification: AI-Based Interview Preparation Assistant

> **Document Type:** Academic Dataset Specification & ML Formulation  
> **Course:** AI252TA – Machine Learning Operations | Semester V  
> **Repository:** [https://github.com/Tejzraj/MLOps.git](https://github.com/Tejzraj/MLOps.git)  
> **Status:** ADOPTED (Phase 1 evaluation done; faculty review pending)  
> **Current Version:** `v1.0.0`  
> **Last Updated:** 2026-10-15  

---

## 1. Project Context

The **AI-Based Interview Preparation Assistant** is an end-to-end Machine Learning Operations system designed to facilitate structured mock interview simulations for students and job seekers. The complete product will conduct an interactive interview session across diverse engineering domains and difficulty levels, record candidate responses, evaluate answer quality across multiple pedagogical dimensions, and generate a comprehensive feedback report.

Within the overall system architecture, an explicit separation of concerns is maintained:
- **Application Layer:** Manages session state, interview flow, question selection, follow-up generation, and final report compilation.
- **ML / MLOps Layer:** Strictly focused on a tractable, defensible ML task: **evaluating the quality of a candidate's textual response to an interview question against domain expectations.**

---

## 2. Problem Definition

In traditional interview preparation, candidates receive subjective or infrequent feedback due to the scarcity of experienced human evaluators. Automated assessment requires an ML model capable of distinguishing between superficial, partially accurate, and conceptually thorough technical responses.

The challenge is formulated as:
$$\text{Quality} = f(\text{Question}, \text{Candidate Answer}, \text{Reference Answer}, \text{Expected Concepts}, \text{Domain}, \text{Difficulty})$$

The ML model must compute semantic relevance, conceptual completeness, and factual correctness to categorize the answer into a standardized quality tier.

---

## 3. ML Objective

The primary Phase-1 machine learning objective is **Supervised Multi-Class Classification**.

- **Target Task:** Classify a candidate's answer into one of four ordinal quality tiers:
  - `0`: Poor
  - `1`: Needs Improvement
  - `2`: Good
  - `3`: Excellent

---

## 4. Input Definition & Artifact Roles

The Phase-1 model accepts structured textual inputs from the Mohler ASAG dataset:

| Input Component | Type | Functional Role in Pipeline | Example |
| :--- | :--- | :--- | :--- |
| `question` | `string` | The technical prompt presented to the candidate. | `"What is the difference between a stack and a queue?"` |
| `candidate_answer` | `string` | The verbatim textual answer submitted by the candidate for evaluation. | `"A stack is LIFO and a queue is FIFO."` |
| `reference_answer` | `string` | **Expert reference answer** from the dataset, used to anchor evaluation by detailing expected depth. | `"A stack follows Last-In-First-Out principle... A queue follows First-In-First-Out..."` |

### Critical Conceptual Distinctions
To preserve scientific rigor, the pipeline strictly distinguishes three separate entities:
1. **Candidate Answer (`candidate_answer`):** The unverified text response submitted for assessment.
2. **Reference Answer (`reference_answer`):** An expert reference answer from the dataset used to assist evaluation models. **The reference answer is an evaluation support tool; it is NOT automatically ground truth for candidate quality.**
3. **Annotated Quality Label (`quality_label`):** The ground-truth rating derived from the dataset's human grader scores (`score_avg`, mean of two graders) using thresholds defined in `params.yaml`.

### Data Modality Justification (Text-First Architecture)
* **Phase 1 Modality:** **Text-Only**.
* **Rationale:** Interview speech processing introduces acoustic noise, non-verbal disfluency models, accent biases, and automatic speech recognition (ASR) errors. By focusing strictly on text evaluation in Phase 1, the core MLOps lifecycle (reproducible feature engineering, model registry, drift detection, and evaluation metrics) is thoroughly established without confounding errors from secondary speech-to-text models.
* **Future Extension Roadmap:** In Phase 2+, Audio will be ingested via a decoupled ASR module ($\text{Audio} \rightarrow \text{Whisper ASR} \rightarrow \text{Transcript Text}$) feeding directly into this exact evaluation pipeline.

---

## 5. Output Definition

The primary output is an ordinal classification label:

$$\hat{y} \in \{0, 1, 2, 3\}$$

Along with class prediction probabilities:
$$P(Y = c \mid X), \quad c \in \{0, 1, 2, 3\}$$

---

## 6. Dataset: Mohler Short-Answer Grading (ASAG)

**Source:** Hugging Face dataset `nkazi/MohlerASAG`, subset `cleaned`, split `open_ended`.

**Description:** 2,273 candidate answers across 81 questions from a single introductory data-structures course at the University of North Texas. Each answer has scores from two human graders (0–5 scale) plus the average (`score_avg`).

**License:** CC BY 4.0  
**Citation:** Mohler, M., Bunescu, R., & Mihalcea, R. (2011). Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments. *Proceedings of ACL-HLT 2011*.

**Why this dataset replaces the synthetic data plan:** The Mohler corpus provides authentic student answers with real human grader scores, enabling a defensible ML task with real-world noise and ambiguity. The earlier plan for synthetic interview data across 10 domains with 4 LLM-generated tiers was replaced because it would have required fabricating both answers and labels, introducing uncontrolled bias.

**Note:** The `close_ended` split of Mohler ASAG is not used because its questions have constrained answer formats that differ from the open-ended grading task.

---

## 7. Dataset Schema

The dataset is maintained in Apache Parquet format (with CSV export) at `data/raw/interview_data.parquet`.

| Field Name | Data Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `question_id` | `string` | No | Unique identifier for the prompt (e.g., `Q_001`). |
| `question` | `string` | No | Full question text prompt. |
| `reference_answer` | `string` | No | Expert reference answer from the dataset. |
| `candidate_answer` | `string` | No | Response submitted for evaluation. |
| `score_grader1` | `float64` | No | First human grader score (0–5). |
| `score_grader2` | `float64` | No | Second human grader score (0–5). |
| `score_avg` | `float64` | No | Mean of the two grader scores (0–5). |
| `quality_label` | `int64` | No | Target class label: `0`, `1`, `2`, or `3` (derived from `score_avg`). |

---

## 8. Label Mapping

Labels are derived from the dataset's `score_avg` (mean of two graders, range 0–5) using thresholds from `params.yaml`:

| quality_label (int) | Class Name | score_avg Range |
| :---: | :--- | :--- |
| 0 | Poor | [0.0, 2.0) |
| 1 | Needs Improvement | [2.0, 3.5) |
| 2 | Good | [3.5, 4.5) |
| 3 | Excellent | [4.5, 5.0] |

Class distribution in the full dataset (2,273 rows):

| Class | Count | Percentage |
| :--- | ---: | ---: |
| Poor | 80 | 3.52% |
| Needs Improvement | 370 | 16.28% |
| Good | 396 | 17.42% |
| Excellent | 1,427 | 62.78% |

---

## 9. Grader Agreement (from `reports/annotation_agreement.json`)

Agreement between the two human graders (computed on all 2,273 rows):

| Metric | Value |
| :--- | ---: |
| Quadratic-weighted kappa | 0.4875 |
| Percent agreement | 58.69% |
| Cohen's kappa | 0.2988 |
| Pearson r | 0.5832 |

**Note:** 89 rows have an ambiguous grader scale (one grader used 0–5, the other 1–5); these affect the agreement statistics above but do not affect the labels used for modeling, which come from `score_avg` after the dataset's own cleaning.

---

## 10. Splits

Data is split by `question_id` using grouped splitting to ensure no question appears in more than one split. This prevents leakage: answers to the same question share the reference answer and topical vocabulary.

| Split | Rows | Questions | Poor | Needs Imp. | Good | Excellent |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,561 | 56 | 62 | 233 | 277 | 989 |
| Val | 229 | 8 | 8 | 38 | 33 | 150 |
| Test | 483 | 17 | 10 | 99 | 86 | 288 |

Cross-split text overlap (lowercased, stripped `candidate_answer`):
- Unique strings shared train & val: 2
- Unique strings shared train & test: 6
- Unique strings shared val & test: 1
- Test rows whose answer text also appears in train: 19 (3.93%)

The Poor class has only 10 test rows, making per-class estimates unstable.

---

## 11. Features

Features are fitted on **train only** and applied to val/test:

- **TF-IDF**: 5,000 max features, n-gram range (1,2), English stop words, min_df=2, sublinear_tf=true
- **Length features** (5): candidate answer character count, word count, sentence count, avg word length, avg sentence length
- **Reference similarity** (1): cosine similarity between candidate and reference answer TF-IDF vectors
- **Total feature count**: 2,824

---

## 12. Models

| Model | Key Hyperparameters (from `params.yaml`) |
| :--- | :--- |
| Logistic Regression | C=1.0, solver=lbfgs, max_iter=1000, class_weight=balanced |
| Random Forest | n_estimators=300, max_depth=None, class_weight=balanced |

Both use `class_weight="balanced"`. CV uses grouped 5-fold (by `question_id`) on train only, with a fresh feature pipeline per fold.

---

## 13. Data Quality Notes

- **Duplicates in raw data:** 143 rows have a `candidate_answer` that appears more than once (45 unique answers); 98 redundant copies (keep=first). 105 fully duplicated rows (all columns); 68 fully redundant copies. 1 `(question_id, candidate_answer)` group has more than one `quality_label`.
- **Cross-split text overlap:** 19 test rows (3.93%) share answer text with train; 6 unique answer strings overlap train/test. This may inflate test performance slightly.
- Nothing was changed after seeing test results.

---

## 14. Limitations

1. **Single course**: Only one introductory data-structures course (UNT); generalization unknown.
2. **Skewed labels**: Excellent class is 62.8% of data; Poor is 3.5%.
3. **Tiny Poor test class**: Only 10 test rows → unstable per-class estimates.
4. **Few questions**: 8 validation and 17 test questions → high variance in grouped CV.
5. **Thresholds are a design choice**: The four bins (Poor <2.0, Needs Improvement <3.5, Good <4.5, Excellent ≥4.5) are a design choice, fixed in params.yaml before training.
6. **Moderate grader agreement**: Quadratic-weighted kappa 0.4875; labels have noise.
7. **High variance across questions**: CV f1_macro std ~0.045.
8. **Duplicates and cross-split text overlap** (see §13).
9. **DVC remote is local-only**: Data is not on GitHub; fresh clone requires re-downloading from Hugging Face.
10. **Faculty notification**: Faculty have not yet been informed of the dataset change; this is pending.

---

## 15. Versioning Strategy

- **Semantic Data Versioning:** Datasets adhere to `vMAJOR.MINOR.PATCH`:
  - `v1.0.0`: Mohler ASAG `open_ended` split adopted as Phase 1 dataset.
- **Change Log:** Every modification to schema, rubric, or sample rows triggers a version bump and DVC commit.

---

## 16. DVC Integration

- Raw dataset stored in `data/raw/interview_data.parquet` and `data/raw/interview_data.csv`.
- Tracked via DVC with `dvc.lock` committed to Git.
- DVC remote is local (`~/dvc_storage`); teammates must run `python src/data/download_mohler.py` then `dvc repro` because `dvc pull` will not work for them.
- Pipeline stages: `prepare_dataset` → `load_data` → `preprocess` → `feature_engineering` → `train_baseline` → `train_compare` → `evaluate`.

---

## 17. Faculty Review & Acceptance Criteria

The Phase-1 dataset was accepted into the training pipeline based on:
- [x] Schema adheres to the specified fields (including `question_id`, `question`, `reference_answer`, `candidate_answer`, `score_grader1`, `score_grader2`, `score_avg`, `quality_label`).
- [x] Grouped splitting verified with zero overlapping `question_id` instances between train and test.
- [x] DVC tracking verified with valid Git pointer metadata (`dvc.lock`).
- [ ] Faculty committee review and formal approval (pending)

See full Phase 1 report: [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md)
# Dataset Labeling & Annotation Guidelines

> **Project:** AI-Based Interview Preparation Assistant  
> **Course:** AI252TA – Machine Learning Operations | Semester V  
> **Document Type:** Standardized Annotation Protocol & Rubric  
> **Status:** REFERENCE — Mohler ASAG Human Grader Scores Used (No New Annotation Performed)  
> **Version:** `v1.0.0`  
> **Last Updated:** 2026-10-15  

---

## 1. Purpose & Guiding Principles

This document establishes the scoring protocol that **would have been used** to label candidate interview responses if the project had proceeded with synthetic data creation and human annotation as originally planned.

**Actual Phase 1 Approach:** The project adopted the **Mohler Short-Answer Grading (ASAG)** corpus, which already contains scores from two human graders (`score_grader1`, `score_grader2`) on a 0–5 scale for each candidate answer. Quality labels were derived from the mean score (`score_avg`) using thresholds defined in `params.yaml`. No new human annotation was performed.

> [!IMPORTANT]
> **Core Annotation Rules (Original Plan):**
> 1. **Rubric-Driven Ground Truth:** Ground-truth quality labels must strictly originate from human application of a rubric, **not** from an LLM-generated judgment or automated prompt rating.
> 2. **Reference Answer Role:** The reference answer is an **expert-reviewed reference answer** providing a baseline benchmark of required depth. It is an evaluation aid, **not** automatically ground truth for candidate quality.
> 3. **Proposal Disclaimer:** The scoring definitions and threshold intervals below represent the project's **original proposed methodology** and are **SUBJECT TO FACULTY VALIDATION** and **SUBJECT TO ANNOTATION CALIBRATION** before production labeling.

---

## 2. Multi-Dimensional Evaluation Framework (Original Proposed Rubric)

Every candidate response was to be scored across three independent, orthogonal dimensions ($0 - 5$ points each):

$$\text{Composite Score} = S_{\text{correctness}} + S_{\text{completeness}} + S_{\text{communication}} \quad (0 \le \text{Total} \le 15)$$

```
                                       TOTAL SCORE (0 – 15)
                                                 │
                 ┌───────────────────────────────┼───────────────────────────────┐
                 ▼                               ▼                               ▼
        Correctness (0 – 5)             Completeness (0 – 5)            Communication (0 – 5)
     Factual Truth & Validity        Key Concepts & Trade-offs       Clarity, Structure & Delivery
```

---

## 3. Dimensional Scoring Rubrics (Original Proposed)

### Dimension A: Correctness ($0 - 5$)

Evaluates whether the statements made in the answer are factually accurate, sound, and free of misconceptions.

| Score | Rating | Technical Definition | Example Criteria |
| :---: | :--- | :--- | :--- |
| **0** | **Completely Incorrect / Irrelevant** | The response is factually false, misleading, or completely off-topic. | Stating that UDP provides guaranteed packet delivery. |
| **1** | **Minimal Correct Information** | Contains one accurate keyword or fact, but is dominated by gross misconceptions. | Confusing processes with virtual machines while mentioning memory. |
| **2** | **Partially Correct** | States fundamental truths but contains notable factual errors or invalid assumptions. | Correctly describes hashing, but claims hash collisions cannot occur. |
| **3** | **Mostly Correct** | Core technical mechanism is correct; minor inaccuracies in secondary details. | Accurately explains ACID properties, but slightly mischaracterizes Isolation levels. |
| **4** | **Highly Correct** | Fully accurate across all primary points; negligible technical ambiguity. | Accurate explanation of process memory layout with minor syntax omission. |
| **5** | **Fully Correct** | Fully accurate with no identified substantive factual errors, rigorous, and uses domain terminology accurately. | Accurate explanation of memory isolation, MMU, page tables, and context switching with no identified substantive errors. |

---

### Dimension B: Completeness ($0 - 5$)

Evaluates how thoroughly the candidate addressed the prompt, whether expected concepts were covered, and if relevant trade-offs were discussed.

| Score | Rating | Technical Definition | Example Criteria |
| :---: | :--- | :--- | :--- |
| **0** | **No Meaningful Answer** | The response is blank, a refusal to answer, or completely misses the question. | *"I don't know"*, *"Pass"*, or under 5 irrelevant words. |
| **1** | **Very Incomplete** | Mentions only a single superficial point; ignores all primary requirements. | Mentions that threads are faster, but fails to explain why or how. |
| **2** | **Covers Limited Concepts** | Touches on roughly $30 - 50\%$ of `expected_concepts`; omits major structural elements. | Explains TCP reliability (ACKs) but omits flow control, congestion, or handshakes. |
| **3** | **Covers Major Concepts** | Addresses primary mechanisms ($> 70\%$ of expected concepts), but omits edge cases or trade-offs. | Explains TCP handshake and reliability, but omits congestion control mechanisms. |
| **4** | **Nearly Complete** | Covers almost all primary and secondary concepts; leaves out only minor edge cases. | Thorough comparison of TCP vs UDP including handshakes, headers, and use cases. |
| **5** | **Comprehensive** | Exhaustive coverage; covers primary concepts, underlying mechanisms, edge cases, and architectural trade-offs. | Complete comparison including headers, windowing, congestion algorithms, and QUIC/HTTP3 evolutions. |

---

### Dimension C: Communication Clarity & Structure ($0 - 5$)

Evaluates how well the response is organized, whether thoughts flow logically, and whether the answer is professionally delivered.

| Score | Rating | Technical Definition | Example Criteria |
| :---: | :--- | :--- | :--- |
| **0** | **Unintelligible** | Word salad, broken fragments, or completely indecipherable text. | Random sequences of technical buzzwords without syntactic structure. |
| **1** | **Very Unclear** | Disorganized, rambling, extreme misapplication of terminology; very difficult to follow. | Highly disjointed sentences that obscure the candidate's actual thought process. |
| **2** | **Understandable but Poorly Structured** | The point can be deciphered, but requires effort due to run-on thoughts or poor flow. | A single unbroken paragraph mixing definitions, pros, cons, and unrelated notes. |
| **3** | **Reasonably Clear** | Comprehensible, standard structure (e.g., introduction then explanation); acceptable phrasing. | Clear sentences, readable pacing, standard technical terms used properly. |
| **4** | **Clear & Well-Structured** | Logical flow, good use of bulleted or sequential reasoning, clear transitions. | Uses structured formatting (e.g., *"First... Second... In contrast..."*). |
| **5** | **Concise, Precise, Professional** | Exemplary interview delivery: articulate, highly concise, zero fluff, logical precision. | Professional, structured breakdown directly answering the prompt with maximum signal-to-noise ratio. |

---

## 4. Overall Quality Classification Mapping (Original Proposed)

> [!WARNING]
> **Status of Score Thresholds:** The quality tier mapping below is the **original proposed methodology** for the synthetic data plan. It was **NOT USED** in Phase 1. The Mohler dataset uses a different thresholding scheme on `score_avg` (see below).

| Total Score (Original Proposed) | Class ID | Quality Tier | Practical Interpretation |
| :---: | :---: | :--- | :--- |
| **0 – 3** | **0** | **Poor** | Severe lack of knowledge; unacceptable for an interview candidate. |
| **4 – 7** | **1** | **Needs Improvement** | Partial awareness, but lacks depth; requires substantial preparation. |
| **8 – 11** | **2** | **Good** | Competent candidate; meets standard industry expectations for the role. |
| **12 – 15** | **3** | **Excellent** | Standout performance; demonstrates mastery, trade-offs, and communication polish. |

---

## 5. Actual Phase 1 Label Derivation (Mohler ASAG)

**No new annotation was performed.** Labels were derived from the dataset's existing human grader scores:

1. **Source Scores:** Two human graders scored each candidate answer on a 0–5 scale (`score_grader1`, `score_grader2`).
2. **Average Score:** `score_avg = (score_grader1 + score_grader2) / 2` (range 0.0–5.0).
3. **Threshold Mapping (from `params.yaml`):**

| quality_label | Class Name | score_avg Range |
| :---: | :--- | :--- |
| 0 | Poor | [0.0, 2.0) |
| 1 | Needs Improvement | [2.0, 3.5) |
| 2 | Good | [3.5, 4.5) |
| 3 | Excellent | [4.5, 5.0] |

4. **Resulting Distribution (2,273 rows):**

| Class | Count | Percentage |
| :--- | ---: | ---: |
| Poor | 80 | 3.52% |
| Needs Improvement | 370 | 16.28% |
| Good | 396 | 17.42% |
| Excellent | 1,427 | 62.78% |

---

## 6. Grader Agreement (Mohler Dataset)

Agreement between the two original human graders (from `reports/annotation_agreement.json`):

| Metric | Value |
| :--- | ---: |
| Quadratic-weighted kappa | 0.4875 |
| Percent agreement | 58.69% |
| Cohen's kappa | 0.2988 |
| Pearson r | 0.5832 |

**Note:** 89 rows have an ambiguous grader scale (one grader used 0–5, the other 1–5); these affect the agreement statistics above but do not affect the labels used for modeling, which come from `score_avg` after the dataset's own cleaning.

---

## 7. Edge-Case Handling Protocols (Original Proposed)

Annotators would have adhered to the following rules when encountering challenging response types:

### A. Irrelevant or Off-Topic Answers
- If a candidate answers a completely different question or responds with pleasantries:
  - Correctness = `0`, Completeness = `0`, Communication = $\le 2$.
  - Quality Class = `0 (Poor)`.

### B. Extremely Short Answers ($\le 5$ words)
- Unless the question is a direct binary question, one-line answers cannot demonstrate depth:
  - Max Completeness = `1`.
  - Quality Class = `0` or `1`.

### C. Technically Correct but Poorly Communicated Answers
- If an answer is factually correct (Correctness $= 5$, Completeness $= 4$), but poorly structured (Communication $= 2$):
  - Total Score = $5 + 4 + 2 = 11 \rightarrow$ Class `2 (Good)`.
  - **Rule:** Do NOT artificially depress Correctness due to weak formatting. Maintain separation across dimensions.

### D. Multiple Valid Architectural Approaches
- Engineering problems often have multiple valid solutions (e.g., SQL vs. NoSQL, microservices vs. monolith):
  - If the candidate chooses an unorthodox but defensible approach and articulates sound trade-offs, award full Correctness points.
  - Do not penalize candidates simply because their response differs from the reference answer's preferred implementation.

### E. Domain-Specific Terminology & Acronyms
- Standard industry acronyms (e.g., `OOM`, `Deadlock`, `ACID`, `CAP theorem`, `LRU`) do not require re-definition unless specifically requested by the prompt.

### F. Spelling & Minor Grammatical Errors
- **Principle:** We are evaluating computer science knowledge, not literary perfection.
- Minor typos or non-native phrasing that does **not** alter the technical meaning must **not** penalize Correctness or Completeness, and should carry at most a 1-point reduction on Communication if extreme.

---

## 8. Practical Human Validation Strategy (Original Proposed Plan)

To ensure labeling consistency and academic rigor within a college project setting:
1. **Student Team Annotators:** Scoring would be conducted by student project members using the explicit rubrics above.
2. **Anonymous Annotator Tokens:** Annotators identified strictly by anonymous keys (`annotator_id` = `ANN_01`, `ANN_02`). Zero student names, USNs, emails, or personal identifiers stored.
3. **Dual-Audit Review:** A random $15\% - 20\%$ subset would be independently evaluated by two different team members without viewing each other's scores.
4. **Dispute Resolution Protocol:**
   - Any score variance $\ge 2$ points on any dimension, or any variance that shifts the final quality class, marked as a dispute.
   - Disputed items re-examined by both annotators alongside the reference answer and resolved via rubric consensus.
5. **Consistency Measurement:** Inter-annotator agreement metrics (Cohen's Kappa, percent agreement) computed and logged **only after** the annotation process is executed. No fabricated consistency numbers claimed in advance.
6. **No False Role Claims:** Faculty members recognized as project guides and evaluators; not claimed as annotators unless they actively participate in labeling.

---

## 9. Use of AI-Assisted Synthetic Responses (Original Proposed Plan)

> [!CAUTION]
> **Handling AI-Assisted Synthetic Text (Original Plan):**
> - If AI tools were used to assist in generating candidate responses across quality tiers, they would serve **strictly as a controlled data-generation aid**, subject to faculty review.
> - **Synthetic candidate answers are NOT real student responses.** They must never be represented as empirical human interview behavior.
> - **Synthetic answers must NOT be treated as ground truth.** The prompt used to generate a response does not dictate its label. Every synthetic response must be independently inspected, scored, and validated by human annotators using this rubric.
> - LLM self-grading without human annotation is strictly forbidden.

**Actual Phase 1:** No synthetic responses were generated. The Mohler dataset provides authentic student answers.

---

## 10. Dataset Limitations (Actual Phase 1)

1. **Single course**: Only one introductory data-structures course (UNT); generalization unknown.
2. **Skewed labels**: Excellent class is 62.8% of data; Poor is 3.5%.
3. **Tiny Poor test class**: Only 10 test rows → unstable per-class estimates.
4. **Few questions**: 8 validation and 17 test questions → high variance in grouped CV.
5. **Thresholds are a design choice**: The four bins (Poor <2.0, Needs Improvement <3.5, Good <4.5, Excellent ≥4.5) are a design choice, fixed in params.yaml before training.
6. **Moderate grader agreement**: Quadratic-weighted kappa 0.4875; labels have noise.
7. **High variance across questions**: CV f1_macro std ~0.045.
8. **Duplicates and cross-split text overlap** in raw data.
9. **DVC remote is local-only**: Data is not on GitHub; fresh clone requires re-downloading from Hugging Face.
10. **Faculty notification**: Faculty have not yet been informed of the dataset change; this is pending.

---

## 11. Summary

| Aspect | Original Plan | Actual Phase 1 |
| :--- | :--- | :--- |
| **Data Source** | Synthetic generation across 10 domains | Mohler ASAG (1 domain, real students) |
| **Annotation** | Student team rubric-based (15-point) | Pre-existing human grader scores (0–5 each) |
| **Label Derivation** | 3-dimension rubric → 0–15 → 4 tiers | `score_avg` (0–5) → 4 tiers via thresholds |
| **Grader Agreement** | To be measured (dual-audit) | QWK = 0.4875 (two original graders) |
| **Class Balance** | Target 25% ± 3% per tier | Highly skewed (Excellent 62.8%) |
| **Faculty Approval** | Required before annotation | Dossier prepared; review pending |

See full Phase 1 report: [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md)
# EcoStats-500 Benchmark Validation — Annotation Guidelines

## 1. Task Overview

You are validating the quality of the EcoStats-500 benchmark, a set of question-answer pairs automatically generated from a knowledge graph. Each item consists of an LLM-generated question targeting a specific KG triple, paired with a gold-standard numerical answer extracted from that triple.

**Goal:** Verify that (a) the question is well-formed and unambiguous, and (b) the gold answer correctly answers the question based on the source triple.

**Sample size:** 100 items (randomly sampled from 500)  
**Annotators required:** 2 (independent, then reconcile disagreements)  
**Target agreement:** Cohen's κ ≥ 0.75  
**Estimated time:** ~1.5–2 hours per annotator

---

## 2. Item Structure

Each benchmark item has the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `sample_id` | Unique identifier (B001–B100) | B042 |
| `question` | LLM-generated natural language question | "What is the annual capacity of the plant in US gallons?" |
| `gold_answer` | The expected numerical answer | "75" |
| `source_triple_subject` | Subject of the source KG triple | "Plant" |
| `source_triple_relation` | Relation type | "HasCapacity" |
| `source_triple_object` | Object of the source KG triple | "75 million US gallons per year" |
| `source_topic` | Wikipedia article the triple came from | "Hydrotreated vegetable oil" |

---

## 3. Evaluation Dimensions

You will evaluate each item on **two** dimensions:

### Dimension A: Question Quality

Rate how well the question is formed:

| Score | Label | Criteria |
|-------|-------|----------|
| **3** | Good | Clear, specific, unambiguous. A knowledgeable person could answer it. The question naturally targets the gold answer. |
| **2** | Acceptable | Understandable but has minor issues: slightly vague, could have multiple valid interpretations, or phrasing is awkward. |
| **1** | Poor | Ambiguous, misleading, unanswerable, or does not match the source triple. The question could reasonably lead to a different answer. |

**Common question issues to watch for:**
- **Too vague:** "What is the capacity?" (of what? where? when?)
- **Wrong framing:** Question asks for a percentage but gold answer is an absolute number
- **Missing context:** Question lacks temporal or geographic scope that the triple specifies
- **Leading/biased:** Question presupposes facts not in the triple
- **Trivially answerable:** "What is 75 million US gallons?" (just restates the answer)

### Dimension B: Answer Correctness

Rate whether the gold answer correctly answers the question:

| Score | Label | Criteria |
|-------|-------|----------|
| **3** | Correct | The gold answer directly and completely answers the question. No important information is missing. |
| **2** | Partially Correct | The gold answer is numerically correct but incomplete (e.g., missing units, missing context) or the question requires additional numbers not provided. |
| **1** | Incorrect | The gold answer does not correctly answer the question, or the answer is for a different entity/time period than what the question asks. |

---

## 4. Annotation Procedure

### Step-by-step for each item:

1. **Read the question** carefully
2. **Read the source triple** (subject, relation, object)
3. **Evaluate question quality:**
   - Is the question clear and specific?
   - Does it naturally target the gold answer?
   - Could it be misunderstood?
4. **Evaluate answer correctness:**
   - Does the gold answer actually answer the question?
   - Is the numerical value appropriate?
   - Is important context missing?
5. **Assign scores** for both dimensions
6. **Add notes** for any score below 3

### Rules:
- Judge based on the question-answer pair and the source triple
- Each item must be scored independently
- Do NOT discuss scores with the other annotator until both have finished
- If unsure between two scores, add a note explaining your reasoning

---

## 5. Annotation Sheet

Use the file: `benchmark_iaa_annotation_sheet.csv`

Fill in columns:
- **question_quality**: Enter 1, 2, or 3
- **answer_correctness**: Enter 1, 2, or 3
- **notes**: Required for any score below 3; optional for score 3

Save your completed sheet as: `benchmark_iaa_annotation_[your_name].csv`

---

## 6. Examples

### Example 1 — Both Good (3, 3)
- **Question:** "By what percentage has the global number of trees decreased since the beginning of human agriculture?"
- **Gold answer:** 46%
- **Source triple:** (Number of trees worldwide, Decreases, 46% since the start of human agriculture)
- **Question quality: 3** — Clear, specific, unambiguous
- **Answer correctness: 3** — Directly answers the question

### Example 2 — Vague Question (2, 3)
- **Question:** "What is the annual capacity of the plant in US gallons?"
- **Gold answer:** 75
- **Source triple:** (Plant, HasCapacity, 75 million US gallons per year)
- **Question quality: 2** — "The plant" is vague; which plant? But answerable from context
- **Answer correctness: 3** — Numerically correct (75 million)

### Example 3 — Incomplete Answer (3, 2)
- **Question:** "What are the minimum and maximum depths of Lake Victoria?"
- **Gold answer:** 40
- **Source triple:** (Lake Victoria depth, HasMinValue, 40 meters)
- **Question quality: 3** — Clear question
- **Answer correctness: 2** — Only provides the minimum; question asks for both min and max

### Example 4 — Mismatched (1, 1)
- **Question:** "How much energy does the solar farm produce?"
- **Gold answer:** 500
- **Source triple:** (Solar farm, HasCapacity, 500 MW capacity)
- **Question quality: 1** — Asks about energy production (MWh) but the triple is about capacity (MW)
- **Answer correctness: 1** — 500 MW capacity ≠ energy produced

---

## 7. After Annotation

1. Both annotators submit independent CSV files
2. Cohen's κ will be computed separately for each dimension:
   - κ for question quality (3-class: 1, 2, 3)
   - κ for answer correctness (3-class: 1, 2, 3)
3. Disagreements (difference ≥ 2) will be reviewed jointly
4. Final metrics reported in the paper:
   - **Question quality distribution** (% Good / Acceptable / Poor)
   - **Answer correctness distribution** (% Correct / Partial / Incorrect)
   - **Cohen's κ** for each dimension
   - **Overall benchmark validity** = % of items with question_quality ≥ 2 AND answer_correctness ≥ 2

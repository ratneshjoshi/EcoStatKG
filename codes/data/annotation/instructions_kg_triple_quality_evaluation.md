# KG Triple Quality Evaluation — Annotation Guidelines

## 1. Task Overview

You are evaluating the quality of automatically extracted knowledge graph (KG) triples. Each triple was extracted from Wikipedia text by an LLM and represents a factual claim in the environmental sustainability domain.

**Goal:** Determine whether each triple accurately represents a fact supported by its source text.

**Sample size:** 300 triples (stratified across 29 relation types)  
**Annotators required:** 2 (independent, then reconcile disagreements)  
**Target agreement:** Cohen's κ ≥ 0.75  
**Estimated time:** ~3–4 hours per annotator

---

## 2. Triple Structure

Each triple has the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `sample_id` | Unique identifier (T001–T300) | T042 |
| `subject` | The entity or concept the claim is about | "Wind power electricity supply" |
| `relation` | The type of relationship | "HasNumericValue" |
| `object` | The value or fact being asserted | "2,700 TWh in 2025" |
| `source_topic` | The Wikipedia article the text came from | "Wind power" |
| `source_sentences` | The original Wikipedia sentences used for extraction | (full text provided) |

---

## 3. Scoring Rubric

For each triple, assign **one** of the following scores:

### Score 2 — Correct
The triple is **fully supported** by the source sentences.
- The subject, relation, and object are all accurate
- The numerical value matches the source text
- The relation type correctly describes the relationship

**Example:**
- Source: *"In 2025, wind supplied about 2,700 TWh of electricity"*
- Triple: (Wind power electricity supply, HasNumericValue, 2,700 TWh in 2025)
- **Score: 2** — Exact match with source text

### Score 1 — Partially Correct
The triple is **partially supported** but has minor issues:
- The numerical value is correct but the subject is slightly imprecise or overly specific
- The relation type is debatable but not clearly wrong
- Minor paraphrasing that slightly changes meaning
- Correct value but missing important context (e.g., date, location)

**Example:**
- Source: *"Solar capacity grew from 40 GW in 2010 to 710 GW in 2020"*
- Triple: (Solar capacity, HasGrowthRate, 710 GW)
- **Score: 1** — Value is in the source but presented as a growth rate rather than an absolute value; missing the baseline

### Score 0 — Incorrect
The triple is **not supported** by the source sentences:
- The numerical value does not appear in the source text
- The value is attributed to the wrong subject
- The relation type fundamentally misrepresents the claim
- The triple contradicts the source text
- The value is fabricated or hallucinated

**Example:**
- Source: *"Renewable energy accounted for 29% of global electricity in 2020"*
- Triple: (Renewable energy, HasPercentileValue, 35%)
- **Score: 0** — Wrong number; not in source

---

## 4. Annotation Procedure

### Step-by-step for each triple:

1. **Read the triple** (subject → relation → object)
2. **Read the source sentences** carefully
3. **Verify the object value** — Does this exact number/fact appear in the source?
4. **Verify the subject** — Is the claim correctly attributed?
5. **Verify the relation** — Does the relation type accurately describe the relationship?
6. **Assign a score** (0, 1, or 2)
7. **Add notes** for scores of 0 or 1 explaining the issue

### Rules:
- Judge ONLY based on the provided source sentences — do not look up external sources
- If the source sentences are ambiguous, lean toward Score 1 (partially correct)
- Each triple must be scored independently
- Do NOT discuss scores with the other annotator until both have finished

---

## 5. Annotation Sheet

Use the file: `kg_triple_annotation_sheet.csv`

Fill in columns:
- **score**: Enter 0, 1, or 2
- **notes**: Required for scores 0 and 1; optional for score 2

Save your completed sheet as: `kg_triple_annotation_[your_name].csv`

---

## 6. Common Edge Cases

| Scenario | Guideline | Score |
|----------|-----------|-------|
| Triple value is rounded vs source | If within 1% rounding, accept | 2 |
| Relation could be two types | If the chosen relation is reasonable, accept | 2 |
| Subject is more specific than source | If the claim is still accurate, accept | 2 |
| Subject is overly generic | If it loses important attribution | 1 |
| Value correct but wrong unit | Mark as partially correct | 1 |
| Value appears in source but for different entity | Mark as incorrect | 0 |
| Source text is ambiguous about the exact value | Mark as partially correct with note | 1 |
| Multiple values in source, triple picks one correctly | Accept | 2 |

---

## 7. After Annotation

1. Both annotators submit independent CSV files
2. Cohen's κ will be computed on the 3-class labels (0, 1, 2)
3. Disagreements will be reviewed jointly and resolved by discussion
4. Final metrics reported in the paper:
   - **Triple precision** = (count of Score 2 + Score 1) / 300
   - **Strict precision** = count of Score 2 / 300
   - **Cohen's κ** (inter-annotator agreement)

---

## 8. Relation Type Reference

For reference, here are the 29 relation types and their intended meanings:

| Relation | Meaning | Example Object |
|----------|---------|---------------|
| HasNumericValue | A raw numerical fact | "2,700 TWh in 2025" |
| HasPercentileValue | A percentage-based claim | "29% of global electricity" |
| HasMaxValue | An upper bound or maximum | "up to 800 GW" |
| HasMinValue | A lower bound or minimum | "at least 40 GW" |
| HasAverageValue | A mean or average | "average of 5.2 MW" |
| HasContext | Temporal or situational context | "as of 2020" |
| HasCapacity | Installed/production capacity | "500 MW capacity" |
| HasPolicyTarget | A policy goal with a deadline | "50% by 2030" |
| HasGrowthRate | Rate of increase over time | "grew 15% annually" |
| HasComparison | Comparative claim between entities | "3x more than coal" |
| HasTrend | Directional pattern over time | "increasing since 2010" |
| HasImpact | Effect or consequence with magnitude | "reduced emissions by 20%" |
| HasStatistic | General statistical claim | "statistically significant" |
| HasSource | Attribution to data source | "according to IEA" |
| HasUnitOfMeasurement | Unit specification | "measured in kWh" |
| HasAction | Policy or intervention action | "banned single-use plastics" |
| PercentageOf | Proportion of a whole | "30% of total energy" |
| Increases | Growth or rise in a metric | "increased by 200 MW" |
| Decreases | Decline in a metric | "fell by 12%" |
| Reduces | Action that lowers something | "reduces CO2 by 500 tonnes" |
| Saves | Conservation or saving of resource | "saves 1,000 litres per day" |
| VolumeOf | Quantity or volume measurement | "45 million tonnes" |
| EmissionOf | Emission quantity | "120 Mt CO2" |
| ConsumptionOf | Consumption quantity | "500 GWh per year" |
| EfficiencyOf | Efficiency metric | "92% conversion efficiency" |
| RateOf | Rate measurement | "0.5°C per decade" |
| RiskOf | Risk or probability | "25% probability of failure" |
| PolicyOf | Policy description | "carbon tax of $50/tonne" |
| MitigationOf | Mitigation measure | "offsets 10,000 tonnes CO2" |

# EcoStatKG

**A Domain-Specific Statistical Knowledge Graph and Benchmark for Evaluating Numerical Hallucination in LLMs**

---

## Overview

Large language models frequently produce **numerical hallucinations**—incorrect quantities, percentages, or statistics—yet existing benchmarks focus primarily on entity-level factuality. This repository provides:

1. **EcoStatKG** — A knowledge graph of **50,746 statistical triples** (29 relation types) and **306,231 structural triples** extracted from 6,871 Wikipedia articles on environmental sustainability.
2. **NumHallu** — A benchmark of **500 numerical questions** with verified gold answers derived from held-out KG topics.
3. **Evaluation of 8 retrieval strategies** across 6 LLMs (~170 experimental runs), with full results and analysis.

## Key Findings

| Finding | Detail |
|---------|--------|
| **Coverage > Sophistication** | Simple cosine retrieval achieves 0.989 NEM when the gold triple is in the index—matching graph-based methods |
| **Retrieval as Distractor** | When gold triples are excluded, *all* retrieval methods drop below zero-shot (0.506), as retrieved distractors override parametric knowledge |
| **Confidence Gating** | Threshold filtering at t=0.20 recovers held-out performance from 0.327 → 0.511 |

## Repository Structure

```
EcoStatKG/
├── README.md
├── codes/
│   ├── config.py                       # Central configuration (paths, API keys, model list)
│   ├── utils.py                        # Shared utilities (API client, embedding, chat)
│   ├── requirements.txt                # Python dependencies
│   │
│   ├── 00_topic_expansion.py           # Stage 0: Wikipedia category crawl + embedding filtering
│   ├── 01_corpus_extraction.py         # Stage 1: Article text + statistical sentence extraction
│   ├── 02_triple_extraction.py         # Stage 2: LLM-based statistical triple extraction
│   ├── 03_structural_extraction.py     # Stage 2b: Structural triples from Wikipedia metadata
│   ├── 04_entity_normalization.py      # Stage 3: Entity deduplication via Wikipedia redirects
│   ├── 05_embedding_indexing.py        # Stage 4: Triple embedding + ChromaDB indexing
│   ├── 06_retrieval_methods.py         # Stage 5: All 8 retrieval strategies
│   ├── 07_evaluation.py               # Stage 6: Benchmark generation + NEM/NF1/FP evaluation
│   ├── 08_ablation.py                 # Stage 7: Retrieval granularity × top-k ablation
│   │                                  # Stage 8: Human quality validation (no code; see data/annotation/)
│   ├── 10_significance.py             # Stage 9: Statistical significance tests (bootstrap, McNemar)
│   ├── 11_analysis.py                 # Stage 10: Error taxonomy + per-relation analysis
│   ├── 12_wikidata_comparison.py      # Stage 11: Wikidata baseline comparison
│   ├── 13_graph_baselines.py          # Stage 12: LightRAG + MS GraphRAG baselines
│   ├── 14_threshold_experiment.py     # Stage 13: Similarity threshold filtering
│   ├── 15_graph_heldout.py            # Stage 14: Graph baselines on held-out index
│   │
│   ├── data/
│   │   ├── triples/
│   │   │   ├── kg_triples.json                # 50,746 statistical triples (50 MB)
│   │   │   ├── structural_triples.json        # 306,231 structural triples (62 MB)
│   │   │   ├── extraction_stats.json          # Triple extraction statistics
│   │   │   └── structural_extraction_stats.json
│   │   ├── ecostats/
│   │   │   ├── ecostats_benchmark.json        # 500-question NumHallu benchmark
│   │   │   ├── held_out_topics.json           # 409 held-out topic list
│   │   │   └── benchmark_stats.json           # Benchmark statistics
│   │   ├── annotation/
│   │   │   ├── instructions_kg_triple_quality_evaluation.md
│   │   │   ├── kg_triple_items_for_annotation.json
│   │   │   ├── kg_triple_quality_human_annotations.csv
│   │   │   ├── instructions_benchmark_qa_validation.md
│   │   │   ├── benchmark_qa_items_for_annotation.json
│   │   │   ├── benchmark_qa_quality_human_annotations.csv
│   │   │   └── compute_inter_annotator_agreement.py
│   │   ├── reports/                           # Pipeline execution reports
│   │   ├── expanded_topics.txt                # 6,871 filtered topics
│   │   └── seed_topics.txt                    # Initial topic candidates
│   └── schema/
│       └── statistical_relations.json         # 29-relation schema definition
│
├── results/                                   # Experiment result summaries
│   ├── exp1_full_index/exp1_summary.json      # Full-index: 8 methods × 7 models
│   ├── exp1_main/exp1_summary.json            # Held-out: 8 methods × 7 models
│   ├── exp3_reranker/                         # Reranker ablation
│   ├── exp4_pareto/pareto_data.json           # Latency–accuracy Pareto data
│   ├── exp5_errors/                           # Error taxonomy (3 models)
│   ├── exp6_format/                           # Output format comparison
│   ├── exp7_temperature/                      # Temperature sensitivity
│   ├── exp8_wikidata/                         # Wikidata baseline comparison
│   └── ablation/                              # Granularity × top-k (3 × 5 × 3 seeds)
│
└── docs/
    └── KG_Relations_Survey.md                 # Relation schema design rationale
```

## Quick Start

### Requirements

```bash
pip install -r codes/requirements.txt
```

The pipeline uses API-accessible LLMs (no GPU required). Configure your API keys in `codes/.env`:

```env
OPENAI_API_KEY_1=your_key_here
LLM_BASE_URL=https://your-api-endpoint/v1
```

### Running the Pipeline

```bash
cd codes

# Stage 0: Expand seed categories into topics
python 00_topic_expansion.py

# Stage 1: Extract article text and statistical sentences
python 01_corpus_extraction.py

# Stage 2: Extract statistical triples via LLM
python 02_triple_extraction.py

# Stage 2b: Extract structural triples from Wikipedia metadata
python 03_structural_extraction.py

# Stage 3: Normalize and deduplicate entities
python 04_entity_normalization.py

# Stage 4: Embed triples and build ChromaDB index
python 05_embedding_indexing.py

# Stage 5–6: Run retrieval experiments and evaluate
python 06_retrieval_methods.py
python 07_evaluation.py
```

### Using Pre-built Resources

To skip the pipeline and directly use the KG and benchmark:

```python
import json

# Load the statistical KG
with open('codes/data/triples/kg_triples.json') as f:
    triples = json.load(f)
print(f"Loaded {len(triples)} statistical triples")

# Load the benchmark
with open('codes/data/ecostats/ecostats_benchmark.json') as f:
    benchmark = json.load(f)
print(f"Loaded {len(benchmark)} benchmark questions")

# Example triple
print(triples[0])
# {'subject': 'Solar energy capacity', 'relation': 'HasNumericValue',
#  'object': '3,372 GW', 'source_sentences': ['...']}

# Example benchmark question
print(benchmark[0]['question'])
print(benchmark[0]['gold_numbers'])
```

## KG Statistics

| Metric | Value |
|--------|-------|
| Statistical triples | 50,746 |
| Structural triples | 306,231 |
| Total triples | 356,977 |
| Unique subjects | 38,947 |
| Statistical relations | 29 |
| Structural relations | 3 (RelatedTo, BelongsToCategory, Synonym) |
| Source articles | 6,871 |
| Benchmark questions | 500 |
| Held-out topics | 409 |

## Evaluation Metrics

- **NEM (Numerical Exact Match):** Whether any number extracted from the prediction matches a gold number within 1% relative tolerance.
- **NF1 (Numerical F1):** Token-level F1 between predicted and gold numerical tokens.
- **FP (Factual Precision):** Fraction of predicted numbers that match gold numbers.


## License

This project uses publicly available Wikipedia content (CC BY-SA 3.0). The extraction pipeline code and benchmark are released under the MIT License.

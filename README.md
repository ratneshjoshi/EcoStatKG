# EcoStatKG

**A Domain-Specific Statistical Knowledge Graph and Benchmark for Evaluating Numerical Hallucination in LLMs**

---

## Overview

Large language models frequently produce **numerical hallucinations**—incorrect quantities, percentages, or statistics—yet existing benchmarks focus primarily on entity-level factuality. This repository provides:

1. **EcoStatKG** — A knowledge graph of **50,746 statistical triples** (29 relation types) and **306,231 structural triples** extracted from 6,871 Wikipedia articles on environmental sustainability.
2. **NumHallu** — A benchmark of **500 numerical questions** with KG-derived gold answers from held-out topics. A 100-question subset was separately checked against primary sources.
3. **Evaluation of 8 retrieval strategies** across 6 LLMs (~170 experimental runs), with full results and analysis.

## Key Findings

| Finding | Detail |
|---------|--------|
| **Oracle coverage** | In the full-index upper-bound condition, KG Cosine reaches 0.989 NEM and is statistically indistinguishable from Graph RAG (0.992) |
| **Held-out diagnostic** | When answer-bearing triples are removed by design, direct retrieval methods fall below zero-shot (0.506); CoNLI and CoVe remain near baseline |
| **Threshold filtering** | At t=0.20, KG Cosine rises from 0.327 to 0.511, near zero-shot; this is damage control, not a state-of-the-art improvement |

A bounded clinical/public-health proof of concept (5,174 triples, 120 questions, three models, six methods) reproduces the coverage and distractor patterns. Its KG is not released pending expert medical validation.

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
│   ├── run_experiments.py              # Main comparison and secondary experiments
│   ├── 08_ablation.py                 # Stage 7: Retrieval granularity × top-k ablation
│   │                                  # Stage 8: Human quality validation (no code; see data/annotation/)
│   ├── 10_significance.py             # Stage 9: Statistical significance tests (bootstrap, McNemar)
│   ├── 11_analysis.py                 # Stage 10: Error taxonomy + per-relation analysis
│   ├── 12_wikidata_comparison.py      # Stage 11: Wikidata baseline comparison
│   ├── 13_graph_baselines.py          # Stage 12: LightRAG + MS GraphRAG baselines
│   ├── 14_threshold_experiment.py     # Stage 13: Similarity threshold filtering
│   ├── 15_graph_heldout.py            # Stage 14: Graph baselines on held-out index
│   ├── t4_tolerance_sensitivity.py    # NEM tolerance sensitivity (0.5/1/2/5%)
│   ├── t5_pooled_kgcos_vs_graphrag.py # Pooled KG Cosine vs Graph RAG significance
│   ├── t7_threshold_significance.py   # Confidence-gating significance (--domain env)
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
│   ├── exp1_full_index/exp1_summary.json      # Full-index results
│   ├── exp1_full_index/significance_tests.json # Per-model paired significance tests
│   ├── exp1_main/exp1_summary.json            # Held-out results
│   ├── exp3_reranker/                         # Reranker ablation
│   ├── exp5_errors/                           # Error taxonomy (3 models)
│   ├── exp6_format/                           # Output format comparison
│   ├── exp7_temperature/                      # Temperature sensitivity
│   ├── exp8_wikidata/                         # Wikidata baseline comparison
│   ├── exp9_threshold/                        # Held-out similarity-threshold sweep
│   └── ablation/                              # Granularity × top-k (3 × 5 × 3 seeds)
│
└── docs/
    └── KG_Relations_Survey.md                 # Relation schema design rationale
```

## Quick Start

### Requirements

Use Python 3.12. Install the packages listed in `codes/requirements.txt`:

```bash
python -m pip install -r codes/requirements.txt
```

The requirements file lists package names without pinned versions. The pipeline uses API-accessible LLMs and embeddings; no local GPU is required. Create `codes/.env` from `codes/.env.example` and set the credentials and OpenAI-compatible endpoint for your provider:

```env
OPENAI_API_KEY=your_key_here
LLM_BASE_URL=https://your-api-endpoint/v1
LLM_VERIFY_SSL=true
```

Do not commit `codes/.env` or provider credentials. Multi-key variables (`OPENAI_API_KEY_1`, etc.) are optional for scripts that parallelize API calls.

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

# Stage 4: Embed triples and build the triple and text ChromaDB indices
python 05_embedding_indexing.py

# Stage 5: Run the six-method main comparison for one model
python run_experiments.py --exp 1 --mode both --model mistral-small-24b
```

The published benchmark is included in `codes/data/ecostats/ecostats_benchmark.json`; use it as-is to reproduce the paper. Regenerating it with `python 07_evaluation.py --generate-benchmark --n-samples 500` creates a new LLM-phrased sample and will not reproduce the published questions exactly.

### Reproducing the Main Evaluation

Run the six models reported in the paper from the `codes/` directory. The command resumes completed query files and makes API calls that may incur provider charges.

```bash
for model in mistral-small-24b ministral-14b qwen-27b deepseek-v4-flash glm-5 gpt-oss-120b; do
    python run_experiments.py --exp 1 --mode both --model "$model"
done
```

LightRAG and MS GraphRAG use separate runners. Build each full-index, then query all six models:

```bash
python 13_graph_baselines.py insert-lightrag
python 13_graph_baselines.py insert-graphrag

for model in mistral-small-24b ministral-14b qwen-27b deepseek-v4-flash glm-5 gpt-oss-120b; do
    python 13_graph_baselines.py query --method lightrag --model "$model"
    python 13_graph_baselines.py query --method ms_graphrag --model "$model"
done
```

Build held-out graph indices and query them separately:

```bash
python 15_graph_heldout.py build-lightrag
python 15_graph_heldout.py build-graphrag
python 15_graph_heldout.py query-all --method lightrag
python 15_graph_heldout.py query-all --method ms_graphrag
```

The held-out threshold sweep is run with `python 14_threshold_experiment.py`. Results are written below `results/`; evaluation summaries can be regenerated with `python 07_evaluation.py --evaluate PATH_TO_RESULTS.jsonl`.

### Statistical Analyses

These scripts re-score the per-query outputs written by the runs above (`results/**/*.jsonl`); they make no API calls. Raw per-query outputs are not committed, so run the evaluations first.

```bash
# NEM tolerance sensitivity (0.5%, 1%, 2%, 5%)
python t4_tolerance_sensitivity.py

# Pooled paired bootstrap + McNemar: KG Cosine vs Graph RAG (full-index and held-out)
python t5_pooled_kgcos_vs_graphrag.py

# Per-model paired significance tests against a reference method
python 10_significance.py --results-dir ../results/exp1_full_index

# Confidence-gating significance at t=0.20: recovery, gated KG Cosine vs gated Graph RAG, parity with zero-shot
python t7_threshold_significance.py --domain env
```

On Windows, set `PYTHONUTF8=1` before running these scripts so result files are read as UTF-8.

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

## Citation

```bibtex
@inproceedings{joshi2026ecostatkg,
  title={{EcoStatKG}: A Domain-Specific Statistical Knowledge Graph and Benchmark for Evaluating Numerical Hallucination in {LLMs}},
  author={Joshi, Ratnesh and Sengupta, Sagnik and Ekbal, Asif},
  booktitle={Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL-IJCNLP 2026)},
  year={2026},
  note={To appear}
}
```

## License

This project uses publicly available Wikipedia content (CC BY-SA 3.0). The extraction pipeline code and benchmark are released under the MIT License.

#!/usr/bin/env bash
# =============================================================================
# T7 cross-domain proof-of-concept driver (Clinical / Public-Health).
# Runs the IDENTICAL EcoStatKG pipeline on a second domain via ECOSTATKG_DOMAIN.
# Writes to codes/data_clinical/ and results_clinical/ — the original
# environmental data (codes/data/, results/) is never touched.
#
# Designed to run under tmux so it survives disconnects. Every stage is
# resumable (checkpoints), so re-running the script continues where it stopped.
#
# Usage (inside tmux):
#   tmux new -s t7
#   bash run_t7_clinical.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hallucination}"

export ECOSTATKG_DOMAIN=clinical

mkdir -p ../logs
LOG="../logs/t7_clinical_$(date +%Y%m%d_%H%M%S).log"
# Tee all output to a log file AND the terminal
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# T7 CLINICAL POC PIPELINE  —  start $(date)"
echo "# DOMAIN=$ECOSTATKG_DOMAIN  LOG=$LOG"
echo "############################################################"

run () {  # run <label> <cmd...>
  echo ""
  echo "===================================================================="
  echo ">>> STAGE: $1   ($(date))"
  echo "===================================================================="
  shift
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! STAGE FAILED (exit $rc) — stopping. Re-run the script to resume."
    exit $rc
  fi
}

# Stage 0 — topic expansion (Wikipedia crawl + embedding relevance filter)
# Skip if already expanded (reuse a prior run's topic list; avoids re-crawl).
if [ -f data_clinical/expanded_topics.txt ]; then
  echo ">>> STAGE 00 SKIPPED — data_clinical/expanded_topics.txt exists ($(wc -l < data_clinical/expanded_topics.txt) topics)"
else
  run "00 topic expansion" \
    python 00_topic_expansion.py --max-depth 2 --max-articles 2500 --target 800 --threshold 0.65
fi

# Stage 1 — corpus extraction (Wikipedia API; free). POC-bounded: medical
# articles are number-dense, so we cap topic count to stay deadline-safe.
if [ -f data_clinical/corpus/statistical_corpus.json ]; then
  echo ">>> STAGE 01 SKIPPED — data_clinical/corpus/statistical_corpus.json exists"
else
  run "01 corpus extraction (POC limit 600)" \
    python 01_corpus_extraction.py --resume --limit 600
fi

# Stage 2 — statistical triple extraction (LLM; API-heavy; resumable).
# POC-bounded: dense medical text yields many numeric sentences per topic.
if [ -f data_clinical/triples/kg_triples.jsonl ]; then
  echo ">>> STAGE 02 SKIPPED — data_clinical/triples/kg_triples.jsonl exists ($(wc -l < data_clinical/triples/kg_triples.jsonl) triples)"
else
  run "02 triple extraction (POC limit 500)" \
    python 02_triple_extraction.py --resume --limit 500
fi

# Stage 3 — structural triples (deterministic)
if [ -f data_clinical/triples/structural_triples.json ]; then
  echo ">>> STAGE 03 SKIPPED — structural_triples.json exists"
else
  run "03 structural extraction" \
    python 03_structural_extraction.py
fi

# Stage 4 — entity normalization (guard via chromadb marker: embed done => norm done)
if [ -f data_clinical/chromadb/chroma.sqlite3 ]; then
  echo ">>> STAGE 04 SKIPPED — chromadb exists (normalization already applied)"
else
  run "04 entity normalization" \
    python 04_entity_normalization.py
fi

# Stage 5 — embedding + ChromaDB indexing
if [ -f data_clinical/chromadb/chroma.sqlite3 ]; then
  echo ">>> STAGE 05 SKIPPED — chromadb/chroma.sqlite3 exists"
else
  run "05 embedding indexing" \
    python 05_embedding_indexing.py
fi

# Stage 7 — benchmark generation (POC size: 120 questions). GUARDED: never
# regenerate an existing benchmark (would invalidate written results).
if [ -f data_clinical/ecostats/ecostats_benchmark.json ]; then
  echo ">>> STAGE 07 SKIPPED — ecostats_benchmark.json exists ($(python3 -c "import json;print(len(json.load(open('data_clinical/ecostats/ecostats_benchmark.json'))))") questions)"
else
  run "07 benchmark generation" \
    python 07_evaluation.py --generate-benchmark --n-samples 120
fi

# Derive held-out split = the set of benchmark source topics
if [ -f data_clinical/ecostats/held_out_topics.json ]; then
  echo ">>> HELD-OUT SPLIT SKIPPED — held_out_topics.json exists"
else
  run "derive held-out split" python - <<'PY'
import json, config
b = json.load(open(config.ECOSTATS_DIR / "ecostats_benchmark.json"))
ho = sorted(set(q["source_topic"] for q in b if q.get("source_topic")))
json.dump(ho, open(config.ECOSTATS_DIR / "held_out_topics.json", "w"), indent=2)
print(f"held_out_topics.json written: {len(ho)} topics")
PY
fi

# Stage exp1 — retrieval experiments for 3 models, both conditions (resumable)
for m in qwen-27b glm-5 ministral-14b; do
  run "exp1 $m (full-index + held-out)" \
    python run_experiments.py --exp 1 --model "$m" --mode both
done

echo ""
echo "############################################################"
echo "# T7 CLINICAL POC PIPELINE  —  DONE $(date)"
echo "# KG:      codes/data_clinical/triples/kg_triples.json"
echo "# Bench:   codes/data_clinical/ecostats/ecostats_benchmark.json"
echo "# Results: results_clinical/exp1_full_index/ and exp1_main/"
echo "############################################################"

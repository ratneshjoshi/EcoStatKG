#!/usr/bin/env bash
# =============================================================================
# T7 clinical — similarity threshold filtering (confidence gating) for
# KG Cosine (hpti) and Graph RAG, held-out condition. Tests whether the paper's
# mitigation (fall back to zero-shot when no triple exceeds a cosine threshold)
# recovers clinical held-out performance, as it does for environmental.
# Bounded POC: 2 models. Runs under tmux; resumable.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hallucination}"
export ECOSTATKG_DOMAIN=clinical

mkdir -p ../logs
LOG="../logs/t7_clinical_threshold_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "#### T7 CLINICAL THRESHOLD SWEEP — start $(date) (DOMAIN=$ECOSTATKG_DOMAIN)"
echo "#### thresholds 0.15/0.20/0.25/0.30/0.35/None  x  {hpti, graph_rag}  x  {qwen-27b, glm-5}"

python 14_threshold_experiment.py --models qwen-27b glm-5

echo "#### T7 CLINICAL THRESHOLD SWEEP — DONE $(date)"
echo "#### results in results_clinical/exp9_threshold/ (or wherever the script writes)"

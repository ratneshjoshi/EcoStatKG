"""
Experiment 9: Dynamic Similarity Threshold
=============================================
Tests the effect of filtering low-similarity triples before injection.

When the KG does not cover a query's topic (held-out scenario), cosine
retrieval returns distant, irrelevant triples that can mislead the
generator.  A maximum-distance threshold discards such triples, causing
the model to fall back to parametric (zero-shot-like) generation.

Sweeps thresholds: 0.15, 0.20, 0.25, 0.30, 0.35, None (no filter).
Runs on both KG Cosine and GraphRAG* methods x all active models,
under the held-out evaluation setting.

Usage:
    python 14_threshold_experiment.py
    python 14_threshold_experiment.py --models mistral-small-24b qwen-27b
    python 14_threshold_experiment.py --quick  # 20 queries only
"""

import json
import sys
import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

from config import (
    GENERATOR_MODELS, RESULTS_DIR, ECOSTATS_DIR,
    TOP_K_DEFAULT, MAX_CALLS_PER_MINUTE,
    load_held_out_topics,
)
from utils import GenerationResult

log = logging.getLogger("ecostats.threshold")

THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, None]  # None = no filter (baseline)


def _build_held_out_filter() -> dict | None:
    topics = load_held_out_topics()
    if topics:
        return {"source_topic": {"$nin": topics}}
    return None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _run_queries(queries, run_fn, results_file, label):
    """Run queries with resume support. Same pattern as run_experiments._run_method_queries."""
    answered = set()
    clean_lines = []
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "error" not in rec:
                            answered.add(rec["query"])
                            clean_lines.append(line)
                    except (json.JSONDecodeError, KeyError):
                        pass

        total_lines = sum(1 for l in open(results_file, encoding="utf-8") if l.strip())
        if len(clean_lines) < total_lines:
            with open(results_file, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)

    remaining = [q for q in queries if q not in answered]
    if not remaining:
        print(f"  {label}: already complete ({len(answered)}/{len(queries)})")
        return

    if answered:
        print(f"  {label}: resuming — {len(answered)} done, {len(remaining)} remaining")

    call_count = 0
    window_start = time.time()

    with open(results_file, "a", encoding="utf-8") as f:
        for query in tqdm(remaining, desc=label, initial=len(answered), total=len(queries)):
            call_count += 3
            if call_count >= MAX_CALLS_PER_MINUTE:
                elapsed = time.time() - window_start
                if elapsed < 60.0:
                    time.sleep(60.0 - elapsed)
                call_count = 0
                window_start = time.time()

            try:
                result = run_fn(query)
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                log.error("%s query failed: %s — %s", label, query[:60], e)
                tqdm.write(f"  Error: {e}")
                f.write(json.dumps({"query": query, "answer": "", "error": str(e)},
                                   ensure_ascii=False) + "\n")
            f.flush()


def compute_nem(answer: str, gold_numbers: list[str]) -> int:
    for num in gold_numbers:
        if str(num) in str(answer):
            return 1
    return 0


def run_threshold_experiment(
    models: list[str] | None = None,
    quick: bool = False,
):
    # Lazy import to avoid loading chromadb at module level
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "retrieval_methods",
        Path(__file__).parent / "06_retrieval_methods.py",
    )
    baselines_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines_mod)

    # Load benchmark
    with open(ECOSTATS_DIR / "ecostats_benchmark.json") as f:
        benchmark = json.load(f)
    gold_map = {item["question"]: item["gold_numbers"] for item in benchmark}
    queries = [item["question"] for item in benchmark]
    if quick:
        queries = queries[:20]

    if models is None:
        models = [k for k in GENERATOR_MODELS if k != "devstral-24b"]

    where_filter = _build_held_out_filter()
    exp_dir = RESULTS_DIR / "exp9_threshold"
    exp_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "hpti": lambda q, mk, t: baselines_mod.kg_retrieval(
            query=q, model_key=mk,
            top_k=TOP_K_DEFAULT, granularity="fine",
            use_reranker=True, where_filter=where_filter,
            max_distance=t,
        ),
        "graph_rag": lambda q, mk, t: baselines_mod.graph_rag(
            query=q, model_key=mk,
            where_filter=where_filter,
            max_distance=t,
        ),
    }

    summary = {}

    for model_key in models:
        for method_name, method_fn in methods.items():
            for threshold in THRESHOLDS:
                t_label = f"{threshold:.2f}" if threshold is not None else "none"
                label = f"{method_name}_t{t_label}_{model_key}"
                results_file = exp_dir / f"{label}.jsonl"

                print(f"\n{'='*60}")
                print(f"Running: {label}")

                run_fn = lambda q, m=model_key, t=threshold, fn=method_fn: fn(q, m, t)
                _run_queries(queries, run_fn, results_file, label)

                # Compute NEM
                nem_scores = []
                n_triples_kept = []
                with open(results_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if "error" in rec:
                            continue
                        nem = compute_nem(rec["answer"], gold_map.get(rec["query"], []))
                        nem_scores.append(nem)
                        n_triples_kept.append(len(rec.get("retrieved_triples", [])))

                avg_nem = sum(nem_scores) / len(nem_scores) if nem_scores else 0
                avg_triples = sum(n_triples_kept) / len(n_triples_kept) if n_triples_kept else 0
                zero_triple_pct = sum(1 for n in n_triples_kept if n == 0) / len(n_triples_kept) if n_triples_kept else 0

                summary[label] = {
                    "method": method_name,
                    "model": model_key,
                    "threshold": threshold,
                    "NEM": round(avg_nem, 4),
                    "avg_triples_kept": round(avg_triples, 1),
                    "zero_triple_pct": round(zero_triple_pct, 4),
                    "n_queries": len(nem_scores),
                }
                print(f"  NEM={avg_nem:.4f}  avg_triples={avg_triples:.1f}  "
                      f"zero_triple={zero_triple_pct:.1%}")

    # Save summary
    with open(exp_dir / "exp9_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("THRESHOLD EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"{'Method':<12} {'Model':<20} {'Threshold':>10} {'NEM':>8} {'AvgTrip':>8} {'ZeroPct':>8}")
    print("-" * 68)
    for key, val in sorted(summary.items(), key=lambda x: (x[1]["method"], x[1]["model"], x[1]["threshold"] or 999)):
        t = f"{val['threshold']:.2f}" if val['threshold'] is not None else "none"
        print(f"{val['method']:<12} {val['model']:<20} {t:>10} {val['NEM']:>8.4f} "
              f"{val['avg_triples_kept']:>8.1f} {val['zero_triple_pct']:>8.1%}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 9: Dynamic Similarity Threshold")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(GENERATOR_MODELS.keys()))
    parser.add_argument("--quick", action="store_true", help="Run on 20 queries only")
    args = parser.parse_args()

    run_threshold_experiment(models=args.models, quick=args.quick)

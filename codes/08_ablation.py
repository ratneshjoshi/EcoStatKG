"""
Step 8: Granularity Ablation
==============================
Tests KG retrieval performance across granularity levels:
  - Fine:   single-triple retrieval  ("The HasGrowthRate of Solar Energy is 26.1%.")
  - Medium: condensed triple format  ("Solar Energy - HasGrowthRate: 26.1%")
  - Coarse: topic-level aggregation  ("Solar Energy: HasGrowthRate=26.1%, HasCapacity=580 GW, ...")

Also tests top_k ablation: how many triples to inject (1, 3, 5, 7, 10).

This produces the data for the "Granularity–Accuracy Curve" (Figure in paper).

Usage:
    python 08_ablation.py --model qwen-27b
    python 08_ablation.py --model all --seeds 3
"""

import json
import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm
from itertools import product

import importlib
import sys

from config import (
    GENERATOR_MODELS,
    TOP_K_ABLATION,
    ECOSTATS_DIR, RESULTS_DIR,
    NUM_SEEDS, RANDOM_SEED,
    load_held_out_topics,
)

log = logging.getLogger("ecostats.ablation")

# Import pipeline and evaluation functions
# (can't import 04/06 directly due to leading digits, use importlib)
sys.path.insert(0, str(Path(__file__).parent))

from utils import GenerationResult
from config import TOP_K_DEFAULT


def _import_module(filename: str):
    """Import a module with a numeric prefix."""
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""),
        Path(__file__).parent / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_granularity_ablation(
    model_key: str = "qwen-27b",
    seeds: int = NUM_SEEDS,
    top_k_values: list[int] | None = None,
    granularities: list[str] | None = None,
):
    """
    Run the full granularity × top_k ablation grid.
    
    Grid:
        granularity ∈ {fine, medium, coarse}
        top_k       ∈ {1, 3, 5, 7, 10}
        seeds       ∈ {1, ..., NUM_SEEDS}
    """
    pipeline_mod = _import_module("06_retrieval_methods.py")
    eval_mod = _import_module("07_evaluation.py")

    if top_k_values is None:
        top_k_values = TOP_K_ABLATION
    if granularities is None:
        granularities = ["fine", "medium", "coarse"]

    # Load benchmark
    benchmark_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    queries = [item["question"] for item in benchmark]

    # Build held-out filter
    held_out_topics = load_held_out_topics()
    held_out_filter = {"source_topic": {"$nin": held_out_topics}} if held_out_topics else None

    # Results storage
    ablation_results = []

    total_configs = len(granularities) * len(top_k_values) * seeds
    print(f"Running {total_configs} configurations × {len(queries)} queries")
    print(f"  Granularities: {granularities}")
    print(f"  Top-k values:  {top_k_values}")
    print(f"  Seeds:         {seeds}")
    print(f"  Model:         {model_key}")

    for gran, k, seed in product(granularities, top_k_values, range(seeds)):
        config_label = f"{model_key}_{gran}_k{k}_s{seed}"
        print(f"\n--- {config_label} ---")

        # Run queries
        results_file = RESULTS_DIR / "ablation" / f"{config_label}.jsonl"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Resume support: query-aware (skip successfully answered queries)
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
            # Purge error records
            raw_count = sum(1 for l in open(results_file, encoding="utf-8") if l.strip())
            if len(clean_lines) < raw_count:
                with open(results_file, "w", encoding="utf-8") as f:
                    f.writelines(clean_lines)
                log.info("%s: purged error records, kept %d", config_label, len(clean_lines))

        remaining = [q for q in queries if q not in answered]
        if not remaining:
            print(f"  Already complete ({len(answered)} queries), skipping")
        else:
            if answered:
                print(f"  Resuming: {len(answered)} done, {len(remaining)} remaining")

            with open(results_file, "a", encoding="utf-8") as f:
                for query in tqdm(remaining, desc=config_label, leave=False,
                                  initial=len(answered), total=len(queries)):
                    try:
                        result = pipeline_mod.kg_retrieval(
                            query=query,
                            model_key=model_key,
                            top_k=k,
                            granularity=gran,
                            use_reranker=True,
                            where_filter=held_out_filter,
                        )
                        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"  Error on query: {e}")
                        f.write(json.dumps({
                            "query": query,
                            "model": model_key,
                            "method": "kg_retrieval",
                            "answer": "",
                            "top_k": k,
                            "granularity": gran,
                            "error": str(e),
                        }) + "\n")
                    f.flush()

        # Evaluate
        metrics = eval_mod.evaluate_results(results_file)
        metrics["granularity"] = gran
        metrics["top_k"] = k
        metrics["seed"] = seed
        metrics["model"] = model_key
        ablation_results.append(metrics)

        eval_mod.print_metrics(metrics, label=config_label)

    # Save ablation summary
    summary_path = RESULTS_DIR / "ablation" / f"ablation_summary_{model_key}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Print aggregated results
    print_ablation_summary(ablation_results)

    return ablation_results


def print_ablation_summary(results: list[dict]):
    """Print a nicely formatted ablation summary table."""
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (averaged over seeds)")
    print("=" * 80)
    print(f"{'Gran':<8} {'k':<4} {'NEM':>8} {'NF1':>8} {'FP':>8} {'Latency':>10}")
    print("-" * 50)

    # Aggregate over seeds
    from collections import defaultdict
    agg = defaultdict(lambda: {"NEM": [], "NF1": [], "FP": [], "latency": []})

    for r in results:
        key = (r["granularity"], r["top_k"])
        agg[key]["NEM"].append(r.get("NEM", 0))
        agg[key]["NF1"].append(r.get("NF1", 0))
        agg[key]["FP"].append(r.get("FP", 0))
        agg[key]["latency"].append(r.get("latency_mean_ms", 0))

    for (gran, k), vals in sorted(agg.items()):
        nem = sum(vals["NEM"]) / len(vals["NEM"])
        nf1 = sum(vals["NF1"]) / len(vals["NF1"])
        fp = sum(vals["FP"]) / len(vals["FP"])
        lat = sum(vals["latency"]) / len(vals["latency"])
        print(f"{gran:<8} {k:<4} {nem:>8.4f} {nf1:>8.4f} {fp:>8.4f} {lat:>10.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Granularity ablation study")
    parser.add_argument("--model", type=str, default="qwen-27b",
                        choices=list(GENERATOR_MODELS.keys()) + ["all"])
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--granularities", nargs="+", default=["fine", "medium", "coarse"])
    parser.add_argument("--top-k-values", nargs="+", type=int, default=TOP_K_ABLATION)
    args = parser.parse_args()

    models = list(GENERATOR_MODELS.keys()) if args.model == "all" else [args.model]

    for model in models:
        run_granularity_ablation(
            model_key=model,
            seeds=args.seeds,
            top_k_values=args.top_k_values,
            granularities=args.granularities,
        )

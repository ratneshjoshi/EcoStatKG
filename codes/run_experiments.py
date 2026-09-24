"""
Run All Experiments
=====================
Orchestrates the full experimental pipeline:

  Experiment 1: Main comparison (all retrieval methods x models)
  Experiment 2: Granularity ablation (fine/medium/coarse x top_k sweep)
  Experiment 3: Reranker ablation (with/without reranking)
  Experiment 4: Efficiency Pareto frontier (latency vs accuracy)
  Experiment 5: Error analysis (categorize failure modes)

Usage:
    python run_experiments.py --exp 1        # Main comparison only
    python run_experiments.py --exp all       # Everything
    python run_experiments.py --exp 1 --model qwen-27b --quick  # Quick test
"""

import json
import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm
from itertools import product

from config import (
    GENERATOR_MODELS,
    ECOSTATS_DIR, RESULTS_DIR,
    TOP_K_DEFAULT, NUM_SEEDS,
    MAX_CALLS_PER_MINUTE,
    load_held_out_topics,
)
from utils import GenerationResult

log = logging.getLogger("ecostats.experiments")


def _build_held_out_filter() -> dict | None:
    """Build a ChromaDB where filter that excludes held-out benchmark topics."""
    topics = load_held_out_topics()
    if topics:
        return {"source_topic": {"$nin": topics}}
    return None


def _import_module(filename: str):
    """Import a module with a numeric prefix."""
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""),
        Path(__file__).parent / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_benchmark() -> list[dict]:
    """Load the EcoStats benchmark."""
    path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _run_method_queries(
    queries: list[str],
    run_fn,
    results_file: Path,
    label: str,
):
    """
    Run queries for a single method+model, with:
    - Resume: skips queries already answered *successfully* (by query text match).
      Error records (lines with "error" key) are NOT counted as done
      so they will be retried on the next run.
    - Rate limiting (3 API calls assumed per query)
    """
    # Load already-completed queries (skip error records)
    answered = set()
    clean_lines = []
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "error" not in rec:  # only count successes
                            answered.add(rec["query"])
                            clean_lines.append(line)
                    except (json.JSONDecodeError, KeyError):
                        pass

        # Rewrite file without error records so they don't accumulate
        if len(clean_lines) < sum(1 for l in open(results_file, encoding="utf-8") if l.strip()):
            with open(results_file, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)
            log.info("%s: purged error records, kept %d clean results", label, len(clean_lines))

    remaining = [q for q in queries if q not in answered]

    if not remaining:
        log.info("%s: already complete (%d/%d), skipping", label, len(answered), len(queries))
        print(f"  {label}: already complete ({len(answered)}/{len(queries)}), skipping")
        return

    if answered:
        log.info("%s: resuming -- %d done, %d remaining", label, len(answered), len(remaining))
        print(f"  {label}: resuming -- {len(answered)} done, {len(remaining)} remaining")

    call_count = 0
    window_start = time.time()
    done_so_far = len(answered)

    with open(results_file, "a", encoding="utf-8") as f:
        for query in tqdm(remaining, desc=label, initial=done_so_far, total=len(queries)):
            # Rate limiting (~3 API calls per query)
            call_count += 3
            if call_count >= MAX_CALLS_PER_MINUTE:
                elapsed = time.time() - window_start
                if elapsed < 60.0:
                    wait = 60.0 - elapsed
                    tqdm.write(f"  Rate limit: waiting {wait:.0f}s...")
                    time.sleep(wait)
                call_count = 0
                window_start = time.time()

            try:
                result = run_fn(query)
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                log.error("%s query failed: %s -- %s", label, query[:60], e)
                tqdm.write(f"  Error: {e}")
                # Write error record — will be purged on next resume
                f.write(json.dumps({
                    "query": query, "answer": "", "error": str(e)
                }, ensure_ascii=False) + "\n")
            f.flush()


# ============================================================
# Experiment 1: Main Comparison (Table 1 in paper)
# ============================================================

def experiment_1_main_comparison(
    models: list[str] | None = None,
    quick: bool = False,
    mode: str = "both",
):
    """
    All retrieval methods x all models.
    
    Grid: 6 methods x N models x N queries
    Methods: kg_retrieval, zero_shot, standard_rag, graph_rag, conli, cove
    Supports resumption: skips already-completed method+model files.
    
    mode: 'held_out' (filtered), 'full_index' (unfiltered), or 'both'
    """
    baselines_mod = _import_module("06_retrieval_methods.py")
    eval_mod = _import_module("07_evaluation.py")

    benchmark = load_benchmark()
    queries = [item["question"] for item in benchmark]
    if quick:
        queries = queries[:20]

    if models is None:
        models = list(GENERATOR_MODELS.keys())

    methods = ["kg_retrieval", "zero_shot", "standard_rag", "graph_rag", "conli", "cove"]

    modes_to_run = []
    if mode in ("held_out", "both"):
        modes_to_run.append("held_out")
    if mode in ("full_index", "both"):
        modes_to_run.append("full_index")

    all_results = {}  # mode -> metrics dict

    for eval_mode in modes_to_run:
        if eval_mode == "held_out":
            where_filter = _build_held_out_filter()
            exp_dir = RESULTS_DIR / "exp1_main"
            print("\n" + "#" * 70)
            print("# MODE: HELD-OUT (gold triples excluded from retrieval)")
            print("#" * 70)
        else:
            where_filter = None
            exp_dir = RESULTS_DIR / "exp1_full_index"
            print("\n" + "#" * 70)
            print("# MODE: FULL-INDEX (all triples available)")
            print("#" * 70)

        exp_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = {}

        for model_key in models:
            for method in methods:
                label = f"{method}_{model_key}"
                print(f"\n{'='*60}")
                print(f"[{eval_mode}] Running: {label}")
                results_file = exp_dir / f"{label}.jsonl"

                kwargs = {}
                if method == "kg_retrieval":
                    kwargs["top_k"] = TOP_K_DEFAULT
                    kwargs["where_filter"] = where_filter
                elif method in ("standard_rag", "conli"):
                    kwargs["top_k"] = TOP_K_DEFAULT
                    kwargs["where_filter"] = where_filter
                elif method == "graph_rag":
                    kwargs["where_filter"] = where_filter
                run_fn = lambda q, m=method, mk=model_key, kw=kwargs: \
                    baselines_mod.run_baseline(m, q, mk, **kw)

                _run_method_queries(queries, run_fn, results_file, label)

                metrics = eval_mod.evaluate_results(results_file)
                all_metrics[label] = metrics
                eval_mod.print_metrics(metrics, label=f"[{eval_mode}] {label}")

        # Save summary
        summary_path = exp_dir / "exp1_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        print_comparison_table(all_metrics, models, methods)
        all_results[eval_mode] = all_metrics

    # Print side-by-side if both modes ran
    if len(all_results) == 2:
        print_held_out_vs_full(all_results, models, methods)

    return all_results


def print_held_out_vs_full(all_results: dict, models: list[str], methods: list[str]):
    """Print side-by-side comparison of held-out vs full-index results."""
    held = all_results.get("held_out", {})
    full = all_results.get("full_index", {})
    print("\n" + "=" * 100)
    print("HELD-OUT vs FULL-INDEX COMPARISON")
    print("=" * 100)
    print(f"{'Method':<16} {'Model':<20} {'HO NEM':>8} {'FI NEM':>8} {'Δ':>8}  {'HO NF1':>8} {'FI NF1':>8} {'Δ':>8}")
    print("-" * 96)

    for model in models:
        for method in methods:
            label = f"{method}_{model}"
            h = held.get(label, {})
            f_ = full.get(label, {})
            h_nem = h.get("NEM", 0)
            f_nem = f_.get("NEM", 0)
            h_nf1 = h.get("NF1", 0)
            f_nf1 = f_.get("NF1", 0)
            d_nem = f_nem - h_nem
            d_nf1 = f_nf1 - h_nf1
            print(f"{method:<16} {model:<20} {h_nem:>8.4f} {f_nem:>8.4f} {d_nem:>+8.4f}  {h_nf1:>8.4f} {f_nf1:>8.4f} {d_nf1:>+8.4f}")
        print("-" * 96)


def print_comparison_table(metrics: dict, models: list[str], methods: list[str]):
    """Print a formatted comparison table (Table 1 style)."""
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: MAIN COMPARISON")
    print("=" * 90)
    print(f"{'Method':<16} {'Model':<16} {'NEM':>8} {'NF1':>8} {'FP':>8} {'Lat(ms)':>10} {'Tokens':>8}")
    print("-" * 74)

    for model in models:
        for method in methods:
            label = f"{method}_{model}"
            m = metrics.get(label, {})
            nem = m.get("NEM", 0)
            nf1 = m.get("NF1", 0)
            fp = m.get("FP", 0)
            lat = m.get("latency_mean_ms", 0)
            tok = m.get("tokens_mean", 0)
            print(f"{method:<16} {model:<16} {nem:>8.4f} {nf1:>8.4f} {fp:>8.4f} {lat:>10.0f} {tok:>8.0f}")
        print("-" * 74)


# ============================================================
# Experiment 2: Granularity Ablation (delegated to 07)
# ============================================================

def experiment_2_granularity(model_key: str = "qwen-27b"):
    """Run full granularity ablation (see 08_ablation.py)."""
    ablation_mod = _import_module("08_ablation.py")
    return ablation_mod.run_granularity_ablation(model_key=model_key)


# ============================================================
# Experiment 3: Reranker Ablation
# ============================================================

def experiment_3_reranker(
    model_key: str = "qwen-27b",
    quick: bool = False,
):
    """Compare KG retrieval with and without reranking."""
    baselines_mod = _import_module("06_retrieval_methods.py")
    eval_mod = _import_module("07_evaluation.py")

    benchmark = load_benchmark()
    queries = [item["question"] for item in benchmark]
    if quick:
        queries = queries[:20]

    exp_dir = RESULTS_DIR / "exp3_reranker"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for use_rerank in [True, False]:
        label = f"kg_retrieval_{'rerank' if use_rerank else 'no_rerank'}_{model_key}"
        print(f"\nRunning: {label}")
        results_file = exp_dir / f"{label}.jsonl"

        held_out_filter = _build_held_out_filter()
        run_fn = lambda q, ur=use_rerank: baselines_mod.kg_retrieval(
            query=q, model_key=model_key,
            top_k=TOP_K_DEFAULT, granularity="fine",
            use_reranker=ur,
            where_filter=held_out_filter,
        )
        _run_method_queries(queries, run_fn, results_file, label)

        metrics = eval_mod.evaluate_results(results_file)
        all_metrics[label] = metrics
        eval_mod.print_metrics(metrics, label=label)

    # Save summary
    with open(exp_dir / f"exp3_summary_{model_key}.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    return all_metrics


# ============================================================
# Experiment 4: Efficiency Pareto (Pillar 3)
# ============================================================

def experiment_4_pareto(
    models: list[str] | None = None,
    quick: bool = False,
):
    """
    Compute Pareto frontier (accuracy vs latency) from existing Exp 1 results.
    If Exp 1 results exist, computes Pareto data directly (no new API calls).
    Otherwise falls back to running queries with resume support.
    """
    eval_mod = _import_module("07_evaluation.py")

    if models is None:
        models = list(GENERATOR_MODELS.keys())

    exp1_dir = RESULTS_DIR / "exp1_main"
    exp_dir = RESULTS_DIR / "exp4_pareto"
    exp_dir.mkdir(parents=True, exist_ok=True)

    pareto_data = []
    methods = ["kg_retrieval", "zero_shot", "standard_rag", "graph_rag", "conli", "cove"]

    for model_key in models:
        for method in methods:
            label = f"{method}_{model_key}"
            # Try to load from Exp 1 results first
            exp1_file = exp1_dir / f"{label}.jsonl"
            if exp1_file.exists() and _count_lines(exp1_file) > 0:
                log.info("Exp4: loading %s from Exp 1 results", label)
                metrics = eval_mod.evaluate_results(exp1_file)
            else:
                log.info("Exp4: no Exp 1 results for %s, skipping", label)
                continue

            metrics["method"] = method
            metrics["model"] = model_key
            metrics["passes"] = 1 if method in ("kg_retrieval", "zero_shot", "standard_rag", "graph_rag") else (3 if method == "conli" else 4)
            pareto_data.append(metrics)
            eval_mod.print_metrics(metrics, label=label)

    # Save Pareto data
    pareto_path = exp_dir / "pareto_data.json"
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)

    print("\n=== PARETO SUMMARY (Accuracy vs Efficiency) ===")
    print(f"{'Method':<16} {'Model':<16} {'NEM':>8} {'Latency':>10} {'Tokens':>8} {'Passes':>6}")
    print("-" * 68)
    for p in sorted(pareto_data, key=lambda x: -x.get("NEM", 0)):
        print(f"{p['method']:<16} {p['model']:<16} {p.get('NEM',0):>8.4f} "
              f"{p.get('latency_mean_ms',0):>10.0f} {p.get('tokens_mean',0):>8.0f} {p.get('passes',1):>6}")
    
    log.info("Exp4 Pareto complete: %d data points saved to %s", len(pareto_data), pareto_path)
    return pareto_data


# ============================================================
# Experiment 5: Error Analysis
# ============================================================

def experiment_5_error_analysis(model_key: str = "qwen-27b"):
    """
    Categorize failure modes of KG retrieval:
    - Retrieval failure: correct triple not in top-k
    - Injection failure: triple retrieved but not used by model
    - Hallucination despite injection: model generates different number
    """
    eval_mod = _import_module("07_evaluation.py")

    benchmark = load_benchmark()
    gold_map = {item["question"]: item for item in benchmark}

    # Load KG retrieval results from experiment 1
    results_file = RESULTS_DIR / "exp1_main" / f"kg_retrieval_{model_key}.jsonl"
    # Fallback to old naming
    if not results_file.exists():
        results_file = RESULTS_DIR / "exp1_main" / f"hpti_{model_key}.jsonl"
    if not results_file.exists():
        print(f"Error: Run experiment 1 first. Missing: {results_file}")
        return

    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    categories = {
        "correct": [],
        "retrieval_failure": [],
        "injection_failure": [],
        "hallucination_despite_injection": [],
        "other": [],
    }

    for result in results:
        query = result["query"]
        if query not in gold_map:
            continue

        gold = gold_map[query]
        gold_nums = set(gold["gold_numbers"])
        pred_nums = set(eval_mod.extract_numbers(result["answer"]))

        # Check if answer is correct
        nem = eval_mod.numerical_exact_match(result["answer"], gold["gold_numbers"])

        if nem >= 1.0:
            categories["correct"].append(result)
            continue

        # Check retrieved triples
        retrieved = result.get("retrieved_triples", [])
        retrieved_text = " ".join(str(t) for t in retrieved)
        retrieved_nums = set(eval_mod.extract_numbers(retrieved_text))

        gold_in_retrieved = bool(gold_nums & retrieved_nums)

        if not gold_in_retrieved:
            categories["retrieval_failure"].append(result)
        elif not pred_nums:
            categories["injection_failure"].append(result)
        else:
            categories["hallucination_despite_injection"].append(result)

    # Save analysis
    exp_dir = RESULTS_DIR / "exp5_errors"
    exp_dir.mkdir(parents=True, exist_ok=True)

    analysis = {
        cat: {
            "count": len(items),
            "fraction": len(items) / len(results) if results else 0,
            "examples": [
                {"query": r["query"], "answer": r["answer"][:200]}
                for r in items[:5]
            ],
        }
        for cat, items in categories.items()
    }

    with open(exp_dir / f"error_analysis_{model_key}.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n=== ERROR ANALYSIS ({model_key}) ===")
    total = len(results)
    for cat, items in categories.items():
        pct = len(items) / total * 100 if total else 0
        print(f"  {cat:<35} {len(items):>4} ({pct:>5.1f}%)")


# ============================================================
# Experiment 6: Format Ablation (P1-2)
# ============================================================

def experiment_6_format_ablation(
    model_key: str = "qwen-27b",
    quick: bool = False,
):
    """
    Controlled format ablation: same 5 triples, different presentation.
    Isolates the effect of structured vs. prose formatting.

    Conditions:
      - structured: "[1] The HasGrowthRate of Solar Energy is 26.1%."
      - prose: Triples rewritten as a natural paragraph.
    """
    baselines_mod = _import_module("06_retrieval_methods.py")
    eval_mod = _import_module("07_evaluation.py")

    benchmark = load_benchmark()
    queries = [item["question"] for item in benchmark]
    if quick:
        queries = queries[:20]

    exp_dir = RESULTS_DIR / "exp6_format"
    exp_dir.mkdir(parents=True, exist_ok=True)

    held_out_filter = _build_held_out_filter()
    model = GENERATOR_MODELS[model_key]
    all_metrics = {}

    for fmt in ["structured", "prose"]:
        label = f"kg_retrieval_{fmt}_{model_key}"
        print(f"\nRunning: {label}")
        results_file = exp_dir / f"{label}.jsonl"

        if fmt == "structured":
            run_fn = lambda q: baselines_mod.kg_retrieval(
                query=q, model_key=model_key,
                top_k=TOP_K_DEFAULT, granularity="fine",
                where_filter=held_out_filter,
            )
        else:
            # Prose: retrieve same triples, but convert to paragraph
            def _prose_run(q):
                from utils import create_client, embed_query, chat_completion
                from config import (
                    GROUNDED_SYSTEM_PROMPT, MAX_TOKENS, TOP_P, TEMPERATURE,
                )
                client = create_client()
                import time as _time
                start = _time.time()

                query_emb = embed_query(client, q)
                retrieved = baselines_mod.retrieve_triples(
                    query_emb, "hpti_fine", TOP_K_DEFAULT,
                    where_filter=held_out_filter,
                )
                if hasattr(retrieved, 'documents') and len(retrieved.documents) > 1:
                    retrieved = baselines_mod.rerank_triples(client, q, retrieved)

                # Convert triples to prose paragraph
                docs = retrieved.documents
                prose = "According to environmental statistics, " + ". ".join(
                    d.replace("The ", "the ").rstrip(".") for d in docs
                ) + "."

                response = chat_completion(
                    client, model,
                    messages=[
                        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{prose}\n\nQuestion: {q}"},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    top_p=TOP_P,
                )
                latency = (_time.time() - start) * 1000
                from utils import GenerationResult
                return GenerationResult(
                    query=q, model=model_key, method="kg_retrieval_prose",
                    answer=response.choices[0].message.content.strip(),
                    retrieved_triples=[{"document": d} for d in docs],
                    top_k=TOP_K_DEFAULT, latency_ms=latency,
                    token_usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                )
            run_fn = _prose_run

        _run_method_queries(queries, run_fn, results_file, label)
        metrics = eval_mod.evaluate_results(results_file)
        all_metrics[label] = metrics
        eval_mod.print_metrics(metrics, label=label)

    summary_path = exp_dir / f"exp6_summary_{model_key}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


# ============================================================
# Experiment 7: Temperature Sweep (P2-2)
# ============================================================

TEMPERATURE_SWEEP = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

def experiment_7_temperature(
    model_key: str = "qwen-27b",
    quick: bool = False,
):
    """
    Temperature ablation: KG retrieval vs zero-shot at varying temperatures.
    Shows whether KG grounding remains effective as randomness increases.
    """
    baselines_mod = _import_module("06_retrieval_methods.py")
    eval_mod = _import_module("07_evaluation.py")

    benchmark = load_benchmark()
    queries = [item["question"] for item in benchmark]
    if quick:
        queries = queries[:20]

    exp_dir = RESULTS_DIR / "exp7_temperature"
    exp_dir.mkdir(parents=True, exist_ok=True)

    held_out_filter = _build_held_out_filter()
    all_metrics = {}

    for temp in TEMPERATURE_SWEEP:
        for method in ["kg_retrieval", "zero_shot"]:
            label = f"{method}_t{temp}_{model_key}"
            print(f"\nRunning: {label}")
            results_file = exp_dir / f"{label}.jsonl"

            if method == "kg_retrieval":
                run_fn = lambda q, t=temp: baselines_mod.kg_retrieval(
                    query=q, model_key=model_key,
                    top_k=TOP_K_DEFAULT, granularity="fine",
                    where_filter=held_out_filter,
                    temperature=t,
                )
            else:
                run_fn = lambda q, t=temp: baselines_mod.zero_shot(
                    query=q, model_key=model_key,
                    temperature=t,
                )

            _run_method_queries(queries, run_fn, results_file, label)
            metrics = eval_mod.evaluate_results(results_file)
            all_metrics[label] = metrics
            eval_mod.print_metrics(metrics, label=label)

    summary_path = exp_dir / f"exp7_summary_{model_key}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("TEMPERATURE SWEEP SUMMARY")
    print(f"{'='*60}")
    _delta = "\u0394"
    print(f"{'Temp':>6} {'KGRetr NEM':>10} {'ZeroShot NEM':>14} {_delta:>8}")
    print("-" * 42)
    for temp in TEMPERATURE_SWEEP:
        h_key = f"kg_retrieval_t{temp}_{model_key}"
        z_key = f"zero_shot_t{temp}_{model_key}"
        h_nem = all_metrics.get(h_key, {}).get("NEM", 0)
        z_nem = all_metrics.get(z_key, {}).get("NEM", 0)
        print(f"{temp:>6.1f} {h_nem:>10.4f} {z_nem:>14.4f} {h_nem - z_nem:>+8.4f}")

    return all_metrics


# ============================================================
# Main Dispatcher
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="Which experiment to run (1-8 or 'all')")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: test with 20 queries only")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["held_out", "full_index", "both"],
                        help="Evaluation mode: held_out, full_index, or both (default: both)")
    args = parser.parse_args()

    models = [args.model] if args.model else None
    single_model = args.model or "qwen-27b"

    experiments = args.exp.split(",") if args.exp != "all" else ["1", "2", "3", "4", "5", "6", "7"]

    for exp in experiments:
        exp = exp.strip()
        print(f"\n{'#' * 70}")
        print(f"# EXPERIMENT {exp}")
        print(f"{'#' * 70}")

        if exp == "1":
            experiment_1_main_comparison(models=models, quick=args.quick, mode=args.mode)
        elif exp == "2":
            experiment_2_granularity(model_key=single_model)
        elif exp == "3":
            experiment_3_reranker(model_key=single_model, quick=args.quick)
        elif exp == "4":
            experiment_4_pareto(models=models, quick=args.quick)
        elif exp == "5":
            experiment_5_error_analysis(model_key=single_model)
        elif exp == "6":
            experiment_6_format_ablation(model_key=single_model, quick=args.quick)
        elif exp == "7":
            experiment_7_temperature(model_key=single_model, quick=args.quick)
        else:
            print(f"Unknown experiment: {exp}")

    print("\n\nAll experiments complete. Results in:", RESULTS_DIR)

"""
Step 7: Evaluation
===================
1. Generates the evaluation benchmark from the KG (query + gold triple pairs)
2. Computes evaluation metrics:
   - Numerical Exact Match (NEM): extracted number matches gold within tolerance
   - Numerical F1 (NF1): token-level F1 on numerical tokens
   - Factual Precision (FP): fraction of numerical claims that are correct
   - Latency: wall-clock time per query (ms)
   - Tokens: total API tokens consumed per query
   - Abstention rate: % classified as Unverifiable (out of KG scope)
   - Coverage: % of claims the system can judge (Supported + Contradicted)

Usage:
    python 07_evaluation.py --generate-benchmark        # Create test benchmark
    python 07_evaluation.py --evaluate results/exp1/    # Score a results file
"""

import json
import re
import argparse
import random
import time
from pathlib import Path
from tqdm import tqdm

from openai import OpenAI

from config import (
    GENERATOR_MODELS, EXTRACTION_MODEL,
    TRIPLES_DIR, ECOSTATS_DIR, RESULTS_DIR,
    NUM_EVAL_SAMPLES, RANDOM_SEED,
    QUESTION_GENERATION_PROMPT,
    TEMPERATURE, MAX_CALLS_PER_MINUTE,
)
from utils import create_client, chat_completion


# ============================================================
# Part 1: EcoStats Benchmark Generation
# ============================================================

def load_triples() -> list[dict]:
    """Load all KG triples."""
    path = TRIPLES_DIR / "kg_triples.jsonl"
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triples.append(json.loads(line))
    return triples


def extract_numbers(text: str) -> list[str]:
    """Extract all numerical tokens from text."""
    # Matches integers, decimals, percentages, and numbers with units
    pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?'
    return re.findall(pattern, str(text))


def generate_question(
    client: OpenAI,
    triple: dict,
) -> str:
    """Use LLM to generate a natural question targeting a specific triple."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        subject=triple["subject"],
        relation=triple["relation"],
        object=triple["object"],
    )

    response = chat_completion(
        client, EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # Slightly higher for question diversity
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def generate_ecostats_benchmark(
    n_samples: int = NUM_EVAL_SAMPLES,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Generate the EcoStats benchmark: n_samples (query, gold_triple) pairs.
    
    Sampling strategy:
    - Stratified by relation type (ensure coverage of diverse relation types)
    - Filtered to triples with clear numerical objects
    """
    random.seed(seed)
    client = create_client()
    triples = load_triples()

    # Filter to triples with extractable numbers
    numerical_triples = [
        t for t in triples
        if extract_numbers(str(t["object"]))
    ]

    if len(numerical_triples) < n_samples:
        print(f"Warning: only {len(numerical_triples)} numerical triples available "
              f"(requested {n_samples})")
        n_samples = len(numerical_triples)

    # Stratified sampling by relation
    by_relation = {}
    for t in numerical_triples:
        by_relation.setdefault(t["relation"], []).append(t)

    # Proportional allocation per relation
    selected = []
    relations = list(by_relation.keys())
    per_relation = max(1, n_samples // len(relations))
    remainder = n_samples - per_relation * len(relations)

    for rel in relations:
        pool = by_relation[rel]
        k = min(per_relation, len(pool))
        selected.extend(random.sample(pool, k))

    # Fill remainder from largest groups
    if len(selected) < n_samples:
        remaining_pool = [t for t in numerical_triples if t not in selected]
        random.shuffle(remaining_pool)
        selected.extend(remaining_pool[:n_samples - len(selected)])

    selected = selected[:n_samples]
    random.shuffle(selected)

    # Generate questions via LLM (with rate limiting and checkpointing)
    benchmark = []
    checkpoint_path = ECOSTATS_DIR / "ecostats_benchmark_partial.json"

    # Resume from checkpoint if it exists
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        start_idx = len(benchmark)
        print(f"Resuming from checkpoint: {start_idx}/{len(selected)} done")

    call_count = 0
    window_start = time.time()

    for i, triple in enumerate(tqdm(selected[start_idx:], desc="Generating questions",
                                     initial=start_idx, total=len(selected))):
        # Rate limiting
        call_count += 1
        if call_count >= MAX_CALLS_PER_MINUTE:
            elapsed = time.time() - window_start
            if elapsed < 60.0:
                wait = 60.0 - elapsed
                tqdm.write(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)
            call_count = 0
            window_start = time.time()

        question = generate_question(client, triple)
        benchmark.append({
            "id": start_idx + i,
            "question": question,
            "gold_triple": {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": str(triple["object"]),
            },
            "gold_numbers": extract_numbers(str(triple["object"])),
            "source_topic": triple.get("source_topic", ""),
        })

        # Checkpoint every 50 questions
        if (start_idx + i + 1) % 50 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(benchmark, f, indent=2, ensure_ascii=False)
            tqdm.write(f"  Checkpoint saved: {len(benchmark)} questions")

    # Save benchmark
    out_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Save statistics
    relation_dist = {}
    for item in benchmark:
        r = item["gold_triple"]["relation"]
        relation_dist[r] = relation_dist.get(r, 0) + 1

    stats = {
        "total_samples": len(benchmark),
        "unique_relations": len(relation_dist),
        "relation_distribution": dict(sorted(relation_dist.items(), key=lambda x: -x[1])),
    }
    stats_path = ECOSTATS_DIR / "benchmark_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nEcoStats benchmark generated: {len(benchmark)} samples")
    print(f"Unique relations: {len(relation_dist)}")
    print(f"Saved to: {out_path}")

    return benchmark


# ============================================================
# Part 2: Evaluation Metrics
# ============================================================

def normalize_number(s: str) -> float | None:
    """Convert a number string to float for comparison."""
    try:
        s = s.replace(",", "").replace("%", "").strip()
        return float(s)
    except (ValueError, AttributeError):
        return None


def numerical_exact_match(predicted: str, gold_numbers: list[str], tolerance: float = 0.01) -> float:
    """
    NEM: Check if any extracted number from prediction matches a gold number.
    Returns 1.0 if at least one gold number is found, 0.0 otherwise.
    Tolerance is relative (1% default).
    """
    pred_numbers = extract_numbers(predicted)

    for gold in gold_numbers:
        gold_val = normalize_number(gold)
        if gold_val is None:
            continue
        for pred in pred_numbers:
            pred_val = normalize_number(pred)
            if pred_val is None:
                continue
            # Relative tolerance
            if gold_val == 0:
                if pred_val == 0:
                    return 1.0
            elif abs(pred_val - gold_val) / abs(gold_val) <= tolerance:
                return 1.0

    return 0.0


def numerical_f1(predicted: str, gold_numbers: list[str]) -> float:
    """
    NF1: Token-level F1 between predicted numerical tokens and gold numerical tokens.
    """
    pred_nums = set(extract_numbers(predicted))
    gold_nums = set(gold_numbers)

    if not gold_nums:
        return 1.0 if not pred_nums else 0.0
    if not pred_nums:
        return 0.0

    # Normalize for comparison
    pred_normalized = set()
    for p in pred_nums:
        v = normalize_number(p)
        if v is not None:
            pred_normalized.add(round(v, 4))

    gold_normalized = set()
    for g in gold_nums:
        v = normalize_number(g)
        if v is not None:
            gold_normalized.add(round(v, 4))

    if not gold_normalized:
        return 0.0

    tp = len(pred_normalized & gold_normalized)
    precision = tp / len(pred_normalized) if pred_normalized else 0
    recall = tp / len(gold_normalized) if gold_normalized else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def factual_precision(predicted: str, all_gold_numbers: list[str]) -> float:
    """
    FP: Fraction of numerical claims in the prediction that are correct.
    Measures how many of the model's numbers match gold.
    """
    pred_nums = extract_numbers(predicted)
    if not pred_nums:
        return 1.0  # No claims made = vacuously precise

    gold_set = set()
    for g in all_gold_numbers:
        v = normalize_number(g)
        if v is not None:
            gold_set.add(round(v, 4))

    correct = 0
    for p in pred_nums:
        v = normalize_number(p)
        if v is not None and round(v, 4) in gold_set:
            correct += 1

    return correct / len(pred_nums)


def evaluate_results(results_path: Path) -> dict:
    """
    Evaluate a results file against the EcoStats benchmark.
    
    Expected format: JSONL with {query, model, method, answer, ...}
    """
    # Load benchmark
    benchmark_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Create question → gold mapping
    gold_map = {item["question"]: item for item in benchmark}

    # Load results
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Compute metrics
    nem_scores = []
    nf1_scores = []
    fp_scores = []
    latencies = []
    tokens = []

    for result in results:
        query = result["query"]
        if query not in gold_map:
            continue

        gold = gold_map[query]
        pred = result["answer"]
        gold_nums = gold["gold_numbers"]

        nem_scores.append(numerical_exact_match(pred, gold_nums))
        nf1_scores.append(numerical_f1(pred, gold_nums))
        fp_scores.append(factual_precision(pred, gold_nums))

        if "latency_ms" in result:
            latencies.append(result["latency_ms"])
        if "token_usage" in result:
            t = result["token_usage"].get("total_tokens", 0)
            if t:
                tokens.append(t)

    n = len(nem_scores)
    if n == 0:
        return {"error": "No matching results found"}

    metrics = {
        "n_evaluated": n,
        "NEM": sum(nem_scores) / n,
        "NF1": sum(nf1_scores) / n,
        "FP": sum(fp_scores) / n,
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
        "tokens_mean": sum(tokens) / len(tokens) if tokens else 0,
    }

    return metrics


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print evaluation metrics."""
    if label:
        print(f"\n=== {label} ===")
    print(f"  Samples evaluated: {metrics['n_evaluated']}")
    print(f"  NEM  (Numerical Exact Match): {metrics['NEM']:.4f}")
    print(f"  NF1  (Numerical F1):          {metrics['NF1']:.4f}")
    print(f"  FP   (Factual Precision):     {metrics['FP']:.4f}")
    if metrics.get("latency_mean_ms"):
        print(f"  Latency (mean):               {metrics['latency_mean_ms']:.0f} ms")
        print(f"  Latency (p95):                {metrics['latency_p95_ms']:.0f} ms")
    if metrics.get("tokens_mean"):
        print(f"  Tokens (mean):                {metrics['tokens_mean']:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation & Benchmark")
    parser.add_argument("--generate-benchmark", action="store_true",
                        help="Generate the EcoStats-500 benchmark")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to results JSONL file to evaluate")
    parser.add_argument("--n-samples", type=int, default=NUM_EVAL_SAMPLES)
    args = parser.parse_args()

    if args.generate_benchmark:
        generate_ecostats_benchmark(n_samples=args.n_samples)

    if args.evaluate:
        metrics = evaluate_results(Path(args.evaluate))
        print_metrics(metrics, label=Path(args.evaluate).stem)

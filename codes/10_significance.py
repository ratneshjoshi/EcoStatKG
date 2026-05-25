"""
Step 10: Statistical Significance Tests
=========================================
Computes pairwise statistical significance between methods:
  1. Paired bootstrap resampling (10K iterations) with 95% CIs
  2. McNemar's exact test for binary NEM outcomes
  3. Bonferroni correction for multiple comparisons

Runs entirely post-hoc from existing .jsonl result files — no API calls.

Usage:
    python 10_significance.py                          # All exp1 results
    python 10_significance.py --results-dir results/exp1_main
    python 10_significance.py --reference kg_retrieval --alpha 0.01
"""

import json
import argparse
import logging
from pathlib import Path
from itertools import combinations
import math
import random

from config import ECOSTATS_DIR, RESULTS_DIR

log = logging.getLogger("ecostats.significance")


# ============================================================
# Number extraction (duplicated from 07 to keep standalone)
# ============================================================

import re

def _extract_numbers(text: str) -> list[str]:
    pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?'
    return re.findall(pattern, str(text))


def _normalize_number(s: str) -> float | None:
    s = s.strip().rstrip('%')
    s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def _nem_score(predicted: str, gold_numbers: list[str], tolerance: float = 0.01) -> int:
    """Binary NEM: 1 if any gold number matched, 0 otherwise."""
    pred_numbers = _extract_numbers(predicted)
    for gold in gold_numbers:
        gold_val = _normalize_number(gold)
        if gold_val is None:
            continue
        for pred in pred_numbers:
            pred_val = _normalize_number(pred)
            if pred_val is None:
                continue
            if gold_val == 0:
                if pred_val == 0:
                    return 1
            elif abs(pred_val - gold_val) / abs(gold_val) <= tolerance:
                return 1
    return 0


# ============================================================
# Bootstrap Resampling
# ============================================================

def paired_bootstrap(
    scores_a: list[int],
    scores_b: list[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test: is mean(A) - mean(B) significantly different from 0?

    Returns dict with:
      - delta: observed difference (mean_a - mean_b)
      - ci_lower, ci_upper: 95% CI of the difference
      - p_value: two-sided p-value (fraction of bootstrap deltas with opposite sign)
    """
    assert len(scores_a) == len(scores_b)
    n = len(scores_a)
    rng = random.Random(seed)

    observed_delta = sum(scores_a) / n - sum(scores_b) / n

    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in indices) / n
        mean_b = sum(scores_b[i] for i in indices) / n
        bootstrap_deltas.append(mean_a - mean_b)

    bootstrap_deltas.sort()
    ci_lower = bootstrap_deltas[int(0.025 * n_bootstrap)]
    ci_upper = bootstrap_deltas[int(0.975 * n_bootstrap)]

    # Two-sided p-value: proportion of deltas on the opposite side of 0
    if observed_delta >= 0:
        p_value = sum(1 for d in bootstrap_deltas if d <= 0) / n_bootstrap
    else:
        p_value = sum(1 for d in bootstrap_deltas if d >= 0) / n_bootstrap
    p_value = min(p_value * 2, 1.0)  # two-sided

    return {
        "delta": round(observed_delta, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "p_value": round(p_value, 6),
        "n_bootstrap": n_bootstrap,
    }


# ============================================================
# McNemar's Test
# ============================================================

def mcnemar_test(scores_a: list[int], scores_b: list[int]) -> dict:
    """
    McNemar's exact test for paired binary outcomes.

    Contingency table:
                B correct   B wrong
    A correct     n11         n10
    A wrong       n01         n00

    Tests whether n10 != n01 (discordant pairs).
    """
    assert len(scores_a) == len(scores_b)

    n11 = n10 = n01 = n00 = 0
    for a, b in zip(scores_a, scores_b):
        if a == 1 and b == 1:
            n11 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact binomial test: P(X >= max(n10,n01)) under H0: p=0.5
        k = max(n10, n01)
        n = discordant
        # Two-sided: sum both tails
        p_value = 0.0
        for i in range(k, n + 1):
            p_value += math.comb(n, i) * (0.5 ** n)
        p_value = min(p_value * 2, 1.0)  # two-sided

    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "discordant_pairs": discordant,
        "p_value": round(p_value, 6),
    }


# ============================================================
# Load and align results
# ============================================================

def load_query_scores(results_path: Path, benchmark: list[dict]) -> dict[str, int]:
    """Load a results file and compute per-query NEM scores.
    Returns {question: 0|1}."""
    gold_map = {item["question"]: item["gold_numbers"] for item in benchmark}

    scores = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            query = rec.get("query", "")
            if query in gold_map and "error" not in rec:
                scores[query] = _nem_score(rec.get("answer", ""), gold_map[query])
    return scores


def align_scores(
    scores_dict: dict[str, dict[str, int]],
) -> tuple[list[str], dict[str, list[int]]]:
    """Align multiple score dicts to the same query set (intersection)."""
    all_keys = [set(s.keys()) for s in scores_dict.values()]
    common = set.intersection(*all_keys) if all_keys else set()
    common_sorted = sorted(common)

    aligned = {}
    for label, scores in scores_dict.items():
        aligned[label] = [scores[q] for q in common_sorted]

    return common_sorted, aligned


# ============================================================
# Main analysis
# ============================================================

def run_significance_analysis(
    results_dir: Path,
    reference: str = "kg_retrieval",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> dict:
    """
    Run pairwise significance tests: reference method vs all others.
    Groups by model.
    """
    # Load benchmark
    benchmark_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Find all result files
    result_files = sorted(results_dir.glob("*.jsonl"))
    if not result_files:
        print(f"No .jsonl files found in {results_dir}")
        return {}

    # Parse: {method}_{model}.jsonl
    by_model = {}
    for rf in result_files:
        stem = rf.stem  # e.g., "hpti_qwen-27b"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        method, model = parts
        by_model.setdefault(model, {})[method] = rf

    all_results = {}
    total_comparisons = 0

    for model, method_files in sorted(by_model.items()):
        if reference not in method_files:
            print(f"  Skipping {model}: no {reference} results")
            continue

        # Load all scores
        scores_dict = {}
        for method, fpath in method_files.items():
            scores_dict[method] = load_query_scores(fpath, benchmark)

        queries, aligned = align_scores(scores_dict)
        n_queries = len(queries)

        if n_queries == 0:
            print(f"  Skipping {model}: no aligned queries")
            continue

        baselines = [m for m in aligned if m != reference]
        total_comparisons += len(baselines)

    # Bonferroni correction
    bonferroni_alpha = alpha / max(total_comparisons, 1)

    print(f"Significance tests: α={alpha}, Bonferroni α={bonferroni_alpha:.6f} "
          f"({total_comparisons} comparisons)")
    print(f"Bootstrap iterations: {n_bootstrap}")

    for model, method_files in sorted(by_model.items()):
        if reference not in method_files:
            continue

        scores_dict = {}
        for method, fpath in method_files.items():
            scores_dict[method] = load_query_scores(fpath, benchmark)

        queries, aligned = align_scores(scores_dict)
        n_queries = len(queries)
        if n_queries == 0:
            continue

        ref_scores = aligned[reference]
        ref_mean = sum(ref_scores) / len(ref_scores)
        baselines = sorted(m for m in aligned if m != reference)

        print(f"\n{'='*70}")
        print(f"Model: {model} ({n_queries} queries)")
        print(f"  {reference} NEM = {ref_mean:.4f}")
        print(f"{'='*70}")

        model_results = {}
        for baseline in baselines:
            base_scores = aligned[baseline]
            base_mean = sum(base_scores) / len(base_scores)

            boot = paired_bootstrap(ref_scores, base_scores, n_bootstrap=n_bootstrap)
            mcn = mcnemar_test(ref_scores, base_scores)

            sig_boot = boot["p_value"] < bonferroni_alpha
            sig_mcn = mcn["p_value"] < bonferroni_alpha

            result = {
                "reference": reference,
                "baseline": baseline,
                "model": model,
                "n_queries": n_queries,
                "ref_mean": round(ref_mean, 4),
                "base_mean": round(base_mean, 4),
                "bootstrap": boot,
                "mcnemar": mcn,
                "significant_bootstrap": sig_boot,
                "significant_mcnemar": sig_mcn,
                "bonferroni_alpha": round(bonferroni_alpha, 6),
            }
            model_results[baseline] = result

            sig_marker = "***" if (sig_boot and sig_mcn) else ("**" if sig_boot else ("*" if sig_mcn else "   "))
            print(f"  vs {baseline:<16} Δ={boot['delta']:+.4f}  "
                  f"95%CI=[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]  "
                  f"p_boot={boot['p_value']:.4f}  p_mcn={mcn['p_value']:.4f}  {sig_marker}")

        all_results[model] = model_results

    # Save
    output_path = results_dir / "significance_tests.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SIGNIFICANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'Baseline':<16} {'Δ NEM':>8} {'p (boot)':>10} {'p (McN)':>10} {'Sig?':>6}")
    print("-" * 70)
    for model, baselines in sorted(all_results.items()):
        for baseline, r in sorted(baselines.items()):
            sig = "YES" if r["significant_bootstrap"] else "no"
            print(f"{model:<16} {baseline:<16} {r['bootstrap']['delta']:>+8.4f} "
                  f"{r['bootstrap']['p_value']:>10.4f} {r['mcnemar']['p_value']:>10.4f} "
                  f"{sig:>6}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--results-dir", type=str,
                        default=str(RESULTS_DIR / "exp1_main"),
                        help="Directory with .jsonl result files")
    parser.add_argument("--reference", type=str, default="kg_retrieval",
                        help="Reference method (default: kg_retrieval)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap iterations")
    args = parser.parse_args()

    run_significance_analysis(
        results_dir=Path(args.results_dir),
        reference=args.reference,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()

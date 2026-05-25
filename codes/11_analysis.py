"""
Step 11: Post-Hoc Analysis
============================
Analyses that run on existing result files with no API calls:
  A) Per-relation NEM breakdown (P2-3)
  B) Triple attribution / grounding analysis (P2-4)

Usage:
    python 11_analysis.py --task relation --results results/exp1_main/kg_retrieval_qwen-27b.jsonl
    python 11_analysis.py --task attribution --results results/exp1_main/kg_retrieval_qwen-27b.jsonl
    python 11_analysis.py --task all --results-dir results/exp1_main
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

from config import ECOSTATS_DIR, RESULTS_DIR


# ============================================================
# Number helpers (shared with 07/10)
# ============================================================

def _extract_numbers(text: str) -> list[str]:
    return re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?', str(text))


def _normalize_number(s: str) -> float | None:
    s = s.strip().rstrip('%').replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def _nem_score(predicted: str, gold_numbers: list[str], tolerance: float = 0.01) -> int:
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
# A) Per-Relation Breakdown
# ============================================================

def per_relation_breakdown(results_path: Path) -> dict:
    """
    Group NEM scores by the gold triple's relation type.
    Returns {relation: {n, nem, correct, examples}}.
    """
    benchmark_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    gold_map = {item["question"]: item for item in benchmark}

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    by_relation = defaultdict(lambda: {"correct": 0, "total": 0, "examples": []})

    for rec in results:
        query = rec.get("query", "")
        if query not in gold_map or "error" in rec:
            continue
        gold = gold_map[query]
        relation = gold["gold_triple"]["relation"]
        nem = _nem_score(rec.get("answer", ""), gold["gold_numbers"])

        entry = by_relation[relation]
        entry["total"] += 1
        entry["correct"] += nem

        # Keep a few examples for each relation
        if len(entry["examples"]) < 3:
            entry["examples"].append({
                "question": query[:100],
                "gold": gold["gold_numbers"],
                "predicted_snippet": rec.get("answer", "")[:150],
                "nem": nem,
            })

    # Compute NEM per relation
    breakdown = {}
    for rel, data in sorted(by_relation.items(), key=lambda x: -x[1]["total"]):
        nem = data["correct"] / data["total"] if data["total"] > 0 else 0
        breakdown[rel] = {
            "n": data["total"],
            "nem": round(nem, 4),
            "correct": data["correct"],
            "examples": data["examples"],
        }

    # Print table
    method_model = results_path.stem
    print(f"\n{'='*70}")
    print(f"PER-RELATION NEM BREAKDOWN: {method_model}")
    print(f"{'='*70}")
    print(f"{'Relation':<35} {'N':>5} {'NEM':>8} {'Correct':>8}")
    print("-" * 60)
    for rel, data in sorted(breakdown.items(), key=lambda x: -x[1]["nem"]):
        print(f"{rel:<35} {data['n']:>5} {data['nem']:>8.4f} {data['correct']:>5}/{data['n']}")

    overall_n = sum(d["total"] for d in by_relation.values())
    overall_c = sum(d["correct"] for d in by_relation.values())
    print(f"{'OVERALL':<35} {overall_n:>5} {overall_c/overall_n:>8.4f} {overall_c:>5}/{overall_n}")

    return breakdown


# ============================================================
# B) Triple Attribution Analysis
# ============================================================

def attribution_analysis(results_path: Path) -> dict:
    """
    For each result with retrieved triples, determine which triple(s)
    the model's answer is grounded in by matching numbers.

    Returns attribution statistics and example items for figure generation.
    """
    benchmark_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    gold_map = {item["question"]: item for item in benchmark}

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    stats = {
        "total": 0,
        "has_triples": 0,
        "attributed": 0,        # answer numbers match a retrieved triple
        "gold_in_context": 0,   # gold answer appears in at least one triple
        "attributed_to_rank": [],  # rank of the best-matching triple
        "examples": [],
    }

    for rec in results:
        query = rec.get("query", "")
        if query not in gold_map or "error" in rec:
            continue

        gold = gold_map[query]
        answer = rec.get("answer", "")
        triples = rec.get("retrieved_triples", [])
        stats["total"] += 1

        if not triples:
            continue
        stats["has_triples"] += 1

        answer_nums = set()
        for n in _extract_numbers(answer):
            v = _normalize_number(n)
            if v is not None:
                answer_nums.add(round(v, 4))

        gold_nums_norm = set()
        for g in gold["gold_numbers"]:
            v = _normalize_number(g)
            if v is not None:
                gold_nums_norm.add(round(v, 4))

        # Check each triple for number overlap with answer
        best_rank = None
        best_overlap = 0
        gold_found_rank = None

        for rank, triple in enumerate(triples):
            doc = triple.get("document", "")
            triple_nums = set()
            for n in _extract_numbers(doc):
                v = _normalize_number(n)
                if v is not None:
                    triple_nums.add(round(v, 4))

            # Does this triple contain the gold answer?
            if gold_nums_norm & triple_nums and gold_found_rank is None:
                gold_found_rank = rank

            # How many answer numbers match this triple?
            overlap = len(answer_nums & triple_nums)
            if overlap > best_overlap:
                best_overlap = overlap
                best_rank = rank

        if best_rank is not None and best_overlap > 0:
            stats["attributed"] += 1
            stats["attributed_to_rank"].append(best_rank)

        if gold_found_rank is not None:
            stats["gold_in_context"] += 1

        # Save examples for figure
        if len(stats["examples"]) < 20:
            stats["examples"].append({
                "question": query,
                "answer": answer[:200],
                "gold_numbers": gold["gold_numbers"],
                "n_triples": len(triples),
                "best_match_rank": best_rank,
                "best_match_overlap": best_overlap,
                "gold_in_context_rank": gold_found_rank,
                "triples": [
                    {
                        "rank": i,
                        "text": t.get("document", "")[:200],
                        "is_best_match": i == best_rank,
                        "contains_gold": i == gold_found_rank,
                    }
                    for i, t in enumerate(triples[:5])
                ],
            })

    # Compute summary
    n = stats["total"]
    n_tri = stats["has_triples"]
    summary = {
        "total_queries": n,
        "queries_with_triples": n_tri,
        "attributed_pct": round(stats["attributed"] / n_tri * 100, 1) if n_tri else 0,
        "gold_in_context_pct": round(stats["gold_in_context"] / n_tri * 100, 1) if n_tri else 0,
    }

    if stats["attributed_to_rank"]:
        ranks = stats["attributed_to_rank"]
        summary["mean_attribution_rank"] = round(sum(ranks) / len(ranks), 2)
        summary["rank_1_pct"] = round(sum(1 for r in ranks if r == 0) / len(ranks) * 100, 1)
        summary["rank_top3_pct"] = round(sum(1 for r in ranks if r < 3) / len(ranks) * 100, 1)

    # Print
    method_model = results_path.stem
    print(f"\n{'='*70}")
    print(f"ATTRIBUTION ANALYSIS: {method_model}")
    print(f"{'='*70}")
    print(f"  Queries with triples: {n_tri}/{n}")
    print(f"  Gold answer in context: {stats['gold_in_context']}/{n_tri} "
          f"({summary['gold_in_context_pct']}%)")
    print(f"  Answer attributed to triple: {stats['attributed']}/{n_tri} "
          f"({summary['attributed_pct']}%)")
    if stats["attributed_to_rank"]:
        print(f"  Mean attribution rank: {summary['mean_attribution_rank']}")
        print(f"  Attributed to rank 1: {summary['rank_1_pct']}%")
        print(f"  Attributed to top 3: {summary['rank_top3_pct']}%")

    output = {"summary": summary, "examples": stats["examples"]}
    return output


# ============================================================
# Batch: All files in a directory
# ============================================================

def analyze_directory(results_dir: Path, task: str = "all"):
    """Run analysis on all .jsonl files in a directory."""
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {results_dir}")
        return

    all_relation = {}
    all_attribution = {}

    for fpath in files:
        label = fpath.stem
        print(f"\n--- {label} ---")

        if task in ("relation", "all"):
            all_relation[label] = per_relation_breakdown(fpath)

        if task in ("attribution", "all"):
            # Only for methods with retrieval
            if any(k in label for k in ("kg_retrieval", "hpti", "standard_rag", "graph_rag", "conli")):
                all_attribution[label] = attribution_analysis(fpath)

    # Save combined results
    output_dir = results_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    if all_relation:
        with open(output_dir / "per_relation_breakdown.json", "w") as f:
            json.dump(all_relation, f, indent=2)
        print(f"\nRelation breakdown saved to {output_dir / 'per_relation_breakdown.json'}")

    if all_attribution:
        with open(output_dir / "attribution_analysis.json", "w") as f:
            json.dump(all_attribution, f, indent=2)
        print(f"Attribution analysis saved to {output_dir / 'attribution_analysis.json'}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc analysis of results")
    parser.add_argument("--task", choices=["relation", "attribution", "all"],
                        default="all", help="Which analysis to run")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to a single .jsonl results file")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory of .jsonl files to analyze")
    args = parser.parse_args()

    if args.results:
        path = Path(args.results)
        if args.task in ("relation", "all"):
            per_relation_breakdown(path)
        if args.task in ("attribution", "all"):
            attribution_analysis(path)
    elif args.results_dir:
        analyze_directory(Path(args.results_dir), task=args.task)
    else:
        # Default: analyze exp1
        analyze_directory(RESULTS_DIR / "exp1_main", task=args.task)


if __name__ == "__main__":
    main()

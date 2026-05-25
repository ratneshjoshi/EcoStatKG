"""
Compute inter-annotator agreement (Cohen's kappa) and aggregate scores
for both annotation tasks.

Usage:
    python compute_agreement.py --task kg --a1 kg_triple_annotation_ann1.csv --a2 kg_triple_annotation_ann2.csv
    python compute_agreement.py --task benchmark --a1 benchmark_iaa_annotation_ann1.csv --a2 benchmark_iaa_annotation_ann2.csv
"""

import argparse
import csv
import sys
from collections import Counter


def cohens_kappa(labels1: list, labels2: list) -> float:
    """Compute Cohen's kappa for two lists of categorical labels."""
    assert len(labels1) == len(labels2), "Label lists must have equal length"
    n = len(labels1)
    if n == 0:
        return 0.0

    categories = sorted(set(labels1) | set(labels2))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels1, labels2):
        matrix[cat_idx[a]][cat_idx[b]] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    p_e = 0.0
    for i in range(k):
        row_sum = sum(matrix[i][j] for j in range(k))
        col_sum = sum(matrix[j][i] for j in range(k))
        p_e += (row_sum * col_sum) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def load_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run_kg_task(a1_path: str, a2_path: str):
    a1 = load_csv(a1_path)
    a2 = load_csv(a2_path)

    a1_map = {r['sample_id']: r for r in a1}
    a2_map = {r['sample_id']: r for r in a2}
    common_ids = sorted(set(a1_map) & set(a2_map))

    if not common_ids:
        print("ERROR: No matching sample_ids found between the two files.")
        sys.exit(1)

    scores1 = [int(a1_map[sid]['score']) for sid in common_ids]
    scores2 = [int(a2_map[sid]['score']) for sid in common_ids]

    kappa = cohens_kappa(scores1, scores2)

    # Aggregate (use annotator 1 scores, or average for final)
    all_scores = [(s1 + s2) / 2 for s1, s2 in zip(scores1, scores2)]

    n = len(common_ids)
    correct = sum(1 for s in all_scores if s >= 1.5)  # both gave 2 or one gave 2 + one gave 1
    strict_correct = sum(1 for s1, s2 in zip(scores1, scores2) if s1 == 2 and s2 == 2)
    partial = sum(1 for s in all_scores if 0.5 <= s < 1.5)
    incorrect = sum(1 for s in all_scores if s < 0.5)

    # Disagreements
    disagreements = [(sid, scores1[i], scores2[i]) for i, sid in enumerate(common_ids)
                     if abs(scores1[i] - scores2[i]) >= 2]

    print("=" * 60)
    print("KG TRIPLE QUALITY EVALUATION — RESULTS")
    print("=" * 60)
    print(f"Items annotated:     {n}")
    print(f"Cohen's kappa:       {kappa:.4f}")
    print()
    print("Score distribution (averaged):")
    print(f"  Correct (≥1.5):    {correct} ({correct/n*100:.1f}%)")
    print(f"  Partial (0.5-1.5): {partial} ({partial/n*100:.1f}%)")
    print(f"  Incorrect (<0.5):  {incorrect} ({incorrect/n*100:.1f}%)")
    print()
    print(f"Strict precision (both annotators = 2): {strict_correct/n*100:.1f}%")
    print(f"Lenient precision (avg ≥ 1.5):          {correct/n*100:.1f}%")
    print()

    if disagreements:
        print(f"Major disagreements (|diff| ≥ 2): {len(disagreements)}")
        for sid, s1, s2 in disagreements:
            print(f"  {sid}: Annotator1={s1}, Annotator2={s2}")
    else:
        print("No major disagreements.")

    # Kappa interpretation
    if kappa >= 0.81:
        interp = "Almost perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    else:
        interp = "Slight/Poor"
    print(f"\nKappa interpretation: {interp} ({kappa:.4f})")


def run_benchmark_task(a1_path: str, a2_path: str):
    a1 = load_csv(a1_path)
    a2 = load_csv(a2_path)

    a1_map = {r['sample_id']: r for r in a1}
    a2_map = {r['sample_id']: r for r in a2}
    common_ids = sorted(set(a1_map) & set(a2_map))

    if not common_ids:
        print("ERROR: No matching sample_ids found between the two files.")
        sys.exit(1)

    qq1 = [int(a1_map[sid]['question_quality']) for sid in common_ids]
    qq2 = [int(a2_map[sid]['question_quality']) for sid in common_ids]
    ac1 = [int(a1_map[sid]['answer_correctness']) for sid in common_ids]
    ac2 = [int(a2_map[sid]['answer_correctness']) for sid in common_ids]

    kappa_qq = cohens_kappa(qq1, qq2)
    kappa_ac = cohens_kappa(ac1, ac2)

    n = len(common_ids)

    # Average scores
    qq_avg = [(s1 + s2) / 2 for s1, s2 in zip(qq1, qq2)]
    ac_avg = [(s1 + s2) / 2 for s1, s2 in zip(ac1, ac2)]

    qq_good = sum(1 for s in qq_avg if s >= 2.5)
    qq_accept = sum(1 for s in qq_avg if 1.5 <= s < 2.5)
    qq_poor = sum(1 for s in qq_avg if s < 1.5)

    ac_correct = sum(1 for s in ac_avg if s >= 2.5)
    ac_partial = sum(1 for s in ac_avg if 1.5 <= s < 2.5)
    ac_incorrect = sum(1 for s in ac_avg if s < 1.5)

    # Overall validity
    valid = sum(1 for qq, ac in zip(qq_avg, ac_avg) if qq >= 1.5 and ac >= 1.5)

    print("=" * 60)
    print("BENCHMARK IAA VALIDATION — RESULTS")
    print("=" * 60)
    print(f"Items annotated:          {n}")
    print(f"Cohen's kappa (Q quality): {kappa_qq:.4f}")
    print(f"Cohen's kappa (A correct): {kappa_ac:.4f}")
    print()
    print("Question Quality distribution (averaged):")
    print(f"  Good (≥2.5):       {qq_good} ({qq_good/n*100:.1f}%)")
    print(f"  Acceptable (1.5-): {qq_accept} ({qq_accept/n*100:.1f}%)")
    print(f"  Poor (<1.5):       {qq_poor} ({qq_poor/n*100:.1f}%)")
    print()
    print("Answer Correctness distribution (averaged):")
    print(f"  Correct (≥2.5):    {ac_correct} ({ac_correct/n*100:.1f}%)")
    print(f"  Partial (1.5-):    {ac_partial} ({ac_partial/n*100:.1f}%)")
    print(f"  Incorrect (<1.5):  {ac_incorrect} ({ac_incorrect/n*100:.1f}%)")
    print()
    print(f"Overall benchmark validity (both ≥ Acceptable): {valid/n*100:.1f}%")

    # Disagreements
    qq_disagree = [(common_ids[i], qq1[i], qq2[i]) for i in range(n) if abs(qq1[i] - qq2[i]) >= 2]
    ac_disagree = [(common_ids[i], ac1[i], ac2[i]) for i in range(n) if abs(ac1[i] - ac2[i]) >= 2]

    if qq_disagree:
        print(f"\nQuestion quality major disagreements: {len(qq_disagree)}")
        for sid, s1, s2 in qq_disagree:
            print(f"  {sid}: Ann1={s1}, Ann2={s2}")

    if ac_disagree:
        print(f"\nAnswer correctness major disagreements: {len(ac_disagree)}")
        for sid, s1, s2 in ac_disagree:
            print(f"  {sid}: Ann1={s1}, Ann2={s2}")

    if not qq_disagree and not ac_disagree:
        print("\nNo major disagreements.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement")
    parser.add_argument("--task", choices=["kg", "benchmark"], required=True)
    parser.add_argument("--a1", required=True, help="Path to annotator 1 CSV")
    parser.add_argument("--a2", required=True, help="Path to annotator 2 CSV")
    args = parser.parse_args()

    if args.task == "kg":
        run_kg_task(args.a1, args.a2)
    else:
        run_benchmark_task(args.a1, args.a2)

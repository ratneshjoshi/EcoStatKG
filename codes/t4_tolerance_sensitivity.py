"""
T4: NEM tolerance sensitivity + validation.
Re-scores stored predictions at multiple relative tolerances. No API calls.
Reproduces paper NEM at tol=0.01 as a validation check first.
"""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BENCH = ROOT / "codes" / "data" / "ecostats" / "ecostats_benchmark.json"

# 6 models used in the paper (exclude devstral-24b, the extractor)
PAPER_MODELS = ["mistral-small-24b", "ministral-14b", "qwen-27b",
                "deepseek-v4-flash", "glm-5", "gpt-oss-120b"]
METHODS = ["zero_shot", "standard_rag", "hpti", "graph_rag",
           "conli", "cove", "lightrag", "ms_graphrag"]
METHOD_LABEL = {"zero_shot": "Zero-shot", "standard_rag": "Standard RAG",
                "hpti": "KG Cosine", "graph_rag": "Graph RAG", "conli": "CoNLI",
                "cove": "CoVe", "lightrag": "LightRAG", "ms_graphrag": "MS GraphRAG"}
TOLERANCES = [0.005, 0.01, 0.02, 0.05]

# Paper-reported full-index NEM (Table 3) for validation
PAPER_FI = {"zero_shot": 0.508, "standard_rag": 0.669, "hpti": 0.989,
            "graph_rag": 0.992, "conli": 0.717, "cove": 0.496,
            "lightrag": 0.922, "ms_graphrag": 0.397}


def extract_numbers(text):
    return re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?', str(text))

def normalize_number(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return None

def nem(predicted, gold_numbers, tol):
    preds = [normalize_number(p) for p in extract_numbers(predicted)]
    preds = [p for p in preds if p is not None]
    for gold in gold_numbers:
        gv = normalize_number(gold)
        if gv is None:
            continue
        for pv in preds:
            if gv == 0:
                if pv == 0:
                    return 1
            elif abs(pv - gv) / abs(gv) <= tol:
                return 1
    return 0


def load_gold():
    bench = json.load(open(BENCH))
    return {item["question"]: item["gold_numbers"] for item in bench}


def score_file(path, gold_map, tol):
    """Return list of per-question NEM at given tolerance (clean records only)."""
    scores = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("query", "")
            if q in gold_map and "error" not in rec:
                scores.append(nem(rec.get("answer", ""), gold_map[q], tol))
    return scores


def method_avg(condition_dir, method, gold_map, tol):
    """Average NEM across the 6 paper models for one method."""
    per_model = []
    for m in PAPER_MODELS:
        p = condition_dir / f"{method}_{m}.jsonl"
        if p.exists():
            s = score_file(p, gold_map, tol)
            if s:
                per_model.append(sum(s) / len(s))
    return sum(per_model) / len(per_model) if per_model else None


def main():
    gold_map = load_gold()
    fi = RESULTS / "exp1_full_index"
    ho = RESULTS / "exp1_main"

    # ---- Validation at tol=0.01 vs paper ----
    print("=== VALIDATION: full-index NEM at tol=1% vs paper ===")
    print(f"{'Method':<14}{'ours':>8}{'paper':>8}{'diff':>8}")
    for meth in METHODS:
        ours = method_avg(fi, meth, gold_map, 0.01)
        pap = PAPER_FI[meth]
        print(f"{METHOD_LABEL[meth]:<14}{ours:>8.3f}{pap:>8.3f}{ours-pap:>+8.3f}")

    # ---- T4: tolerance sensitivity ----
    for cond_name, cond_dir in [("FULL-INDEX", fi), ("HELD-OUT", ho)]:
        print(f"\n=== T4 TOLERANCE SENSITIVITY: {cond_name} (avg over 6 models) ===")
        print(f"{'Method':<14}" + "".join(f"{t*100:>8g}%" for t in TOLERANCES))
        for meth in METHODS:
            row = f"{METHOD_LABEL[meth]:<14}"
            for tol in TOLERANCES:
                v = method_avg(cond_dir, meth, gold_map, tol)
                row += f"{v:>9.3f}" if v is not None else f"{'--':>9}"
            print(row)


if __name__ == "__main__":
    main()

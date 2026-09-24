"""
T7 threshold-significance: does confidence gating significantly recover clinical
held-out NEM, and is the gated result algorithm-agnostic / at zero-shot parity?

Paired bootstrap (10K) + McNemar on per-question NEM, pooled over the two
threshold-swept clinical models (qwen-27b, glm-5[=deepseek-v4-flash alias]).
Reads existing result files only; no API calls.

Three tests at t=0.20:
  (a) RECOVERY   : hpti gated (t=0.20) vs no-gate (tnone)  -> should be significant
  (b) ALGORITHM  : hpti (t=0.20) vs graph_rag (t=0.20)     -> should be non-sig
  (c) PARITY     : hpti (t=0.20) vs zero_shot              -> should be non-sig
"""
import json, re, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "codes" / "data_clinical" / "ecostats" / "ecostats_benchmark.json"
THR = ROOT / "results_clinical" / "exp9_threshold"
MAIN = ROOT / "results_clinical" / "exp1_main"
MODELS = ["qwen-27b", "glm-5"]  # glm-5 endpoint = deepseek-v4-flash after June alias switch

gold = {i["question"]: i["gold_numbers"] for i in json.load(open(BENCH))}

def nums(t): return re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?', str(t))
def norm(s):
    try: return float(str(s).replace(",", "").replace("%", "").strip())
    except: return None
def nem(pred, golds, tol=0.01):
    ps = [p for p in (norm(x) for x in nums(pred)) if p is not None]
    for g in golds:
        gv = norm(g)
        if gv is None: continue
        for pv in ps:
            if gv == 0:
                if pv == 0: return 1
            elif abs(pv - gv) / abs(gv) <= tol: return 1
    return 0
def perq(path):
    o = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); q = r.get("query", "")
            if q in gold and "error" not in r:
                o[q] = nem(r.get("answer", ""), gold[q])
    return o
def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = n10 + n01
    if n == 0: return 1.0, n10, n01
    k = max(n10, n01)
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1)) * 2
    return min(p, 1.0), n10, n01
def boot(a, b, B=10000, seed=42):
    rng = random.Random(seed); n = len(a); obs = sum(a) / n - sum(b) / n; ds = []
    for _ in range(B):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        ds.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    ds.sort(); lo = ds[int(.025 * B)]; hi = ds[int(.975 * B)]
    p = (sum(1 for d in ds if d <= 0) / B) if obs >= 0 else (sum(1 for d in ds if d >= 0) / B)
    return obs, lo, hi, min(p * 2, 1.0)
def pooled(fileA, fileB):
    A = []; B = []
    for m in MODELS:
        pa = perq(fileA(m)); pb = perq(fileB(m)); c = sorted(set(pa) & set(pb))
        A += [pa[q] for q in c]; B += [pb[q] for q in c]
    return A, B
def report(name, A, B):
    o, lo, hi, pb = boot(A, B); pm, n10, n01 = mcnemar(A, B)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
    print(f"\n{name} (n={len(A)})")
    print(f"  A={sum(A)/len(A):.3f}  B={sum(B)/len(B):.3f}  Delta={o:+.3f}  "
          f"95% CI=[{lo:+.3f},{hi:+.3f}]  p_boot={pb:.4f}  "
          f"McNemar p={pm:.4f} (Awin={n10}, Bwin={n01})  -> {sig}")

def main():
    A, B = pooled(lambda m: THR / f"hpti_t0.20_{m}.jsonl", lambda m: THR / f"hpti_tnone_{m}.jsonl")
    report("(a) RECOVERY   hpti gated t=0.20  vs  no-gate (tnone)", A, B)
    A, B = pooled(lambda m: THR / f"hpti_t0.20_{m}.jsonl", lambda m: THR / f"graph_rag_t0.20_{m}.jsonl")
    report("(b) ALGORITHM  hpti t=0.20  vs  graph_rag t=0.20", A, B)
    A, B = pooled(lambda m: THR / f"hpti_t0.20_{m}.jsonl", lambda m: MAIN / f"zero_shot_{m}.jsonl")
    report("(c) PARITY     hpti t=0.20  vs  zero_shot", A, B)

if __name__ == "__main__":
    main()

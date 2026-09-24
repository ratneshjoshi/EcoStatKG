"""
T5 (clinical): KG Cosine (kg_retrieval) vs Graph RAG significance test on the
clinical cross-domain results, pooling the 3 evaluated models' per-question NEM.
Mirrors t5_pooled_kgcos_vs_graphrag.py (env domain). No API calls.
"""
import json, re, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_clinical"
BENCH = ROOT / "codes" / "data_clinical" / "ecostats" / "ecostats_benchmark.json"
# glm-5 endpoint resolves to deepseek-v4-flash after the June alias switch;
# reported as deepseek-v4-flash in the paper.
MODELS = ["qwen-27b", "glm-5", "ministral-14b"]

def extract_numbers(t): return re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?', str(t))
def norm(s):
    try: return float(str(s).replace(",", "").replace("%", "").strip())
    except: return None
def nem(pred, golds, tol=0.01):
    ps=[norm(p) for p in extract_numbers(pred)]; ps=[p for p in ps if p is not None]
    for g in golds:
        gv=norm(g)
        if gv is None: continue
        for pv in ps:
            if gv==0:
                if pv==0: return 1
            elif abs(pv-gv)/abs(gv)<=tol: return 1
    return 0

def per_q(path, gold):
    out={}
    if not path.exists(): return out
    for line in open(path):
        line=line.strip()
        if not line: continue
        r=json.loads(line); q=r.get("query","")
        if q in gold and "error" not in r:
            out[q]=nem(r.get("answer",""), gold[q])
    return out

def mcnemar(a,b):
    n10=sum(1 for x,y in zip(a,b) if x==1 and y==0)
    n01=sum(1 for x,y in zip(a,b) if x==0 and y==1)
    n=n10+n01
    if n==0: return 1.0,n10,n01
    k=max(n10,n01)
    p=sum(math.comb(n,i)*0.5**n for i in range(k,n+1))*2
    return min(p,1.0),n10,n01

def bootstrap(a,b,n_boot=10000,seed=42):
    rng=random.Random(seed); n=len(a)
    obs=sum(a)/n - sum(b)/n
    deltas=[]
    for _ in range(n_boot):
        idx=[rng.randint(0,n-1) for _ in range(n)]
        deltas.append(sum(a[i] for i in idx)/n - sum(b[i] for i in idx)/n)
    deltas.sort()
    lo=deltas[int(0.025*n_boot)]; hi=deltas[int(0.975*n_boot)]
    p = (sum(1 for d in deltas if d<=0)/n_boot) if obs>=0 else (sum(1 for d in deltas if d>=0)/n_boot)
    return obs, lo, hi, min(p*2,1.0)

def main():
    bench=json.load(open(BENCH))
    gold={i["question"]:i["gold_numbers"] for i in bench}
    for cond in ["exp1_full_index","exp1_main"]:
        d=RESULTS/cond
        A=[]; B=[]
        for m in MODELS:
            ha=per_q(d/f"kg_retrieval_{m}.jsonl", gold)
            gb=per_q(d/f"graph_rag_{m}.jsonl", gold)
            common=sorted(set(ha)&set(gb))
            A+=[ha[q] for q in common]; B+=[gb[q] for q in common]
        if not A:
            print(f"=== {cond}: no paired data found ==="); continue
        obs,lo,hi,pb=bootstrap(A,B)
        pm,n10,n01=mcnemar(A,B)
        label = "FULL-INDEX" if cond=="exp1_full_index" else "HELD-OUT"
        print(f"=== T5 CLINICAL KG Cosine vs Graph RAG: {label} (n={len(A)} paired) ===")
        print(f"  KG Cosine NEM={sum(A)/len(A):.4f}  Graph RAG NEM={sum(B)/len(B):.4f}")
        print(f"  Delta={obs:+.4f}  bootstrap 95% CI=[{lo:+.4f}, {hi:+.4f}]  p_boot={pb:.4f}")
        print(f"  McNemar: discordant (KGwin={n10}, Graphwin={n01})  p_mcnemar={pm:.4f}")
        print()

if __name__=="__main__":
    main()

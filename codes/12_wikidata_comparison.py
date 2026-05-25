"""
Step 12: Wikidata Comparison
==============================
Extracts numerical claims from Wikidata for topics matching EcoStatKG,
embeds them, indexes in ChromaDB, and runs KG retrieval against the Wikidata index
for a direct comparison of KG coverage effectiveness.

Pipeline:
  1. Query Wikidata SPARQL for numerical properties of EcoStatKG topics
  2. Convert to triple format (subject, relation, object)
  3. Embed and index in ChromaDB as 'wikidata_fine'
  4. Run KG retrieval with wikidata_fine vs hpti_fine on the benchmark
  5. Compare NEM scores

Usage:
    python 12_wikidata_comparison.py --extract          # Step 1-3: Build Wikidata index
    python 12_wikidata_comparison.py --compare          # Step 4-5: Run comparison
    python 12_wikidata_comparison.py --all              # Full pipeline
"""

import json
import argparse
import logging
import time
import re
import requests
from pathlib import Path
from tqdm import tqdm

import chromadb

from config import (
    GENERATOR_MODELS, EMBEDDING_MODEL,
    CHROMA_DIR, ECOSTATS_DIR, RESULTS_DIR, DATA_DIR,
    TOP_K_DEFAULT, MAX_CALLS_PER_MINUTE,
    GROUNDED_SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS, TOP_P,
)
from utils import create_client, embed_query, chat_completion, GenerationResult

log = logging.getLogger("ecostats.wikidata")

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Numerical properties commonly found on environmental Wikidata items
NUMERICAL_PROPERTIES = {
    "P2046": "area",
    "P2044": "elevation",
    "P1082": "population",
    "P2052": "speed_limit",
    "P2048": "height",
    "P2049": "width",
    "P2043": "length",
    "P2054": "density",
    "P2120": "radius",
    "P2234": "volume_as_quantity",
    "P2386": "diameter",
    "P4010": "GDP",
    "P1538": "households",
    "P2196": "students_count",
    "P3362": "operating_income",
    "P2139": "revenue",
    "P2295": "net_profit",
    "P2403": "total_assets",
    "P1128": "employees",
    "P1114": "quantity",
    "P2131": "nominal_GDP",
    "P2132": "nominal_GDP_per_capita",
    "P2299": "PPP_GDP_per_capita",
    "P2218": "net_worth",
    "P2225": "discharge",
    "P2227": "catchment_area",
    "P4511": "vertical_depth",
    "P5765": "capacity",
}

# Map Wikidata properties to EcoStatKG-style relation types
PROPERTY_TO_RELATION = {
    "P2046": "HasArea",
    "P2044": "HasElevation",
    "P1082": "HasPopulation",
    "P2048": "HasHeight",
    "P2049": "HasWidth",
    "P2043": "HasLength",
    "P2054": "HasDensity",
    "P2234": "HasVolume",
    "P5765": "HasCapacity",
    "P2225": "HasDischarge",
    "P2227": "HasCatchmentArea",
    "P1128": "HasEmployees",
    "P2139": "HasRevenue",
    "P1114": "HasQuantity",
}


# ============================================================
# Step 1: Extract from Wikidata SPARQL
# ============================================================

def load_ecostats_topics() -> list[str]:
    """Load EcoStatKG topic list."""
    topics_path = DATA_DIR / "expanded_topics.txt"
    if not topics_path.exists():
        topics_path = DATA_DIR / "seed_topics.txt"
    with open(topics_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def query_wikidata_batch(topics: list[str], batch_size: int = 25) -> list[dict]:
    """
    Query Wikidata SPARQL for numerical properties of given topics.
    Uses Wikipedia sitelinks for reliable entity resolution.
    Returns list of {subject, relation, object, source_topic, wikidata_id}.
    """
    all_triples = []
    prop_ids = list(NUMERICAL_PROPERTIES.keys())
    prop_str = " ".join(f"wdt:{p}" for p in prop_ids)

    for i in tqdm(range(0, len(topics), batch_size), desc="Querying Wikidata"):
        batch = topics[i:i + batch_size]
        # Build Wikipedia sitelink URIs (topics come from Wikipedia article titles)
        sitelink_values = " ".join(
            f'<https://en.wikipedia.org/wiki/{t.replace(" ", "_").replace(chr(34), "")}>'
            for t in batch
        )

        query = f"""
        SELECT ?item ?itemLabel ?prop ?value ?unitLabel ?sitelink WHERE {{
          VALUES ?sitelink {{ {sitelink_values} }}
          ?sitelink schema:about ?item .

          VALUES ?prop {{ {prop_str} }}
          ?item ?prop ?value .
          FILTER(DATATYPE(?value) = xsd:decimal || DATATYPE(?value) = xsd:integer || DATATYPE(?value) = xsd:double || isNumeric(?value))

          OPTIONAL {{
            ?item ?prop ?stmt .
            ?stmt ?psv/wikibase:quantityUnit ?unit .
          }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 5000
        """

        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "EcoStatKG/1.0 (research project)",
        }
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query},
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 429:
                log.warning("Wikidata rate limit, waiting 60s...")
                time.sleep(60)
                resp = requests.get(
                    WIKIDATA_SPARQL_URL,
                    params={"query": query},
                    headers=headers,
                    timeout=60,
                )

            if resp.status_code != 200:
                log.warning("Wikidata query failed (%d) for batch %d", resp.status_code, i)
                continue

            data = resp.json()
            for result in data.get("results", {}).get("bindings", []):
                item_label = result.get("itemLabel", {}).get("value", "")
                prop_uri = result.get("prop", {}).get("value", "")
                value = result.get("value", {}).get("value", "")
                unit = result.get("unitLabel", {}).get("value", "")
                sitelink = result.get("sitelink", {}).get("value", "")

                # Extract property ID
                prop_id = prop_uri.split("/")[-1] if "/" in prop_uri else ""
                relation = PROPERTY_TO_RELATION.get(
                    prop_id, NUMERICAL_PROPERTIES.get(prop_id, prop_id)
                )

                # Format object with unit
                obj = f"{value} {unit}".strip() if unit and unit != "1" else str(value)

                # Match to source topic via sitelink URL
                source_topic = ""
                if sitelink:
                    wiki_title = sitelink.split("/wiki/")[-1].replace("_", " ")
                    for t in batch:
                        if t.lower() == wiki_title.lower():
                            source_topic = t
                            break
                if not source_topic:
                    for t in batch:
                        if t.lower() in item_label.lower() or item_label.lower() in t.lower():
                            source_topic = t
                            break

                all_triples.append({
                    "subject": item_label,
                    "relation": relation,
                    "object": obj,
                    "source_topic": source_topic or item_label,
                    "source": "wikidata",
                    "wikidata_id": result.get("item", {}).get("value", ""),
                })

        except Exception as e:
            log.warning("Wikidata batch %d error: %s", i, e)
            continue

        # Be polite to Wikidata
        time.sleep(2)

    return all_triples


def extract_wikidata_triples() -> list[dict]:
    """Full extraction pipeline: topics → SPARQL → triples."""
    topics = load_ecostats_topics()
    print(f"Querying Wikidata for {len(topics)} EcoStatKG topics...")

    triples = query_wikidata_batch(topics)

    # Deduplicate
    seen = set()
    unique = []
    for t in triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # Save
    output_path = DATA_DIR / "wikidata" / "wikidata_triples.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in unique:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Extracted {len(unique)} unique Wikidata triples (from {len(triples)} raw)")
    print(f"Saved to {output_path}")

    return unique


# ============================================================
# Step 2-3: Embed and index in ChromaDB
# ============================================================

def index_wikidata_triples(triples: list[dict] | None = None):
    """Embed Wikidata triples and index in ChromaDB."""
    if triples is None:
        triples_path = DATA_DIR / "wikidata" / "wikidata_triples.jsonl"
        triples = []
        with open(triples_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    triples.append(json.loads(line))

    if not triples:
        print("No triples to index")
        return

    client = create_client()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Create/reset collection
    try:
        chroma_client.delete_collection("wikidata_fine")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        "wikidata_fine",
        metadata={"hnsw:space": "cosine"},
    )

    # Format documents like EcoStatKG fine-grained triples
    documents = []
    metadatas = []
    ids = []
    for i, t in enumerate(triples):
        doc = f"The {t['relation']} of {t['subject']} is {t['object']}."
        documents.append(doc)
        metadatas.append({
            "subject": t["subject"],
            "relation": t["relation"],
            "object": t["object"],
            "source_topic": t["source_topic"],
            "source": "wikidata",
        })
        ids.append(f"wd_{i}")

    # Embed in batches
    BATCH_SIZE = 100
    print(f"Embedding {len(documents)} Wikidata triples...")

    for start in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding"):
        end = min(start + BATCH_SIZE, len(documents))
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        # Embed batch
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_docs,
        )
        embeddings = [d.embedding for d in response.data]

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        time.sleep(1)  # rate limiting

    print(f"Indexed {collection.count()} triples in 'wikidata_fine' collection")


# ============================================================
# Step 4-5: Run comparison experiment
# ============================================================

def run_comparison(
    model_key: str = "qwen-27b",
    quick: bool = False,
) -> dict:
    """
    Run KG retrieval with EcoStatKG vs Wikidata index on the same benchmark.
    """
    from config import load_held_out_topics

    benchmark_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    queries = [item["question"] for item in benchmark]
    if quick:
        queries = queries[:20]

    exp_dir = RESULTS_DIR / "exp8_wikidata"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = GENERATOR_MODELS[model_key]
    api_client = create_client()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Held-out filter (for EcoStatKG only)
    held_out_topics = load_held_out_topics()
    held_out_filter = {"source_topic": {"$nin": held_out_topics}} if held_out_topics else None

    all_metrics = {}

    for kg_source in ["ecostats", "wikidata"]:
        collection_name = "hpti_fine" if kg_source == "ecostats" else "wikidata_fine"

        # Check collection exists
        try:
            coll = chroma_client.get_collection(collection_name)
            coll_count = coll.count()
        except Exception:
            print(f"Collection '{collection_name}' not found, skipping")
            continue

        label = f"kg_retrieval_{kg_source}_{model_key}"
        print(f"\nRunning: {label} ({coll_count} triples in index)")
        results_file = exp_dir / f"{label}.jsonl"

        wf = held_out_filter if kg_source == "ecostats" else None

        # Resume support
        answered = set()
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if "error" not in rec:
                                answered.add(rec["query"])
                        except (json.JSONDecodeError, KeyError):
                            pass

        remaining = [q for q in queries if q not in answered]
        if not remaining:
            print(f"  Already complete ({len(answered)} queries)")
        else:
            print(f"  Resuming: {len(answered)} done, {len(remaining)} remaining")

            call_count = 0
            window_start = time.time()

            with open(results_file, "a", encoding="utf-8") as f:
                for query in tqdm(remaining, desc=f"  {label}", leave=False):
                    call_count += 1
                    if call_count >= MAX_CALLS_PER_MINUTE:
                        elapsed = time.time() - window_start
                        if elapsed < 60.0:
                            time.sleep(60.0 - elapsed + 1)
                        call_count = 0
                        window_start = time.time()

                    try:
                        # Embed query
                        query_emb = embed_query(api_client, query)

                        # Retrieve from the target collection
                        query_kwargs = {
                            "query_embeddings": [query_emb],
                            "n_results": TOP_K_DEFAULT,
                        }
                        if wf:
                            query_kwargs["where"] = wf
                        results = coll.query(**query_kwargs)

                        docs = results["documents"][0]
                        context = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))

                        start = time.time()
                        response = chat_completion(
                            api_client, model,
                            messages=[
                                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                            ],
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                            top_p=TOP_P,
                        )
                        latency = (time.time() - start) * 1000

                        result = GenerationResult(
                            query=query, model=model_key,
                            method=f"kg_retrieval_{kg_source}",
                            answer=response.choices[0].message.content.strip(),
                            retrieved_triples=[{"document": d} for d in docs],
                            top_k=TOP_K_DEFAULT, latency_ms=latency,
                            token_usage={
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens,
                            },
                        )
                        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    except Exception as e:
                        log.error("Query failed: %s -- %s", query[:60], e)
                        f.write(json.dumps({
                            "query": query, "answer": "", "error": str(e),
                        }, ensure_ascii=False) + "\n")
                    f.flush()

        # Evaluate
        import importlib.util
        spec = importlib.util.spec_from_file_location("07", str(Path(__file__).parent / "07_evaluation.py"))
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)

        metrics = eval_mod.evaluate_results(results_file)
        all_metrics[label] = metrics
        eval_mod.print_metrics(metrics, label=label)

    # Summary
    summary_path = exp_dir / f"exp8_summary_{model_key}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("KG COMPARISON: EcoStatKG vs Wikidata")
    print(f"{'='*60}")
    for label, m in all_metrics.items():
        print(f"  {label}: NEM={m.get('NEM', 0):.4f}, NF1={m.get('NF1', 0):.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Wikidata comparison experiment")
    parser.add_argument("--extract", action="store_true", help="Extract + index Wikidata triples")
    parser.add_argument("--compare", action="store_true", help="Run comparison experiment")
    parser.add_argument("--all", action="store_true", help="Full pipeline")
    parser.add_argument("--model", type=str, default="qwen-27b")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.all or args.extract:
        triples = extract_wikidata_triples()
        index_wikidata_triples(triples)

    if args.all or args.compare:
        run_comparison(model_key=args.model, quick=args.quick)


if __name__ == "__main__":
    main()

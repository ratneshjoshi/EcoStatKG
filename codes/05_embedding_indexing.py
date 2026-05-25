"""
Step 04: Embedding & Indexing
=============================
Embeds KG triples using the API embedding model and stores them in ChromaDB
for dense retrieval. Supports multiple granularity levels for ablation.

Granularity levels:
    - fine:   One document per triple ("The HasGrowthRate of Solar Energy is 26.1% annually.")
    - medium: One document per triple, compact format ("Solar Energy - HasGrowthRate: 26.1% annually")
    - coarse: One document per topic (all triples concatenated)

Usage:
    python 04_embedding_indexing.py                      # build all 3 granularities
    python 04_embedding_indexing.py --granularity fine    # triple-level only
    python 04_embedding_indexing.py --verify              # run sample query after indexing
    python 04_embedding_indexing.py --include-structural  # also index structural triples
"""

import json
import argparse
import time
from pathlib import Path

from tqdm import tqdm
import chromadb
from openai import OpenAI

from config import (
    API_BASE_URL, API_KEY, HTTP_CLIENT,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    TRIPLES_DIR, CHROMA_DIR, REPORTS_DIR,
    MAX_CALLS_PER_MINUTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_client() -> OpenAI:
    """Create OpenAI-compatible client for embedding API."""
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        http_client=HTTP_CLIENT,
    )


def load_triples(include_structural: bool = False) -> list[dict]:
    """Load triples from JSONL files."""
    triples = []

    # Always load LLM triples
    kg_path = TRIPLES_DIR / "kg_triples.jsonl"
    with open(kg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                t["_source"] = "llm"
                triples.append(t)

    if include_structural:
        st_path = TRIPLES_DIR / "structural_triples.jsonl"
        with open(st_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    t["_source"] = "structural"
                    triples.append(t)

    return triples


def triple_to_text(triple: dict, level: str = "fine") -> str:
    """
    Convert a triple to natural language for embedding.

    Granularity levels:
        - fine:   "The HasGrowthRate of Solar Energy is 26.1% annually."
        - medium: "Solar Energy - HasGrowthRate: 26.1% annually"
        - coarse: "Solar Energy: HasGrowthRate=26.1% annually"
    """
    s = triple["subject"]
    r = triple["relation"]
    o = triple["object"]

    if level == "fine":
        return f"The {r} of {s} is {o}."
    elif level == "medium":
        return f"{s} - {r}: {o}"
    else:
        return f"{s}: {r}={o}"


def group_by_topic(triples: list[dict]) -> dict[str, list[dict]]:
    """Group triples by source topic for coarse-grained embedding."""
    groups: dict[str, list[dict]] = {}
    for t in triples:
        topic = t.get("source_topic", t["subject"])
        groups.setdefault(topic, []).append(t)
    return groups


# ---------------------------------------------------------------------------
# Embedding with retry + rate limiting
# ---------------------------------------------------------------------------

def embed_batch_with_retry(
    client: OpenAI,
    texts: list[str],
    batch_size: int = 20,
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embed a list of texts using the API.
    - 30 calls/minute rate limit
    - Exponential backoff on errors
    - Progress bar
    """
    all_embeddings: list[list[float]] = []
    call_count = 0
    window_start = time.time()
    n_batches = -(-len(texts) // batch_size)

    for i in tqdm(range(0, len(texts), batch_size), total=n_batches,
                  desc="Embedding", unit="batch"):
        batch = texts[i:i + batch_size]

        # Rate limiting
        call_count += 1
        if call_count >= MAX_CALLS_PER_MINUTE:
            elapsed = time.time() - window_start
            if elapsed < 60.0:
                wait = 60.0 - elapsed + 1.0
                tqdm.write(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)
            call_count = 0
            window_start = time.time()

        # Retry with backoff
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                for item in response.data:
                    all_embeddings.append(item.embedding)
                break
            except Exception as e:
                wait = 2 ** attempt * 2
                tqdm.write(f"  Error (attempt {attempt + 1}/{max_retries}): {e}")
                tqdm.write(f"  Retrying in {wait}s...")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    tqdm.write(f"  FAILED after {max_retries} attempts, skipping batch")
                    # Pad with zero vectors so indices stay aligned
                    for _ in batch:
                        all_embeddings.append([0.0] * EMBEDDING_DIMENSIONS)

    return all_embeddings


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    triples: list[dict],
    granularity: str = "fine",
    collection_name: str | None = None,
) -> int:
    """
    Build a ChromaDB collection from triples at a given granularity.
    Returns document count.
    """
    api_client = create_client()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    name = collection_name or f"hpti_{granularity}"

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name)
        print(f"  Deleted existing collection '{name}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare documents based on granularity
    if granularity in ("fine", "medium"):
        texts = [triple_to_text(t, level=granularity) for t in triples]
        ids = [f"t_{i}" for i in range(len(triples))]
        metadatas = [
            {
                "subject": t["subject"],
                "relation": t["relation"],
                "object": str(t["object"])[:500],
                "source_topic": t.get("source_topic", ""),
                "source": t.get("_source", "llm"),
            }
            for t in triples
        ]
    elif granularity == "coarse":
        groups = group_by_topic(triples)
        texts = []
        ids = []
        metadatas = []
        for i, (topic, topic_triples) in enumerate(groups.items()):
            combined = " | ".join(
                f"{t['relation']}: {t['object']}" for t in topic_triples
            )
            texts.append(f"{topic}: {combined}")
            ids.append(f"topic_{i}")
            metadatas.append({
                "subject": topic,
                "relation": "MULTI",
                "object": combined[:500],
                "source_topic": topic,
                "triple_count": str(len(topic_triples)),
            })
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    print(f"  Embedding {len(texts):,} documents at '{granularity}' granularity...")
    embeddings = embed_batch_with_retry(api_client, texts)

    # Insert into ChromaDB in batches (limit ~4000 per add)
    CHROMA_BATCH = 4000
    for i in tqdm(range(0, len(texts), CHROMA_BATCH),
                  desc="Inserting into ChromaDB", unit="batch"):
        end = min(i + CHROMA_BATCH, len(texts))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    count = collection.count()
    print(f"  Collection '{name}': {count:,} documents indexed")
    return count


def verify_retrieval(queries: list[str] | None = None):
    """Quick sanity check: retrieve top-5 for sample queries."""
    if queries is None:
        queries = [
            "What is the annual growth rate of solar energy?",
            "How much CO2 does the transportation sector emit?",
            "What percentage of electricity comes from renewable sources?",
        ]

    api_client = create_client()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for query in queries:
        response = api_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        query_embedding = response.data[0].embedding

        print(f"\nQuery: {query}")
        for level in ("fine", "medium", "coarse"):
            name = f"hpti_{level}"
            try:
                collection = chroma_client.get_collection(name)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                )
                print(f"\n  --- {level.upper()} (top 5) ---")
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    print(f"    [{meta['relation']}] (dist={dist:.4f}) {doc[:120]}")
            except Exception as e:
                print(f"  {level}: not found ({e})")
        time.sleep(2)  # rate limit between queries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed and index KG triples")
    parser.add_argument(
        "--granularity",
        choices=["fine", "medium", "coarse", "all"],
        default="all",
        help="Granularity level for embedding (default: all)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification queries after indexing",
    )
    parser.add_argument(
        "--include-structural", action="store_true",
        help="Also index structural triples (RelatedTo, BelongsToCategory, Synonym)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 04: Embedding & Indexing")
    print("=" * 60)

    triples = load_triples(include_structural=args.include_structural)
    print(f"\nLoaded {len(triples):,} triples")
    if args.include_structural:
        llm_count = sum(1 for t in triples if t.get("_source") == "llm")
        st_count = len(triples) - llm_count
        print(f"  LLM: {llm_count:,}, Structural: {st_count:,}")

    levels = ["fine", "medium", "coarse"] if args.granularity == "all" else [args.granularity]
    results = {}

    start_time = time.time()
    for level in levels:
        print(f"\n{'=' * 40}")
        print(f"Building '{level}' index...")
        print(f"{'=' * 40}")
        count = build_index(triples, granularity=level)
        results[level] = count
        print()

    elapsed = time.time() - start_time

    # Summary
    print("=" * 60)
    print("Indexing Summary")
    print("=" * 60)
    for level, count in results.items():
        print(f"  {level}: {count:,} documents")
    print(f"  Total time: {elapsed / 60:.1f} min")

    # Save report
    report = {
        "triples_loaded": len(triples),
        "include_structural": args.include_structural,
        "collections": results,
        "elapsed_seconds": round(elapsed, 1),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chroma_dir": str(CHROMA_DIR),
    }
    report_path = REPORTS_DIR / "embedding_indexing_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Report saved to {report_path}")

    if args.verify:
        print("\n" + "=" * 60)
        print("Verification Queries")
        print("=" * 60)
        verify_retrieval()

    print("\n" + "=" * 60)
    print("Embedding & Indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

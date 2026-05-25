"""
Step 6: Retrieval Methods
==========================
Implements all retrieval/generation methods for EcoStatKG evaluation:
  1. Zero-shot (no retrieval — pure parametric knowledge)
  2. Standard RAG (text-chunk retrieval from raw Wikipedia text)
  3. GraphRAG (2-hop subgraph retrieval from ChromaDB)
  4. KG Retrieval (cosine retrieval + optional reranker from ChromaDB)
  5. CoNLI (Chain-of-NLI: generate → verify via NLI → revise contradictions)
  6. CoVe (Chain-of-Verification: generate → extract claims → self-verify → refine)

All methods use the same API backend and models for fair comparison.

Usage:
    python 06_baselines.py --method zero_shot --model qwen-27b --query "..."
    python 06_baselines.py --method kg_retrieval --model mistral-small-24b --query "..."
"""

import json
import time
from dataclasses import dataclass, field

import chromadb
from tqdm import tqdm
from openai import OpenAI

from config import (
    API_BASE_URL, API_KEY, HTTP_CLIENT,
    EMBEDDING_MODEL, RERANKER_MODEL, GENERATOR_MODELS,
    CHROMA_DIR, CORPUS_DIR,
    GROUNDED_SYSTEM_PROMPT,
    ZERO_SHOT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT,
    TEMPERATURE, MAX_TOKENS, TOP_P,
    TOP_K_DEFAULT, MAX_CALLS_PER_MINUTE,
)
from utils import (
    GenerationResult, RetrievalResult,
    embed_query, create_client, chat_completion,
)


# ============================================================
# Baseline 1: Zero-Shot (no retrieval)
# ============================================================

def zero_shot(
    query: str,
    model_key: str = "qwen-27b",
    temperature: float = TEMPERATURE,
) -> GenerationResult:
    """Pure parametric generation — no external knowledge."""
    client = create_client()
    model = GENERATOR_MODELS[model_key]

    start = time.time()
    response = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
    latency = (time.time() - start) * 1000

    return GenerationResult(
        query=query,
        model=model_key,
        method="zero_shot",
        answer=response.choices[0].message.content.strip(),
        latency_ms=latency,
        token_usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    )


# ============================================================
# Baseline 2: Standard RAG (text-chunk retrieval)
# ============================================================

def _ensure_rag_collection(client: OpenAI) -> chromadb.Collection:
    """
    Build a text-chunk RAG index from the raw Wikipedia corpus.
    Chunks are ~256 tokens (split by paragraph).
    This is only built once and reused.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        return chroma_client.get_collection("rag_chunks")
    except Exception:
        pass  # Collection doesn't exist yet

    # Load raw corpus
    corpus_file = CORPUS_DIR / "statistical_corpus.json"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Chunk into ~256-token paragraphs
    chunks = []
    metas = []
    for topic, data in corpus.items():
        text = data.get("full_text", "")
        paragraphs = text.split("\n\n") if text else data.get("sentences", [])
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) > 50:  # Skip tiny fragments
                chunks.append(para[:1024])  # Cap chunk size
                metas.append({"topic": topic, "chunk_idx": str(i)})

    if not chunks:
        raise ValueError("No chunks found in corpus. Run 01_corpus_extraction.py first.")

    # Embed all chunks
    print(f"Building RAG index: {len(chunks)} chunks to embed...")
    all_embeddings = []
    batch_size = 20
    call_count = 0
    window_start = time.time()
    n_batches = -(-len(chunks) // batch_size)
    for i in tqdm(range(0, len(chunks), batch_size), total=n_batches, desc="RAG embedding", unit="batch"):
        batch = chunks[i:i + batch_size]
        call_count += 1
        if call_count >= MAX_CALLS_PER_MINUTE:
            elapsed = time.time() - window_start
            if elapsed < 60.0:
                wait = 60.0 - elapsed
                tqdm.write(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)
            call_count = 0
            window_start = time.time()

        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])

    # Create collection
    collection = chroma_client.create_collection(
        "rag_chunks", metadata={"hnsw:space": "cosine"}
    )
    batch_sz = 4000
    n_insert = -(-len(chunks) // batch_sz)
    for i in tqdm(range(0, len(chunks), batch_sz), total=n_insert, desc="ChromaDB insert", unit="batch"):
        end = min(i + batch_sz, len(chunks))
        collection.add(
            ids=[f"chunk_{j}" for j in range(i, end)],
            embeddings=all_embeddings[i:end],
            documents=chunks[i:end],
            metadatas=metas[i:end],
        )

    return collection


def _translate_filter_for_rag(where_filter: dict | None) -> dict | None:
    """Translate source_topic filter to topic for rag_chunks collection."""
    if where_filter is None:
        return None
    if "source_topic" in where_filter:
        return {"topic": where_filter["source_topic"]}
    return where_filter


def standard_rag(
    query: str,
    model_key: str = "qwen-27b",
    top_k: int = TOP_K_DEFAULT,
    where_filter: dict | None = None,
) -> GenerationResult:
    """Standard RAG: retrieve text chunks, concatenate, generate."""
    client = create_client()
    model = GENERATOR_MODELS[model_key]

    start = time.time()

    collection = _ensure_rag_collection(client)
    query_emb = embed_query(client, query)

    query_kwargs = {"query_embeddings": [query_emb], "n_results": top_k}
    rag_filter = _translate_filter_for_rag(where_filter)
    if rag_filter:
        query_kwargs["where"] = rag_filter
    results = collection.query(**query_kwargs)
    context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(results["documents"][0]))

    response = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
    latency = (time.time() - start) * 1000

    return GenerationResult(
        query=query,
        model=model_key,
        method="standard_rag",
        answer=response.choices[0].message.content.strip(),
        retrieved_triples=[{"document": d} for d in results["documents"][0]],
        top_k=top_k,
        latency_ms=latency,
        token_usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    )


# ============================================================
# Baseline 3: GraphRAG (2-hop subgraph retrieval)
# ============================================================

GRAPH_RAG_TOP_K = 50

def graph_rag(
    query: str,
    model_key: str = "qwen-27b",
    top_k: int = GRAPH_RAG_TOP_K,
    where_filter: dict | None = None,
    max_distance: float | None = None,
) -> GenerationResult:
    """
    GraphRAG: 2-hop subgraph retrieval from the knowledge graph.

    Hop 1: Cosine-similarity retrieval of seed triples.
    Hop 2: Entity-based expansion — retrieve all triples sharing
            subjects with hop-1 results (graph neighbor traversal).
    Results are deduplicated and capped at top_k.
    """
    client = create_client()
    model = GENERATOR_MODELS[model_key]
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    start = time.time()

    try:
        collection = chroma_client.get_collection("hpti_fine")
    except Exception:
        raise ValueError("KG index not found. Run 04_embedding_indexing.py first.")

    # --- Hop 1: Embedding-based seed retrieval ---
    hop1_k = max(top_k // 2, 10)
    query_emb = embed_query(client, query)
    query_kwargs = {"query_embeddings": [query_emb], "n_results": hop1_k}
    if where_filter:
        query_kwargs["where"] = where_filter
    hop1_results = collection.query(**query_kwargs)

    hop1_ids = list(hop1_results["ids"][0])
    hop1_docs = list(hop1_results["documents"][0])
    hop1_metas = list(hop1_results["metadatas"][0])
    hop1_dists = list(hop1_results["distances"][0])

    # Dynamic threshold: filter hop-1 seeds by distance
    if max_distance is not None:
        keep = [(i, d, doc, meta) for i, (d, doc, meta)
                in enumerate(zip(hop1_dists, hop1_docs, hop1_metas))
                if d <= max_distance]
        if keep:
            _, hop1_dists, hop1_docs, hop1_metas = zip(*keep)
            hop1_ids_set = {hop1_ids[k[0]] for k in keep}
            hop1_docs, hop1_metas = list(hop1_docs), list(hop1_metas)
        else:
            hop1_ids_set = set()
            hop1_docs, hop1_metas = [], []
    else:
        hop1_ids_set = set(hop1_ids)

    # If all hop-1 triples were filtered, fall back to zero-shot
    if not hop1_docs:
        response = chat_completion(
            client, model,
            messages=[
                {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        latency = (time.time() - start) * 1000
        return GenerationResult(
            query=query,
            model=model_key,
            method="graph_rag",
            answer=response.choices[0].message.content.strip(),
            retrieved_triples=[],
            top_k=top_k,
            latency_ms=latency,
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    # --- Hop 2: Entity-based neighbor expansion ---
    subjects = list({m["subject"] for m in hop1_metas if m.get("subject")})

    hop2_docs = []
    hop2_ids = set()
    if subjects:
        # Build subject filter (ChromaDB $or needs ≥2 elements)
        if len(subjects) == 1:
            subj_filter = {"subject": subjects[0]}
        else:
            subj_filter = {"$or": [{"subject": s} for s in subjects]}

        # Combine with held-out filter if present
        if where_filter:
            combined_filter = {"$and": [where_filter, subj_filter]}
        else:
            combined_filter = subj_filter

        hop2_results = collection.get(
            where=combined_filter,
            limit=top_k * 3,
        )

        for doc_id, doc in zip(hop2_results["ids"], hop2_results["documents"]):
            if doc_id not in hop1_ids_set and doc_id not in hop2_ids:
                hop2_docs.append(doc)
                hop2_ids.add(doc_id)

    # --- Combine: hop-1 first (cosine-ranked), then hop-2 (expanded) ---
    seen = set()
    unique_docs = []
    for d in hop1_docs + hop2_docs:
        if d not in seen:
            seen.add(d)
            unique_docs.append(d)

    final_docs = unique_docs[:top_k]
    context = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(final_docs))

    response = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
    latency = (time.time() - start) * 1000

    return GenerationResult(
        query=query,
        model=model_key,
        method="graph_rag",
        answer=response.choices[0].message.content.strip(),
        retrieved_triples=[{"document": d} for d in final_docs],
        top_k=top_k,
        latency_ms=latency,
        token_usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    )


# ============================================================
# Baseline 4: CoNLI (Chain-of-NLI)
# ============================================================

_CONLI_VERIFY_PROMPT = """You are a fact-checking assistant. Given a claim and a set of reference facts, determine if each claim is:
- SUPPORTED: The facts confirm this claim
- CONTRADICTED: The facts contradict this claim  
- NOT_ENOUGH_INFO: The facts neither confirm nor contradict

Reference facts:
{facts}

Claim: {claim}

For each numerical statement in the claim, output a JSON object:
{{"verdict": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO", "corrected_claim": "<corrected version if CONTRADICTED, else null>"}}

Output ONLY the JSON object."""

_CONLI_REVISE_PROMPT = """Revise the following answer to correct any contradicted claims.

Original answer:
{original}

Corrections needed:
{corrections}

Provide the corrected answer, keeping all supported claims unchanged."""


def conli(
    query: str,
    model_key: str = "qwen-27b",
    top_k: int = TOP_K_DEFAULT,
    where_filter: dict | None = None,
) -> GenerationResult:
    """
    CoNLI: Generate → Retrieve evidence → NLI-verify → Revise.
    Multi-pass approach (2-3 API calls per query).
    """
    client = create_client()
    model = GENERATOR_MODELS[model_key]
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    start = time.time()

    # Step 1: Initial generation (zero-shot)
    initial = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    initial_answer = initial.choices[0].message.content.strip()
    total_tokens = initial.usage.total_tokens

    # Step 2: Retrieve reference facts
    query_emb = embed_query(client, query)
    try:
        collection = chroma_client.get_collection("hpti_fine")
        query_kwargs = {"query_embeddings": [query_emb], "n_results": top_k}
        if where_filter:
            query_kwargs["where"] = where_filter
        results = collection.query(**query_kwargs)
        facts = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(results["documents"][0]))
    except Exception:
        facts = "(No reference facts available)"

    # Step 3: NLI verification
    verify_response = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": "You are a precise NLI fact-checker."},
            {"role": "user", "content": _CONLI_VERIFY_PROMPT.format(
                facts=facts, claim=initial_answer
            )},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    verification = verify_response.choices[0].message.content.strip()
    total_tokens += verify_response.usage.total_tokens

    # Step 4: Revise if needed
    if "CONTRADICTED" in verification:
        revise_response = chat_completion(
            client, model,
            messages=[
                {"role": "system", "content": "You are a precise editor."},
                {"role": "user", "content": _CONLI_REVISE_PROMPT.format(
                    original=initial_answer, corrections=verification
                )},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        final_answer = revise_response.choices[0].message.content.strip()
        total_tokens += revise_response.usage.total_tokens
    else:
        final_answer = initial_answer

    latency = (time.time() - start) * 1000

    return GenerationResult(
        query=query,
        model=model_key,
        method="conli",
        answer=final_answer,
        top_k=top_k,
        latency_ms=latency,
        token_usage={"total_tokens": total_tokens},
    )


# ============================================================
# Baseline 5: CoVe (Chain-of-Verification)
# ============================================================

_COVE_EXTRACT_PROMPT = """Extract all factual claims from the following answer as a numbered list.
Focus on claims containing specific numbers, dates, percentages, or statistics.

Answer: {answer}

List each factual claim on its own line, numbered 1., 2., etc."""

_COVE_VERIFY_PROMPT = """For each factual claim below, independently verify whether it is likely correct.
Provide your best answer to each verification question.

Claims to verify:
{claims}

For each claim, respond with:
CLAIM N: [CORRECT/INCORRECT/UNCERTAIN] - [your brief justification or corrected value]"""

_COVE_REFINE_PROMPT = """Given the original answer and the verification results, produce a refined answer
that corrects any incorrect claims while keeping correct ones.

Original answer: {original}

Verification results:
{verifications}

Provide the refined, corrected answer:"""


def cove(
    query: str,
    model_key: str = "qwen-27b",
) -> GenerationResult:
    """
    CoVe: Generate → Extract claims → Self-verify → Refine.
    Multi-pass approach (3-4 API calls per query).
    """
    client = create_client()
    model = GENERATOR_MODELS[model_key]

    start = time.time()
    total_tokens = 0

    # Step 1: Initial generation
    initial = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    initial_answer = initial.choices[0].message.content.strip()
    total_tokens += initial.usage.total_tokens

    # Step 2: Extract factual claims
    extract = chat_completion(
        client, model,
        messages=[
            {"role": "user", "content": _COVE_EXTRACT_PROMPT.format(answer=initial_answer)},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    claims = extract.choices[0].message.content.strip()
    total_tokens += extract.usage.total_tokens

    # Step 3: Self-verify claims
    verify = chat_completion(
        client, model,
        messages=[
            {"role": "user", "content": _COVE_VERIFY_PROMPT.format(claims=claims)},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    verifications = verify.choices[0].message.content.strip()
    total_tokens += verify.usage.total_tokens

    # Step 4: Refine
    if "INCORRECT" in verifications:
        refine = chat_completion(
            client, model,
            messages=[
                {"role": "user", "content": _COVE_REFINE_PROMPT.format(
                    original=initial_answer, verifications=verifications
                )},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        final_answer = refine.choices[0].message.content.strip()
        total_tokens += refine.usage.total_tokens
    else:
        final_answer = initial_answer

    latency = (time.time() - start) * 1000

    return GenerationResult(
        query=query,
        model=model_key,
        method="cove",
        answer=final_answer,
        latency_ms=latency,
        token_usage={"total_tokens": total_tokens},
    )


# ============================================================
# Method 6: KG Retrieval (cosine + optional reranker)
# ============================================================

def retrieve_triples(
    query_embedding: list[float],
    collection_name: str = "hpti_fine",
    top_k: int = TOP_K_DEFAULT,
    where_filter: dict | None = None,
    max_distance: float | None = None,
) -> RetrievalResult:
    """
    Retrieve top-k triples from ChromaDB by cosine similarity.

    Args:
        where_filter: Optional ChromaDB metadata filter, e.g.
                      {"source_topic": {"$nin": [...]}} for held-out evaluation.
        max_distance: If set, discard any triple with cosine distance > this
                      threshold.  When all triples exceed the threshold the
                      result is empty, causing the generator to fall back to
                      parametric (zero-shot-like) behaviour.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(collection_name)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Dynamic similarity threshold: drop low-quality retrievals
    if max_distance is not None:
        keep = [(d, doc, meta) for d, doc, meta in zip(distances, documents, metadatas)
                if d <= max_distance]
        if keep:
            distances, documents, metadatas = zip(*keep)
            distances, documents, metadatas = list(distances), list(documents), list(metadatas)
        else:
            distances, documents, metadatas = [], [], []

    return RetrievalResult(
        documents=documents,
        metadatas=metadatas,
        distances=distances,
    )


def rerank_triples(
    client: OpenAI,
    query: str,
    retrieval: RetrievalResult,
    top_n: int | None = None,
) -> RetrievalResult:
    """
    Rerank retrieved triples using the reranker API.
    Falls back to original ranking if reranker is unavailable.
    """
    if top_n is None:
        top_n = len(retrieval.documents)

    try:
        import httpx as _httpx

        response = _httpx.post(
            f"{API_BASE_URL.replace('/v1', '')}/v1/rerank",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": RERANKER_MODEL,
                "query": query,
                "documents": retrieval.documents,
                "top_n": top_n,
            },
            verify=False,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        ranked_indices = [r["index"] for r in data["results"]]
        return RetrievalResult(
            documents=[retrieval.documents[i] for i in ranked_indices[:top_n]],
            metadatas=[retrieval.metadatas[i] for i in ranked_indices[:top_n]],
            distances=[r["relevance_score"] for r in data["results"][:top_n]],
        )

    except Exception as e:
        print(f"  Reranker unavailable ({e}), using cosine ranking")
        return retrieval


def inject_and_generate(
    client: OpenAI,
    query: str,
    retrieved: RetrievalResult,
    model: str,
    system_prompt: str = GROUNDED_SYSTEM_PROMPT,
    temperature: float = TEMPERATURE,
) -> tuple[str, dict]:
    """
    Inject retrieved triples into the prompt and generate an answer.
    Returns (answer_text, token_usage).
    """
    numbered_facts = retrieved.to_numbered_facts()

    user_message = f"""Context:
{numbered_facts}

Question: {query}"""

    response = chat_completion(
        client, model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, usage


def kg_retrieval(
    query: str,
    model_key: str = "qwen-27b",
    top_k: int = TOP_K_DEFAULT,
    granularity: str = "fine",
    use_reranker: bool = True,
    where_filter: dict | None = None,
    temperature: float = TEMPERATURE,
    max_distance: float | None = None,
) -> GenerationResult:
    """
    KG Retrieval pipeline: embed → retrieve → rerank → inject → generate.

    Args:
        query: Natural language question
        model_key: Key from GENERATOR_MODELS
        top_k: Number of triples to retrieve
        granularity: 'fine', 'medium', or 'coarse'
        use_reranker: Whether to apply reranking
        where_filter: Optional ChromaDB metadata filter for held-out evaluation
        max_distance: If set, discard triples with cosine distance above this
                      threshold before injection.

    Returns:
        GenerationResult with answer and metadata
    """
    client = create_client()
    model = GENERATOR_MODELS[model_key]
    collection_name = f"hpti_{granularity}"

    start_time = time.time()

    query_embedding = embed_query(client, query)
    retrieved = retrieve_triples(query_embedding, collection_name, top_k,
                                 where_filter=where_filter,
                                 max_distance=max_distance)

    if use_reranker and len(retrieved.documents) > 1:
        retrieved = rerank_triples(client, query, retrieved)

    # If all triples were filtered out, fall back to zero-shot
    if not retrieved.documents:
        response = chat_completion(
            client, model,
            messages=[
                {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    else:
        answer, usage = inject_and_generate(client, query, retrieved, model,
                                            temperature=temperature)

    latency_ms = (time.time() - start_time) * 1000

    return GenerationResult(
        query=query,
        model=model_key,
        method="kg_retrieval",
        answer=answer,
        retrieved_triples=[
            {"document": doc, **meta}
            for doc, meta in zip(retrieved.documents, retrieved.metadatas)
        ],
        top_k=top_k,
        granularity=granularity,
        latency_ms=latency_ms,
        token_usage=usage,
    )


# Backward-compat alias (existing results use "hpti" method label)
hpti_pipeline = kg_retrieval


# ============================================================
# Dispatcher
# ============================================================

BASELINES = {
    "zero_shot": zero_shot,
    "standard_rag": standard_rag,
    "graph_rag": graph_rag,
    "kg_retrieval": kg_retrieval,
    "conli": conli,
    "cove": cove,
}


def run_baseline(
    method: str,
    query: str,
    model_key: str = "qwen-27b",
    **kwargs,
) -> GenerationResult:
    """Run any baseline by name."""
    if method not in BASELINES:
        raise ValueError(f"Unknown baseline '{method}'. Choose from: {list(BASELINES.keys())}")
    return BASELINES[method](query, model_key, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline methods")
    parser.add_argument("--method", type=str, required=True, choices=list(BASELINES.keys()))
    parser.add_argument("--model", type=str, default="qwen-27b", choices=list(GENERATOR_MODELS.keys()))
    parser.add_argument("--query", type=str, default="What is the global solar energy capacity?")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    kwargs = {}
    if args.method in ("standard_rag", "graph_rag", "conli"):
        kwargs["top_k"] = args.top_k
    if args.method == "kg_retrieval":
        kwargs["top_k"] = args.top_k
        kwargs["use_reranker"] = not args.no_rerank

    result = run_baseline(args.method, args.query, args.model, **kwargs)

    print(f"\n=== {args.method.upper()} Result ===")
    print(f"Model:   {result.model}")
    print(f"Latency: {result.latency_ms:.0f} ms")
    print(f"Tokens:  {result.token_usage}")
    print(f"\nAnswer:\n{result.answer}")

"""
Graph-Based RAG Baselines using EcoStatKG
==========================================
Insert our existing KG (49,510 LLM triples + 306K structural triples)
into LightRAG and MS GraphRAG, then run benchmark queries.

This tests: same KG, different retrieval strategies.

Usage:
    # Step 1: Insert our KG into LightRAG (embeddings only, no LLM extraction)
    python 13_graph_baselines.py insert-lightrag

    # Step 2: Insert our KG into MS GraphRAG format
    python 13_graph_baselines.py insert-graphrag

    # Step 3: Run benchmark queries
    python 13_graph_baselines.py query --method lightrag --model devstral-24b
    python 13_graph_baselines.py query --method ms_graphrag --model devstral-24b
"""

import os
import ssl
import sys
import json
import time
import asyncio
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

from config import (
    API_BASE_URL, API_KEY, CORPUS_DIR, DATA_DIR, RESULTS_DIR,
    ECOSTATS_DIR, GENERATOR_MODELS, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    GROUNDED_SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS, TOP_P,
)
from utils import create_client, chat_completion

log = logging.getLogger("ecostats.graph_baselines")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LIGHTRAG_DIR = DATA_DIR / "lightrag_kg"
MS_GRAPHRAG_DIR = DATA_DIR / "ms_graphrag_kg"

API_KEYS = [
    os.getenv("OPENAI_API_KEY_1"),
    os.getenv("OPENAI_API_KEY_2"),
    os.getenv("OPENAI_API_KEY_3"),
    os.getenv("OPENAI_API_KEY_4"),
    os.getenv("OPENAI_API_KEY_5"),
    os.getenv("OPENAI_API_KEY_6"),
]
API_KEYS = [k for k in API_KEYS if k]
# Also set OPENAI_API_KEY for libraries that need it
if not os.getenv("OPENAI_API_KEY") and API_KEYS:
    os.environ["OPENAI_API_KEY"] = API_KEYS[0]


# =============================================================================
# Load our KG triples
# =============================================================================

def load_triples():
    """Load LLM-extracted triples (the ones in ChromaDB hpti_fine)."""
    kg_path = DATA_DIR / "triples" / "kg_triples.json"
    with open(kg_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    log.info(f"Loaded {len(triples)} LLM triples")
    return triples


def load_structural_triples():
    """Load structural triples (Wikipedia hyperlinks etc.)."""
    st_path = DATA_DIR / "triples" / "structural_triples.json"
    with open(st_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    log.info(f"Loaded {len(triples)} structural triples")
    return triples


# =============================================================================
# Convert our triples to LightRAG format
# =============================================================================

def triples_to_lightrag_format(triples, structural_triples=None):
    """
    Convert our (subject, relation, object) triples into LightRAG's
    custom_kg format: {chunks, entities, relationships}.

    LightRAG expects:
    - chunks: [{content, source_id, file_path}]
    - entities: [{entity_name, entity_type, description, source_id, file_path}]
    - relationships: [{src_id, tgt_id, description, keywords, weight, source_id, file_path}]
    """
    chunks = []
    entities = {}      # entity_name -> entity_data
    relationships = []

    # Group triples by source_topic to create chunk documents
    topic_triples = defaultdict(list)
    for t in triples:
        topic_triples[t["source_topic"]].append(t)

    # Create chunks (one per topic, containing all triples as text)
    for topic, topic_ts in topic_triples.items():
        text_lines = []
        for t in topic_ts:
            text_lines.append(
                f"The {t['relation']} of {t['subject']} is {t['object']}."
            )
        chunk_content = f"Topic: {topic}\n" + "\n".join(text_lines)
        source_id = f"topic-{topic}"
        chunks.append({
            "content": chunk_content,
            "source_id": source_id,
            "file_path": f"ecostats/{topic}",
        })

    log.info(f"Created {len(chunks)} chunks from {len(triples)} triples")

    # Create entities: subjects and objects are entities
    for t in triples:
        subj = t["subject"]
        if subj not in entities:
            entities[subj] = {
                "entity_name": subj,
                "entity_type": "statistic",
                "description": f"{subj} ({t['relation']}: {t['object']})",
                "source_id": f"topic-{t['source_topic']}",
                "file_path": f"ecostats/{t['source_topic']}",
            }
        else:
            # Append to description for richer context
            existing = entities[subj]
            if len(existing["description"]) < 500:
                existing["description"] += f"; {t['relation']}: {t['object']}"

    log.info(f"Created {len(entities)} unique entities from LLM triples")

    # Create relationships from LLM triples
    for t in triples:
        relationships.append({
            "src_id": t["subject"],
            "tgt_id": t["source_topic"],  # Link subject to its topic
            "description": f"{t['subject']} {t['relation']} {t['object']}",
            "keywords": f"{t['relation']} {t['source_topic']}",
            "weight": 1.0,
            "source_id": f"topic-{t['source_topic']}",
            "file_path": f"ecostats/{t['source_topic']}",
        })

    # Add structural triples as additional relationships
    if structural_triples:
        for t in structural_triples:
            subj = t["subject"]
            obj = t["object"]

            if subj not in entities:
                entities[subj] = {
                    "entity_name": subj,
                    "entity_type": "topic",
                    "description": f"{subj} (environmental topic)",
                    "source_id": f"topic-{t['source_topic']}",
                    "file_path": f"ecostats/{t['source_topic']}",
                }
            if obj not in entities:
                entities[obj] = {
                    "entity_name": obj,
                    "entity_type": "topic",
                    "description": f"{obj} (related topic)",
                    "source_id": f"topic-{t['source_topic']}",
                    "file_path": f"ecostats/{t['source_topic']}",
                }

            relationships.append({
                "src_id": subj,
                "tgt_id": obj,
                "description": f"{subj} is related to {obj}",
                "keywords": f"{t['relation']} {t['source_topic']}",
                "weight": 0.5,
                "source_id": f"topic-{t['source_topic']}",
                "file_path": f"ecostats/{t['source_topic']}",
            })
        log.info(f"Added {len(structural_triples)} structural relationships")

    log.info(f"Total: {len(chunks)} chunks, {len(entities)} entities, {len(relationships)} relationships")

    return {
        "chunks": chunks,
        "entities": list(entities.values()),
        "relationships": relationships,
    }


# =============================================================================
# LightRAG: Insert and Query
# =============================================================================

def get_lightrag_instance(api_key=None):
    """Create a LightRAG instance for the configured LLM API.
    Uses round-robin across all API keys for embeddings (6x throughput).
    """
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    working_dir = str(LIGHTRAG_DIR)
    os.makedirs(working_dir, exist_ok=True)

    if api_key is None:
        api_key = API_KEYS[0]

    # Round-robin counter for embedding calls across all keys
    _key_counter = {"idx": 0}
    _num_keys = len(API_KEYS)

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        kwargs.pop("response_format", None)
        return await openai_complete_if_cache(
            "Mistral-Small-24B-Instruct-2501-FP8-dynamic",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=API_BASE_URL,
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_DIMENSIONS,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        """Embed texts using round-robin across all API keys for 6x throughput.
        Each API call batches up to 30 texts; 6 calls fire in parallel = 180 texts/wave.
        """
        batch_size = 30  # texts per API call (API supports up to ~50)
        all_results = [None] * len(texts)  # preserve order

        # Create (index, sub_batch, api_key) tasks
        tasks = []
        for i in range(0, len(texts), batch_size):
            sub = texts[i:i+batch_size]
            key_idx = _key_counter["idx"] % _num_keys
            _key_counter["idx"] += 1
            tasks.append((i, sub, API_KEYS[key_idx]))

        # Fire in waves of _num_keys parallel calls
        for wave_start in range(0, len(tasks), _num_keys):
            wave = tasks[wave_start:wave_start + _num_keys]
            coros = [
                openai_embed.func(sub, model=EMBEDDING_MODEL,
                                  api_key=key, base_url=API_BASE_URL)
                for _, sub, key in wave
            ]
            results = await asyncio.gather(*coros)
            for (orig_i, sub, _), emb in zip(wave, results):
                # emb is (len(sub), dim) array
                for j in range(len(sub)):
                    all_results[orig_i + j] = emb[j]
            if wave_start + _num_keys < len(tasks):
                await asyncio.sleep(0.5)  # Light pause — 6 keys = 180 req/min budget
        return np.array(all_results)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name="Mistral-Small-24B-Instruct-2501-FP8-dynamic",
        embedding_func=embedding_func,
        embedding_batch_num=180,  # texts per LightRAG internal batch (6 keys * 30 texts)
        embedding_func_max_async=1,  # we parallelize inside embedding_func
        llm_model_max_async=2,
        addon_params={
            "language": "English",
            "entity_types": ["organization", "location", "statistic", "resource", "event", "policy"],
        },
    )
    return rag


async def insert_kg_to_lightrag(include_structural=False):
    """Insert our KG triples into LightRAG using insert_custom_kg."""
    log.info("=== Inserting EcoStatKG into LightRAG ===")

    triples = load_triples()
    structural = load_structural_triples() if include_structural else None

    custom_kg = triples_to_lightrag_format(triples, structural)

    rag = get_lightrag_instance()
    await rag.initialize_storages()

    try:
        total_ents = len(custom_kg["entities"])
        total_rels = len(custom_kg["relationships"])
        total_chunks = len(custom_kg["chunks"])
        log.info(f"Inserting {total_chunks} chunks, {total_ents} entities, {total_rels} relationships")

        # Insert in larger batches — 6 keys with batch=30 each = 180 texts/wave
        batch_size = 2000
        entity_batches = [custom_kg["entities"][i:i+batch_size]
                         for i in range(0, total_ents, batch_size)]
        rel_batch_size = 3000
        rel_batches = [custom_kg["relationships"][i:i+rel_batch_size]
                      for i in range(0, total_rels, rel_batch_size)]
        chunk_batch_size = 1000
        chunk_batches = [custom_kg["chunks"][i:i+chunk_batch_size]
                        for i in range(0, total_chunks, chunk_batch_size)]

        log.info(f"Split into {len(chunk_batches)} chunk batches, "
                f"{len(entity_batches)} entity batches, {len(rel_batches)} rel batches")

        for i, chunk_batch in enumerate(chunk_batches):
            log.info(f"Chunk batch {i+1}/{len(chunk_batches)} ({len(chunk_batch)} chunks)")
            mini_kg = {"chunks": chunk_batch, "entities": [], "relationships": []}
            await rag.ainsert_custom_kg(mini_kg)
            await asyncio.sleep(1)

        for i, ent_batch in enumerate(entity_batches):
            log.info(f"Entity batch {i+1}/{len(entity_batches)} ({len(ent_batch)} entities)")
            mini_kg = {"chunks": [], "entities": ent_batch, "relationships": []}
            await rag.ainsert_custom_kg(mini_kg)
            await asyncio.sleep(1)

        for i, rel_batch in enumerate(rel_batches):
            log.info(f"Relation batch {i+1}/{len(rel_batches)} ({len(rel_batch)} rels)")
            mini_kg = {"chunks": [], "entities": [], "relationships": rel_batch}
            await rag.ainsert_custom_kg(mini_kg)
            await asyncio.sleep(1)

        log.info("LightRAG KG insertion complete!")

    finally:
        await rag.finalize_storages()


async def query_lightrag(query: str, model_key: str = "mistral-small-24b",
                         mode: str = "hybrid", rag_instance=None):
    """Query LightRAG and generate answer using our model.
    If rag_instance is provided, reuse it (avoids reloading 2.6GB storage each time).
    """
    from lightrag import QueryParam

    owns_rag = rag_instance is None
    rag = rag_instance or get_lightrag_instance()
    if owns_rag:
        await rag.initialize_storages()

    try:
        start = time.time()

        param = QueryParam(
            mode=mode,
            only_need_context=True,
            top_k=10,
        )
        context = await rag.aquery(query, param=param)
        retrieval_ms = (time.time() - start) * 1000

        client = create_client()
        model = GENERATOR_MODELS[model_key]

        gen_start = time.time()
        response = chat_completion(
            client, model,
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        total_ms = (time.time() - start) * 1000

        return {
            "query": query,
            "model": model_key,
            "method": "lightrag",
            "answer": response.choices[0].message.content.strip(),
            "context": context[:2000] if context else "",
            "latency_ms": total_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": (time.time() - gen_start) * 1000,
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
    finally:
        if owns_rag:
            await rag.finalize_storages()


# =============================================================================
# MS GraphRAG: Convert KG to community structure and query
# =============================================================================

def setup_ms_graphrag():
    """
    Convert our KG into a community-based structure following MS GraphRAG's
    core algorithm: entity graph → community detection → community summaries.
    """
    import pandas as pd
    import networkx as nx

    os.makedirs(str(MS_GRAPHRAG_DIR), exist_ok=True)

    triples = load_triples()

    # Build NetworkX graph from triples
    G = nx.Graph()
    for t in triples:
        subj = t["subject"]
        topic = t["source_topic"]
        G.add_node(subj, type="statistic",
                   description=f"{subj} ({t['relation']}: {t['object']})")
        G.add_node(topic, type="topic",
                   description=f"{topic} (environmental topic)")
        G.add_edge(subj, topic, weight=1.0,
                   description=f"{subj} {t['relation']} {t['object']}",
                   relation=t["relation"])

    log.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Community detection using Louvain
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, resolution=1.0, seed=42)
    log.info(f"Found {len(communities)} communities")

    # Convert to list for indexing
    communities = [sorted(list(c)) for c in communities]

    # Create DataFrames
    entities_records = []
    node_to_community = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = comm_id
            data = G.nodes[node]
            entities_records.append({
                "id": node, "name": node,
                "type": data.get("type", "unknown"),
                "description": data.get("description", ""),
                "community": comm_id,
            })

    entities_df = pd.DataFrame(entities_records)

    rel_records = []
    for src, tgt, data in G.edges(data=True):
        rel_records.append({
            "source": src, "target": tgt,
            "weight": data.get("weight", 1.0),
            "description": data.get("description", ""),
            "relation": data.get("relation", "RelatedTo"),
        })
    relationships_df = pd.DataFrame(rel_records)

    comm_records = []
    for comm_id, members in enumerate(communities):
        comm_records.append({
            "id": comm_id,
            "title": f"Community {comm_id}",
            "level": 0,
            "size": len(members),
            "members": members[:20],
        })
    communities_df = pd.DataFrame(comm_records)

    text_records = []
    for i, t in enumerate(triples):
        text_records.append({
            "id": f"tu-{i}",
            "text": f"The {t['relation']} of {t['subject']} is {t['object']}.",
            "entity_ids": [t["subject"], t["source_topic"]],
        })
    text_units_df = pd.DataFrame(text_records)

    # Save
    entities_df.to_parquet(MS_GRAPHRAG_DIR / "entities.parquet")
    relationships_df.to_parquet(MS_GRAPHRAG_DIR / "relationships.parquet")
    communities_df.to_parquet(MS_GRAPHRAG_DIR / "communities.parquet")
    text_units_df.to_parquet(MS_GRAPHRAG_DIR / "text_units.parquet")

    log.info(f"Saved: {len(entities_df)} entities, {len(relationships_df)} rels, "
             f"{len(communities_df)} communities, {len(text_units_df)} text units")

    return entities_df, relationships_df, communities_df, text_units_df


async def generate_community_reports(communities_df, entities_df, relationships_df, triples):
    """Generate community summaries using LLM, parallelized across 6 API keys."""
    import pandas as pd
    import httpx
    import concurrent.futures

    reports_path = MS_GRAPHRAG_DIR / "community_reports.parquet"
    if reports_path.exists():
        log.info("Community reports already exist, loading...")
        return pd.read_parquet(reports_path)

    model = GENERATOR_MODELS["mistral-small-24b"]

    topic_triples = defaultdict(list)
    for t in triples:
        topic_triples[t["source_topic"]].append(t)

    # Build all prompts first
    tasks = []
    for _, row in communities_df.iterrows():
        comm_id = row["id"]
        members = row["members"]

        facts = []
        for member in members[:10]:
            for t in topic_triples.get(member, [])[:3]:
                facts.append(f"- {t['subject']} {t['relation']} {t['object']}")

        if not facts:
            member_ents = entities_df[entities_df["name"].isin(members)]
            for _, ent in member_ents.head(5).iterrows():
                facts.append(f"- {ent['name']}: {ent['description']}")

        facts_text = "\n".join(facts[:15])
        prompt = (f"Summarize this community of related environmental statistics topics.\n"
                  f"Members include: {', '.join(members[:10])}\n\n"
                  f"Key facts:\n{facts_text}\n\n"
                  f"Write a 2-3 sentence summary of what this community covers and its key statistics.")

        tasks.append({"comm_id": comm_id, "members": members, "prompt": prompt, "size": row["size"]})

    log.info(f"Generating {len(tasks)} community reports using {len(API_KEYS)} API keys in parallel")

    def _do_report(task, api_key):
        from openai import OpenAI as _OAI
        client = _OAI(
            api_key=api_key, base_url=API_BASE_URL,
            http_client=httpx.Client(verify=False, timeout=60.0),
        )
        try:
            response = chat_completion(
                client, model,
                messages=[
                    {"role": "system", "content": "You are a concise summarizer of environmental statistics."},
                    {"role": "user", "content": task["prompt"]},
                ],
                temperature=0.1, max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"Community {task['comm_id']} summary failed: {e}")
            return f"Community covering: {', '.join(task['members'][:5])}"

    # Process in waves of len(API_KEYS) parallel calls
    reports = []
    n_keys = len(API_KEYS)
    loop = asyncio.get_event_loop()

    for wave_start in range(0, len(tasks), n_keys):
        wave = tasks[wave_start:wave_start + n_keys]
        pairs = [(t, API_KEYS[i % n_keys]) for i, t in enumerate(wave)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_keys) as pool:
            summaries = list(pool.map(lambda p: _do_report(p[0], p[1]), pairs))

        for task, summary in zip(wave, summaries):
            reports.append({
                "id": task["comm_id"], "community": task["comm_id"],
                "title": f"Community {task['comm_id']}: {', '.join(task['members'][:3])}",
                "summary": summary, "full_content": summary,
                "level": 0, "rank": task["size"],
            })

        done = min(wave_start + len(wave), len(tasks))
        if done % 60 < n_keys or done == len(tasks):
            log.info(f"  Community reports: {done}/{len(tasks)}")

        if wave_start + n_keys < len(tasks):
            await asyncio.sleep(0.5)

    reports_df = pd.DataFrame(reports)
    reports_df.to_parquet(reports_path)
    log.info(f"Generated {len(reports_df)} community reports")
    return reports_df


async def query_ms_graphrag(query: str, model_key: str = "devstral-24b"):
    """
    Query using MS GraphRAG-style community-based retrieval:
    1. Embed query → find relevant entities
    2. Map entities to communities → get community reports
    3. Retrieve relevant triples from those communities
    4. Generate answer grounded on community context
    """
    import pandas as pd

    entities_df = pd.read_parquet(MS_GRAPHRAG_DIR / "entities.parquet")
    reports_df = pd.read_parquet(MS_GRAPHRAG_DIR / "community_reports.parquet")
    text_units_df = pd.read_parquet(MS_GRAPHRAG_DIR / "text_units.parquet")

    start = time.time()

    # Step 1: Find relevant entities using embedding similarity
    client_oai = create_client()
    query_emb_response = client_oai.embeddings.create(
        model=EMBEDDING_MODEL, input=[query],
    )
    query_emb = np.array(query_emb_response.data[0].embedding)

    # Load or build entity embeddings (cached)
    ent_emb_cache = MS_GRAPHRAG_DIR / "entity_embeddings.npy"
    ent_names_cache = MS_GRAPHRAG_DIR / "entity_names.json"

    if ent_emb_cache.exists():
        entity_embeddings = np.load(ent_emb_cache)
        with open(ent_names_cache) as f:
            entity_names = json.load(f)
    else:
        entity_texts = [
            f"{row['name']}: {row['description']}"
            for _, row in entities_df.iterrows()
        ]
        entity_names = entities_df["name"].tolist()

        log.info(f"Embedding {len(entity_texts)} entities (one-time)...")
        entity_embeddings = []
        batch_size = 20
        for i in range(0, len(entity_texts), batch_size):
            batch = entity_texts[i:i+batch_size]
            resp = client_oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            entity_embeddings.extend([d.embedding for d in resp.data])
            if i + batch_size < len(entity_texts):
                time.sleep(2)
            if (i // batch_size) % 100 == 0:
                log.info(f"  Embedded {i+len(batch)}/{len(entity_texts)}")

        entity_embeddings = np.array(entity_embeddings)
        np.save(ent_emb_cache, entity_embeddings)
        with open(ent_names_cache, "w") as f:
            json.dump(entity_names, f)

    # Cosine similarity
    norms = np.linalg.norm(entity_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
    sims = entity_embeddings @ query_emb / norms
    top_k = 10
    top_indices = np.argsort(sims)[-top_k:][::-1]
    top_entities = [entity_names[i] for i in top_indices]

    # Step 2: Map to communities
    relevant_communities = set()
    for ent_name in top_entities:
        ent_rows = entities_df[entities_df["name"] == ent_name]
        for _, row in ent_rows.iterrows():
            relevant_communities.add(row["community"])

    # Step 3: Get community reports + relevant triples
    community_context = []
    for comm_id in list(relevant_communities)[:5]:
        report_rows = reports_df[reports_df["community"] == comm_id]
        for _, row in report_rows.iterrows():
            community_context.append(row["summary"])

    relevant_triples = []
    for ent_name in top_entities:
        matches = text_units_df[
            text_units_df["entity_ids"].apply(
                lambda x: ent_name in x if isinstance(x, list) else False
            )
        ]
        for _, row in matches.head(3).iterrows():
            relevant_triples.append(row["text"])

    retrieval_ms = (time.time() - start) * 1000

    # Build context
    context_parts = []
    if community_context:
        context_parts.append("Community Summaries:\n" + "\n".join(community_context))
    if relevant_triples:
        context_parts.append("Relevant Facts:\n" + "\n".join(
            f"[{i+1}] {t}" for i, t in enumerate(relevant_triples[:10])
        ))
    context = "\n\n".join(context_parts)

    # Generate answer
    model = GENERATOR_MODELS[model_key]
    gen_start = time.time()
    response = chat_completion(
        client_oai, model,
        messages=[
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    total_ms = (time.time() - start) * 1000

    return {
        "query": query,
        "model": model_key,
        "method": "ms_graphrag",
        "answer": response.choices[0].message.content.strip(),
        "context": context[:2000],
        "latency_ms": total_ms,
        "retrieval_ms": retrieval_ms,
        "generation_ms": (time.time() - gen_start) * 1000,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


# =============================================================================
# Batch Query
# =============================================================================

async def batch_query(method: str, model_key: str = "mistral-small-24b",
                      mode: str = None, max_samples: int = None):
    """Run benchmark queries. Reuses LightRAG instance to avoid reloading storage."""
    bench_path = ECOSTATS_DIR / "ecostats_benchmark.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if max_samples:
        benchmark = benchmark[:max_samples]

    output_dir = RESULTS_DIR / "exp1_full_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{method}_{model_key}.jsonl"

    # Resume: skip successfully completed queries (not error records)
    existing = set()
    clean_lines = []
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if "error" not in r:
                            existing.add(r["query"])
                            clean_lines.append(line)
                    except (json.JSONDecodeError, KeyError):
                        pass
        # Purge error records so they get retried
        raw_count = sum(1 for l in open(output_file) if l.strip())
        if len(clean_lines) < raw_count:
            with open(output_file, "w") as f:
                f.writelines(clean_lines)
            log.info(f"Purged error records, kept {len(clean_lines)} clean results")
        log.info(f"Resuming: {len(existing)} already done")

    remaining = [b for b in benchmark if b["question"] not in existing]
    if not remaining:
        log.info(f"{method}_{model_key}: already complete ({len(existing)}/{len(benchmark)})")
        return

    log.info(f"Running {method} with {model_key}: {len(remaining)} queries remaining")

    # Initialize LightRAG once for all queries (avoids reloading 2.6GB per query)
    rag_instance = None
    if method == "lightrag":
        rag_instance = get_lightrag_instance()
        await rag_instance.initialize_storages()
        log.info("LightRAG instance loaded (reused for all queries)")

    try:
        with open(output_file, "a", encoding="utf-8") as fout:
            for i, item in enumerate(remaining):
                query = item["question"]
                try:
                    if method == "lightrag":
                        result = await query_lightrag(
                            query, model_key, mode or "hybrid",
                            rag_instance=rag_instance,
                        )
                    elif method == "ms_graphrag":
                        result = await query_ms_graphrag(query, model_key)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

                    if (i + 1) % 10 == 0:
                        log.info(f"  [{i+1}/{len(remaining)}] latency: {result['latency_ms']:.0f}ms")

                except Exception as e:
                    log.error(f"Error on query {i}: {e}")
                    fout.write(json.dumps({
                        "query": query, "model": model_key, "method": method,
                        "answer": "", "error": str(e), "latency_ms": 0,
                    }, ensure_ascii=False) + "\n")
                    fout.flush()

                await asyncio.sleep(2)
    finally:
        if rag_instance is not None:
            await rag_instance.finalize_storages()
            log.info("LightRAG instance finalized")

    log.info(f"Done! Results: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Graph-based RAG baselines with EcoStatKG")
    parser.add_argument("action",
                        choices=["insert-lightrag", "insert-graphrag", "query", "test"],
                        help="Action to perform")
    parser.add_argument("--method", choices=["lightrag", "ms_graphrag"], default="lightrag")
    parser.add_argument("--model", default="mistral-small-24b")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-structural", action="store_true",
                        help="Include structural triples in LightRAG")

    args = parser.parse_args()

    if args.action == "insert-lightrag":
        asyncio.run(insert_kg_to_lightrag(include_structural=args.include_structural))

    elif args.action == "insert-graphrag":
        triples = load_triples()
        entities_df, rel_df, comm_df, tu_df = setup_ms_graphrag()
        asyncio.run(generate_community_reports(comm_df, entities_df, rel_df, triples))

    elif args.action == "query":
        asyncio.run(batch_query(
            method=args.method, model_key=args.model,
            mode=args.mode, max_samples=args.max_samples,
        ))

    elif args.action == "test":
        async def _test():
            if args.method == "lightrag":
                r = await query_lightrag(
                    "What percentage of global electricity comes from wind power?",
                    model_key=args.model)
            else:
                r = await query_ms_graphrag(
                    "What percentage of global electricity comes from wind power?",
                    model_key=args.model)
            print(json.dumps(r, indent=2))
        asyncio.run(_test())


if __name__ == "__main__":
    main()

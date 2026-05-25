"""
Held-Out Experiments for LightRAG and MS GraphRAG
==================================================
Builds new LightRAG / MS GraphRAG indexes excluding the 409 held-out
topics, then runs benchmark queries across all active models.

Usage:
    # Step 1: Build held-out indexes (hours)
    python 15_graph_heldout.py build-lightrag
    python 15_graph_heldout.py build-graphrag

    # Step 2: Query with each model
    python 15_graph_heldout.py query --method lightrag --model mistral-small-24b
    python 15_graph_heldout.py query --method ms_graphrag --model mistral-small-24b

    # Step 3: Query all models
    python 15_graph_heldout.py query-all --method lightrag
    python 15_graph_heldout.py query-all --method ms_graphrag
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
    API_BASE_URL, API_KEY, DATA_DIR, RESULTS_DIR,
    ECOSTATS_DIR, GENERATOR_MODELS, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    GROUNDED_SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS, TOP_P,
    load_held_out_topics,
)
from utils import create_client, chat_completion

log = logging.getLogger("ecostats.graph_heldout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LIGHTRAG_HELDOUT_DIR = DATA_DIR / "lightrag_kg_heldout"
MS_GRAPHRAG_HELDOUT_DIR = DATA_DIR / "ms_graphrag_kg_heldout"

API_KEYS = [
    os.getenv("OPENAI_API_KEY_1"),
    os.getenv("OPENAI_API_KEY_2"),
    os.getenv("OPENAI_API_KEY_3"),
    os.getenv("OPENAI_API_KEY_4"),
    os.getenv("OPENAI_API_KEY_5"),
    os.getenv("OPENAI_API_KEY_6"),
]
API_KEYS = [k for k in API_KEYS if k]
if not os.getenv("OPENAI_API_KEY") and API_KEYS:
    os.environ["OPENAI_API_KEY"] = API_KEYS[0]


# ── helpers ────────────────────────────────────────────────────────────

def load_filtered_triples():
    """Load LLM triples with held-out topics removed."""
    held_out = set(load_held_out_topics())
    kg_path = DATA_DIR / "triples" / "kg_triples.json"
    with open(kg_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    before = len(triples)
    triples = [t for t in triples if t["source_topic"] not in held_out]
    log.info(f"Filtered triples: {before} → {len(triples)} (removed {before - len(triples)} held-out)")
    return triples


def load_filtered_structural():
    """Load structural triples with held-out topics removed."""
    held_out = set(load_held_out_topics())
    st_path = DATA_DIR / "triples" / "structural_triples.json"
    with open(st_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    before = len(triples)
    triples = [t for t in triples if t.get("source_topic", "") not in held_out]
    log.info(f"Filtered structural: {before} → {len(triples)}")
    return triples


# ── LightRAG held-out ──────────────────────────────────────────────────

def get_lightrag_heldout_instance(api_key=None):
    """Create a LightRAG instance pointing to the held-out directory."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    working_dir = str(LIGHTRAG_HELDOUT_DIR)
    os.makedirs(working_dir, exist_ok=True)

    if api_key is None:
        api_key = API_KEYS[0]

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
        batch_size = 30
        all_results = [None] * len(texts)
        tasks = []
        for i in range(0, len(texts), batch_size):
            sub = texts[i:i+batch_size]
            key_idx = _key_counter["idx"] % _num_keys
            _key_counter["idx"] += 1
            tasks.append((i, sub, API_KEYS[key_idx]))

        async def _embed_with_retry(sub, key, max_retries=5):
            for attempt in range(max_retries):
                try:
                    return await openai_embed.func(
                        sub, model=EMBEDDING_MODEL,
                        api_key=key, base_url=API_BASE_URL)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 5 * (attempt + 1)
                        log.warning(f"Embed retry {attempt+1}/{max_retries}: {str(e)[:80]}")
                        await asyncio.sleep(wait)
                    else:
                        raise

        for wave_start in range(0, len(tasks), _num_keys):
            wave = tasks[wave_start:wave_start + _num_keys]
            coros = [
                _embed_with_retry(sub, key)
                for _, sub, key in wave
            ]
            results = await asyncio.gather(*coros)
            for (orig_i, sub, _), emb in zip(wave, results):
                for j in range(len(sub)):
                    all_results[orig_i + j] = emb[j]
            if wave_start + _num_keys < len(tasks):
                await asyncio.sleep(0.5)
        return np.array(all_results)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name="Mistral-Small-24B-Instruct-2501-FP8-dynamic",
        embedding_func=embedding_func,
        embedding_batch_num=180,
        embedding_func_max_async=1,
        llm_model_max_async=2,
        addon_params={
            "language": "English",
            "entity_types": ["organization", "location", "statistic", "resource", "event", "policy"],
        },
    )
    return rag


async def build_lightrag_heldout():
    """Build LightRAG index excluding held-out topics."""
    from importlib import import_module
    # Reuse the conversion function from 13_graph_baselines
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_baselines", Path(__file__).parent / "13_graph_baselines.py")
    gb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gb)

    triples = load_filtered_triples()
    custom_kg = gb.triples_to_lightrag_format(triples, structural_triples=None)

    rag = get_lightrag_heldout_instance()
    await rag.initialize_storages()

    try:
        total_chunks = len(custom_kg["chunks"])
        total_ents = len(custom_kg["entities"])
        total_rels = len(custom_kg["relationships"])
        log.info(f"Inserting {total_chunks} chunks, {total_ents} entities, {total_rels} relationships")

        batch_size = 2000
        for kind, items, bs in [
            ("chunks", custom_kg["chunks"], 1000),
            ("entities", custom_kg["entities"], 2000),
            ("relationships", custom_kg["relationships"], 3000),
        ]:
            batches = [items[i:i+bs] for i in range(0, len(items), bs)]
            for i, batch in enumerate(batches):
                log.info(f"  {kind} batch {i+1}/{len(batches)} ({len(batch)} items)")
                mini_kg = {"chunks": [], "entities": [], "relationships": []}
                mini_kg[kind] = batch
                for attempt in range(5):
                    try:
                        await rag.ainsert_custom_kg(mini_kg)
                        break
                    except Exception as e:
                        log.warning(f"  Retry {attempt+1}/5 for {kind} batch {i+1}: {e}")
                        await asyncio.sleep(10 * (attempt + 1))
                else:
                    log.error(f"  FAILED after 5 retries: {kind} batch {i+1}")
                await asyncio.sleep(1)

        log.info("LightRAG held-out index built successfully!")
    finally:
        await rag.finalize_storages()


# ── MS GraphRAG held-out ───────────────────────────────────────────────

async def build_graphrag_heldout():
    """Build MS GraphRAG structure excluding held-out topics."""
    import pandas as pd
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    import concurrent.futures
    import httpx

    os.makedirs(str(MS_GRAPHRAG_HELDOUT_DIR), exist_ok=True)

    triples = load_filtered_triples()

    # Build graph
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

    communities = louvain_communities(G, resolution=1.0, seed=42)
    communities = [sorted(list(c)) for c in communities]
    log.info(f"Found {len(communities)} communities")

    # DataFrames
    entities_records = []
    for comm_id, members in enumerate(communities):
        for node in members:
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
            "id": comm_id, "title": f"Community {comm_id}",
            "level": 0, "size": len(members), "members": members[:20],
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
    entities_df.to_parquet(MS_GRAPHRAG_HELDOUT_DIR / "entities.parquet")
    relationships_df.to_parquet(MS_GRAPHRAG_HELDOUT_DIR / "relationships.parquet")
    communities_df.to_parquet(MS_GRAPHRAG_HELDOUT_DIR / "communities.parquet")
    text_units_df.to_parquet(MS_GRAPHRAG_HELDOUT_DIR / "text_units.parquet")

    log.info(f"Saved: {len(entities_df)} entities, {len(relationships_df)} rels, "
             f"{len(communities_df)} communities")

    # Generate community reports
    model = GENERATOR_MODELS["mistral-small-24b"]
    topic_triples = defaultdict(list)
    for t in triples:
        topic_triples[t["source_topic"]].append(t)

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
            return f"Community covering: {', '.join(task['members'][:5])}"

    tasks = []
    for _, row in communities_df.iterrows():
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
        tasks.append({"comm_id": row["id"], "members": members, "prompt": prompt, "size": row["size"]})

    log.info(f"Generating {len(tasks)} community reports")
    reports = []
    n_keys = len(API_KEYS)
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
        if (wave_start // n_keys) % 50 == 0:
            log.info(f"  Reports: {min(wave_start + n_keys, len(tasks))}/{len(tasks)}")
        if wave_start + n_keys < len(tasks):
            await asyncio.sleep(0.5)

    reports_df = pd.DataFrame(reports)
    reports_df.to_parquet(MS_GRAPHRAG_HELDOUT_DIR / "community_reports.parquet")
    log.info(f"Generated {len(reports_df)} community reports")

    # Build entity embeddings
    log.info("Embedding entities for held-out index...")
    client_oai = create_client()
    entity_texts = [
        f"{row['name']}: {row['description']}"
        for _, row in entities_df.iterrows()
    ]
    entity_names = entities_df["name"].tolist()
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
    np.save(MS_GRAPHRAG_HELDOUT_DIR / "entity_embeddings.npy", entity_embeddings)
    with open(MS_GRAPHRAG_HELDOUT_DIR / "entity_names.json", "w") as f:
        json.dump(entity_names, f)

    log.info("MS GraphRAG held-out index built successfully!")


# ── Query functions ────────────────────────────────────────────────────

async def query_lightrag_heldout(query, model_key, rag_instance=None):
    """Query LightRAG held-out index."""
    from lightrag import QueryParam

    owns_rag = rag_instance is None
    rag = rag_instance or get_lightrag_heldout_instance()
    if owns_rag:
        await rag.initialize_storages()

    try:
        start = time.time()
        param = QueryParam(mode="hybrid", only_need_context=True, top_k=10)
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
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        total_ms = (time.time() - start) * 1000

        return {
            "query": query, "model": model_key, "method": "lightrag",
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


async def query_ms_graphrag_heldout(query, model_key):
    """Query MS GraphRAG held-out index."""
    import pandas as pd

    entities_df = pd.read_parquet(MS_GRAPHRAG_HELDOUT_DIR / "entities.parquet")
    reports_df = pd.read_parquet(MS_GRAPHRAG_HELDOUT_DIR / "community_reports.parquet")
    text_units_df = pd.read_parquet(MS_GRAPHRAG_HELDOUT_DIR / "text_units.parquet")

    start = time.time()
    client_oai = create_client()
    query_emb_response = client_oai.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_emb = np.array(query_emb_response.data[0].embedding)

    entity_embeddings = np.load(MS_GRAPHRAG_HELDOUT_DIR / "entity_embeddings.npy")
    with open(MS_GRAPHRAG_HELDOUT_DIR / "entity_names.json") as f:
        entity_names = json.load(f)

    norms = np.linalg.norm(entity_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
    sims = entity_embeddings @ query_emb / norms
    top_indices = np.argsort(sims)[-10:][::-1]
    top_entities = [entity_names[i] for i in top_indices]

    relevant_communities = set()
    for ent_name in top_entities:
        ent_rows = entities_df[entities_df["name"] == ent_name]
        for _, row in ent_rows.iterrows():
            relevant_communities.add(row["community"])

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

    context_parts = []
    if community_context:
        context_parts.append("Community Summaries:\n" + "\n".join(community_context))
    if relevant_triples:
        context_parts.append("Relevant Facts:\n" + "\n".join(
            f"[{i+1}] {t}" for i, t in enumerate(relevant_triples[:10])
        ))
    context = "\n\n".join(context_parts)

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
        "query": query, "model": model_key, "method": "ms_graphrag",
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


# ── Batch queries ──────────────────────────────────────────────────────

async def batch_query_heldout(method, model_key, max_samples=None):
    """Run held-out benchmark queries."""
    with open(ECOSTATS_DIR / "ecostats_benchmark.json") as f:
        benchmark = json.load(f)
    if max_samples:
        benchmark = benchmark[:max_samples]

    output_dir = RESULTS_DIR / "exp1_main"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{method}_{model_key}.jsonl"

    # Resume
    existing = set()
    clean_lines = []
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if "error" not in r:
                            existing.add(r["query"])
                            clean_lines.append(line)
                    except (json.JSONDecodeError, KeyError):
                        pass
        raw_count = sum(1 for l in open(output_file) if l.strip())
        if len(clean_lines) < raw_count:
            with open(output_file, "w") as f:
                f.writelines(clean_lines)

    remaining = [b for b in benchmark if b["question"] not in existing]
    if not remaining:
        log.info(f"{method}_{model_key}: already complete ({len(existing)}/{len(benchmark)})")
        return

    log.info(f"Running held-out {method} with {model_key}: {len(remaining)} remaining")

    rag_instance = None
    if method == "lightrag":
        rag_instance = get_lightrag_heldout_instance()
        await rag_instance.initialize_storages()

    try:
        with open(output_file, "a", encoding="utf-8") as fout:
            for i, item in enumerate(remaining):
                query = item["question"]
                try:
                    if method == "lightrag":
                        result = await query_lightrag_heldout(
                            query, model_key, rag_instance=rag_instance)
                    else:
                        result = await query_ms_graphrag_heldout(query, model_key)

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
        if rag_instance:
            await rag_instance.finalize_storages()

    log.info(f"Done! Results: {output_file}")


async def query_all_models(method):
    """Run held-out queries for all active models."""
    models = [k for k in GENERATOR_MODELS if k != "devstral-24b"]
    for model_key in models:
        log.info(f"\n{'='*60}")
        log.info(f"Model: {model_key}")
        log.info(f"{'='*60}")
        await batch_query_heldout(method, model_key)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Held-out experiments for LightRAG / MS GraphRAG")
    parser.add_argument("action",
                        choices=["build-lightrag", "build-graphrag", "query", "query-all"])
    parser.add_argument("--method", choices=["lightrag", "ms_graphrag"], default="lightrag")
    parser.add_argument("--model", default="mistral-small-24b")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.action == "build-lightrag":
        asyncio.run(build_lightrag_heldout())
    elif args.action == "build-graphrag":
        asyncio.run(build_graphrag_heldout())
    elif args.action == "query":
        asyncio.run(batch_query_heldout(args.method, args.model, args.max_samples))
    elif args.action == "query-all":
        asyncio.run(query_all_models(args.method))


if __name__ == "__main__":
    main()

"""
Step 2: Triple Extraction (LLM-based)
========================================
Uses the LLM API to extract structured KG triples from statistical sentences.
Validates against the 29-relation schema (v2.0, pruned from original 46).

Each triple includes provenance: the source sentence it was extracted from.

Features:
  - 30 calls/min rate limiting with visible countdown
  - Incremental save -- can resume from checkpoint if interrupted
  - Robust error handling (NoneType responses, JSON parse failures)
  - Provenance: source_sentence stored per triple
  - Progress bar with ETA

Usage:
    python 02_triple_extraction.py            # full run
    python 02_triple_extraction.py --resume   # resume from checkpoint
    python 02_triple_extraction.py --limit 50 # test on first 50 topics
"""

import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

from openai import OpenAI

from config import (
    API_BASE_URL, API_KEY, HTTP_CLIENT,
    EXTRACTION_MODEL, SCHEMA_PATH,
    CORPUS_DIR, TRIPLES_DIR,
    MAX_CALLS_PER_MINUTE,
    TRIPLE_EXTRACTION_PROMPT,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_SENTENCES_PER_TOPIC = 20
SENTENCES_PER_BATCH = 5     # Group sentences to reduce API calls
CHECKPOINT_EVERY = 10       # Save checkpoint every N topics (low for crash safety)
MAX_RETRIES = 3
RETRY_DELAY = 5.0


def load_schema() -> tuple[list[str], str]:
    """Load the 29-relation schema (v2.0)."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    names = [r["name"] for r in schema["relations"]]
    formatted = "\n".join(
        f"- {r['name']}: {r['description']} (e.g., {r['example']})"
        for r in schema["relations"]
    )
    print(f"Schema: {len(names)} relations (v{schema.get('schema_version', '?')})")
    return names, formatted


def load_corpus() -> dict:
    """Load the extracted corpus."""
    with open(CORPUS_DIR / "statistical_corpus.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint() -> set[str]:
    """Load set of already-processed topic names from checkpoint."""
    checkpoint_path = TRIPLES_DIR / "_checkpoint_topics.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(processed_topics: set[str]):
    """Save checkpoint of processed topics (atomic write)."""
    checkpoint_path = TRIPLES_DIR / "_checkpoint_topics.json"
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(processed_topics), f)
    tmp_path.replace(checkpoint_path)


def create_client() -> OpenAI:
    """Create OpenAI-compatible client."""
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        http_client=HTTP_CLIENT,
    )


class RateLimiter:
    """Simple 30-calls-per-minute rate limiter with visible waiting."""

    def __init__(self, max_per_minute: int = MAX_CALLS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self.call_count = 0
        self.window_start = time.time()

    def wait_if_needed(self):
        """Block if we've hit the rate limit, with countdown."""
        self.call_count += 1
        if self.call_count >= self.max_per_minute:
            elapsed = time.time() - self.window_start
            if elapsed < 60.0:
                wait = 60.0 - elapsed + 1  # +1s safety margin
                tqdm.write(f"  Rate limit reached ({self.max_per_minute}/min). Waiting {wait:.0f}s...")
                time.sleep(wait)
            self.call_count = 0
            self.window_start = time.time()


def validate_triples(
    raw_triples: list,
    valid_relations: set[str],
    topic: str,
    source_sentences: str = "",
) -> list[dict]:
    """Validate and clean extracted triples. Attach provenance."""
    validated = []
    for t in raw_triples:
        if not isinstance(t, dict):
            continue
        if not all(k in t for k in ("subject", "relation", "object")):
            continue
        if t["relation"] not in valid_relations:
            continue
        obj = str(t["object"])
        if not any(c.isdigit() for c in obj):
            continue
        t["source_topic"] = topic
        t["source_sentences"] = source_sentences
        validated.append(t)
    return validated


def extract_single_batch(
    client: OpenAI,
    topic: str,
    text: str,
    relations_list: str,
    valid_relations: set[str],
    rate_limiter: RateLimiter,
) -> list[dict]:
    """Extract triples from a single text batch (with retry)."""
    prompt = TRIPLE_EXTRACTION_PROMPT.format(
        topic=topic,
        relations_list=relations_list,
        text=text,
    )

    for attempt in range(MAX_RETRIES):
        rate_limiter.wait_if_needed()
        try:
            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise information extraction system. Output ONLY valid JSON arrays."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            content = response.choices[0].message.content
            # Fix: handle None content gracefully
            if content is None:
                return []

            content = content.strip()
            if not content:
                return []

            # Parse JSON (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            triples = json.loads(content)
            if not isinstance(triples, list):
                return []

            return validate_triples(triples, valid_relations, topic, source_sentences=text)

        except json.JSONDecodeError:
            return []  # Bad JSON — skip, don't retry
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  Retry {attempt+1} for '{topic}': {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                tqdm.write(f"  Failed '{topic}' after {MAX_RETRIES} retries: {e}")
                return []

    return []


def extract_topic(
    client: OpenAI,
    topic: str,
    sentences: list[str],
    relations_list: str,
    valid_relations: set[str],
    rate_limiter: RateLimiter,
) -> list[dict]:
    """Extract triples for one topic (may produce multiple API calls)."""
    sentences = sentences[:MAX_SENTENCES_PER_TOPIC]
    all_triples = []

    for i in range(0, len(sentences), SENTENCES_PER_BATCH):
        batch = sentences[i:i + SENTENCES_PER_BATCH]
        combined = "\n\n".join(batch)
        triples = extract_single_batch(
            client, topic, combined, relations_list, valid_relations, rate_limiter
        )
        all_triples.extend(triples)

    return all_triples


def batch_extract(
    corpus: dict,
    resume: bool = False,
) -> list[dict]:
    """
    Process entire corpus sequentially with 30/min rate limiting.
    Saves incrementally so progress isn't lost.
    """
    client = create_client()
    relation_names, relations_formatted = load_schema()
    valid_relations = set(relation_names)
    rate_limiter = RateLimiter(MAX_CALLS_PER_MINUTE)

    # Resume support
    processed = load_checkpoint() if resume else set()
    if processed:
        print(f"  Resuming: {len(processed)} topics already done")

    # Load existing triples if resuming
    all_triples = []
    jsonl_path = TRIPLES_DIR / "kg_triples.jsonl"
    if resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_triples.append(json.loads(line))
        print(f"  Loaded {len(all_triples)} existing triples")

    # Filter to unprocessed topics WITH statistical sentences
    topics_to_process = [
        (topic, data) for topic, data in corpus.items()
        if topic not in processed and data.get("sentence_count", 0) > 0
    ]

    total_calls = sum(
        -(-len(data["sentences"][:MAX_SENTENCES_PER_TOPIC]) // SENTENCES_PER_BATCH)
        for _, data in topics_to_process
    )
    est_minutes = total_calls / MAX_CALLS_PER_MINUTE
    print(f"\n  Topics to process: {len(topics_to_process)}")
    print(f"  Estimated API calls: {total_calls}")
    print(f"  Rate limit: {MAX_CALLS_PER_MINUTE} calls/min")
    print(f"  Est. time: {est_minutes:.0f} min ({est_minutes/60:.1f} h)\n")

    pbar = tqdm(total=len(topics_to_process), desc="Extracting triples", unit="topic")
    checkpoint_counter = 0

    for topic, data in topics_to_process:
        triples = extract_topic(
            client, topic, data["sentences"],
            relations_formatted, valid_relations, rate_limiter
        )
        all_triples.extend(triples)
        processed.add(topic)
        pbar.update(1)
        pbar.set_postfix(triples=len(all_triples))
        checkpoint_counter += 1

        # Incremental save (atomic: write .tmp then rename)
        if checkpoint_counter >= CHECKPOINT_EVERY:
            tmp_jsonl = jsonl_path.with_suffix(".tmp")
            with open(tmp_jsonl, "w", encoding="utf-8") as f:
                for t in all_triples:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            tmp_jsonl.replace(jsonl_path)
            save_checkpoint(processed)
            checkpoint_counter = 0

    # Final save
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in all_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    save_checkpoint(processed)

    pbar.close()
    return all_triples


def save_triples(triples: list[dict], output_dir: Path = TRIPLES_DIR):
    """Save triples in multiple formats + print stats."""
    jsonl_path = output_dir / "kg_triples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    json_path = output_dir / "kg_triples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    relation_counts = {}
    for t in triples:
        r = t["relation"]
        relation_counts[r] = relation_counts.get(r, 0) + 1

    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_triples": len(triples),
            "unique_subjects": len(set(t["subject"] for t in triples)),
            "relation_distribution": dict(sorted(relation_counts.items(), key=lambda x: -x[1])),
        }, f, indent=2)

    print(f"\n=== Triple Extraction Complete ===")
    print(f"Total triples:    {len(triples)}")
    print(f"Unique subjects:  {len(set(t['subject'] for t in triples))}")
    print(f"Relations used:   {len(relation_counts)}")
    print(f"\nTop 10 relations:")
    for r, c in sorted(relation_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {r}: {c}")
    print(f"\nSaved to: {jsonl_path}")

    # Clean up checkpoint
    checkpoint = TRIPLES_DIR / "_checkpoint_topics.json"
    if checkpoint.exists():
        checkpoint.unlink()
        print("Checkpoint cleaned up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract KG triples from corpus")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N topics (0=all, for testing)")
    args = parser.parse_args()

    corpus = load_corpus()

    if args.limit > 0:
        # Take first N topics
        keys = list(corpus.keys())[:args.limit]
        corpus = {k: corpus[k] for k in keys}
        print(f"Limited to {len(corpus)} topics")

    triples = batch_extract(corpus, resume=args.resume)
    save_triples(triples)

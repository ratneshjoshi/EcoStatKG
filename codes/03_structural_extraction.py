"""
Step 2b: Structural Triple Extraction (Wikipedia Metadata)
==========================================================
Extracts deterministic triples from Wikipedia article structure:
  - Hyperlinks  -> RelatedTo
  - Categories  -> IsA / BelongsToCategory
  - Redirects   -> Synonym / AlternativeName

NO LLM calls required -- uses only the free Wikipedia API.
These triples complement LLM-extracted statistical triples (Step 02).

Features:
  - Checkpoint every 200 topics
  - Rate-limited Wikipedia API (0.3s between requests + backoff on 429)
  - Filters internal links to only those that are also in our topic set
  - Category-based IsA triples for taxonomic grounding

Usage:
    python 02b_structural_extraction.py             # full run
    python 02b_structural_extraction.py --resume    # resume from checkpoint
    python 02b_structural_extraction.py --limit 100 # test on first 100 topics
"""

import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from config import DATA_DIR, CORPUS_DIR, TRIPLES_DIR


# === Constants ===
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "EcoStatKG-Builder/1.0 (research project)"}
WIKI_DELAY = 0.3
CHECKPOINT_EVERY = 200

# Categories to skip (maintenance/meta)
SKIP_CAT_PATTERNS = [
    "articles", "pages", "cs1", "webarchive", "wikidata", "commons",
    "short description", "use dmy", "use mdy", "all stub", "wikipedia",
    "lacking", "needing", "cleanup", "accuracy disputes", "template",
    "redirects", "disambiguation", "good articles", "featured articles",
    "living people", "births", "deaths", "alumni",
]


def wiki_api_call(params: dict, max_retries: int = 4) -> dict | None:
    """Wikipedia API call with retry + exponential backoff."""
    params["format"] = "json"
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=WIKI_HEADERS)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            return None
        except Exception:
            return None
    return None


def should_skip_category(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in SKIP_CAT_PATTERNS)


def get_article_links(title: str) -> list[str]:
    """Get all internal links from a Wikipedia article."""
    links = []
    plcontinue = None
    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "pllimit": "500",
            "plnamespace": "0",  # main namespace only
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        data = wiki_api_call(params)
        if not data or "query" not in data:
            break
        pages = data["query"].get("pages", {})
        for page in pages.values():
            for link in page.get("links", []):
                links.append(link["title"])
        if "continue" in data and "plcontinue" in data["continue"]:
            plcontinue = data["continue"]["plcontinue"]
        else:
            break
    return links


def get_article_categories(title: str) -> list[str]:
    """Get categories for a Wikipedia article."""
    cats = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "categories",
        "cllimit": "500",
        "clshow": "!hidden",  # skip hidden categories
    }
    data = wiki_api_call(params)
    if not data or "query" not in data:
        return cats
    pages = data["query"].get("pages", {})
    for page in pages.values():
        for cat in page.get("categories", []):
            name = cat["title"].replace("Category:", "")
            if not should_skip_category(name):
                cats.append(name)
    return cats


def get_redirects_to(title: str) -> list[str]:
    """Get pages that redirect to this title (i.e., synonyms/alternative names)."""
    redirects = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "redirects",
        "rdlimit": "500",
        "rdnamespace": "0",
    }
    data = wiki_api_call(params)
    if not data or "query" not in data:
        return redirects
    pages = data["query"].get("pages", {})
    for page in pages.values():
        for rd in page.get("redirects", []):
            redirects.append(rd["title"])
    return redirects


def load_topic_set() -> set[str]:
    """Load the curated topic set for filtering links."""
    path = DATA_DIR / "seed_topics.txt"
    if not path.exists():
        path = DATA_DIR / "expanded_topics.txt"
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_corpus_topics() -> list[str]:
    """Load topic list from corpus (all topics, including those without stats)."""
    corpus_path = CORPUS_DIR / "statistical_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return list(corpus.keys())


def load_checkpoint() -> tuple[dict, set[str]]:
    """Load checkpoint if it exists."""
    ckpt_triples = TRIPLES_DIR / "_structural_checkpoint_triples.json"
    ckpt_done = TRIPLES_DIR / "_structural_checkpoint_done.json"
    triples = []
    done = set()
    if ckpt_triples.exists() and ckpt_done.exists():
        with open(ckpt_triples, "r", encoding="utf-8") as f:
            triples = json.load(f)
        with open(ckpt_done, "r", encoding="utf-8") as f:
            done = set(json.load(f))
        print(f"Loaded checkpoint: {len(done)} topics, {len(triples)} triples")
    return triples, done


def save_checkpoint(triples: list[dict], done: set[str]):
    """Atomic checkpoint: write .tmp then rename."""
    ckpt_triples = TRIPLES_DIR / "_structural_checkpoint_triples.json"
    ckpt_done = TRIPLES_DIR / "_structural_checkpoint_done.json"
    # Write triples
    tmp = ckpt_triples.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False)
    tmp.replace(ckpt_triples)
    # Write done set
    tmp = ckpt_done.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)
    tmp.replace(ckpt_done)


def extract_structural_triples(
    topics: list[str],
    topic_set: set[str],
    resume: bool = False,
) -> list[dict]:
    """
    Extract structural triples for all topics.
    
    Triple types produced:
      - (A, RelatedTo, B) where B is a hyperlink target also in our topic set
      - (A, BelongsToCategory, C) for each Wikipedia category of A
      - (A, Synonym, R) for each page that redirects to A
    """
    all_triples, done = load_checkpoint() if resume else ([], set())
    if done:
        print(f"Resuming: {len(done)} topics already done")

    remaining = [t for t in topics if t not in done]
    print(f"Processing {len(remaining)} topics for structural triples...")

    # Normalize topic set for case-insensitive matching
    topic_lower_map = {t.lower(): t for t in topic_set}

    api_calls = 0
    new_count = 0

    for topic in tqdm(remaining, desc="Structural extraction"):
        topic_triples = []

        # 1. Hyperlinks -> RelatedTo (only to topics in our set)
        time.sleep(WIKI_DELAY)
        links = get_article_links(topic)
        api_calls += 1
        for link in links:
            if link.lower() in topic_lower_map and link.lower() != topic.lower():
                target = topic_lower_map[link.lower()]
                topic_triples.append({
                    "subject": topic,
                    "relation": "RelatedTo",
                    "object": target,
                    "source_topic": topic,
                    "extraction_method": "wikipedia_hyperlink",
                })

        # 2. Categories -> BelongsToCategory
        time.sleep(WIKI_DELAY)
        cats = get_article_categories(topic)
        api_calls += 1
        for cat in cats:
            topic_triples.append({
                "subject": topic,
                "relation": "BelongsToCategory",
                "object": cat,
                "source_topic": topic,
                "extraction_method": "wikipedia_category",
            })

        # 3. Redirects -> Synonym
        time.sleep(WIKI_DELAY)
        redirects = get_redirects_to(topic)
        api_calls += 1
        for rd in redirects:
            topic_triples.append({
                "subject": topic,
                "relation": "Synonym",
                "object": rd,
                "source_topic": topic,
                "extraction_method": "wikipedia_redirect",
            })

        all_triples.extend(topic_triples)
        done.add(topic)
        new_count += 1

        # Checkpoint
        if new_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_triples, done)
            tqdm.write(f"  Checkpoint: {len(done)} topics, {len(all_triples)} triples, {api_calls} API calls")

    return all_triples


def save_results(triples: list[dict]):
    """Save structural triples and print stats."""
    # JSONL
    jsonl_path = TRIPLES_DIR / "structural_triples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # JSON
    json_path = TRIPLES_DIR / "structural_triples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    # Stats
    relation_counts = Counter(t["relation"] for t in triples)
    method_counts = Counter(t["extraction_method"] for t in triples)

    stats = {
        "total_triples": len(triples),
        "unique_subjects": len(set(t["subject"] for t in triples)),
        "relation_distribution": dict(relation_counts.most_common()),
        "method_distribution": dict(method_counts.most_common()),
    }

    stats_path = TRIPLES_DIR / "structural_extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STRUCTURAL EXTRACTION COMPLETE")
    print(f"  Total triples:   {len(triples)}")
    print(f"  Unique subjects: {stats['unique_subjects']}")
    for rel, count in relation_counts.most_common():
        print(f"  {rel}: {count}")
    print(f"  Saved: {jsonl_path.name}")
    print(f"{'='*60}")

    # Clean up checkpoints
    for f in ["_structural_checkpoint_triples.json", "_structural_checkpoint_done.json"]:
        p = TRIPLES_DIR / f
        if p.exists():
            p.unlink()
    print("Checkpoints cleaned up.")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structural triples from Wikipedia")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N topics (0=all)")
    args = parser.parse_args()

    topic_set = load_topic_set()
    topics = load_corpus_topics()

    if args.limit > 0:
        topics = topics[:args.limit]
        print(f"Limited to {len(topics)} topics")

    print(f"Topic set for link filtering: {len(topic_set)} entries")

    triples = extract_structural_triples(topics, topic_set, resume=args.resume)
    save_results(triples)

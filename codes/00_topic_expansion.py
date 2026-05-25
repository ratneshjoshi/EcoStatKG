"""
Step 0: Topic Expansion via Wikipedia Category Crawl
=====================================================
Starts from a small set of ROOT CATEGORIES (not articles) and crawls
the Wikipedia category tree to discover all relevant article titles
in the environmental sustainability domain.

Approach:
  1. Define root categories (e.g., "Sustainability", "Climate change")
  2. BFS through subcategories up to max_depth
  3. Collect all member articles from each category
  4. Deduplicate articles
  5. Embed all candidate titles, score by cosine similarity to domain centroid
  6. Accept candidates above threshold
  7. Output expanded topic list

Wikipedia API: free, no auth needed
Embedding API: LLM provider (30 calls/min, batched)

Usage:
    python 00_topic_expansion.py --dry-run          # crawl only, see counts
    python 00_topic_expansion.py                    # full run with embedding
    python 00_topic_expansion.py --threshold 0.60   # more permissive filter
"""

import json
import time
import math
import argparse
import numpy as np
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from collections import defaultdict, Counter

from config import (
    DATA_DIR, REPORTS_DIR,
    API_KEY, API_BASE_URL, HTTP_CLIENT,
    EMBEDDING_MODEL, MAX_CALLS_PER_MINUTE,
)
from utils import create_client, embed_batch


# =============================================================
# ROOT CATEGORIES - the single source of truth for domain scope
# =============================================================
# These are Wikipedia category names (not article titles).
# The crawl expands from these roots through subcategories.

ROOT_CATEGORIES = [
    "Sustainability",
    "Climate change",
    "Renewable energy",
    "Environmental protection",
    "Pollution",
    "Natural resource management",
    "Ecology",
    "Conservation biology",
    "Environmental science",
    "Green building",
    "Waste management",
    "Water resources management",
]

# The "domain anchor" - used to compute the centroid for scoring.
# These are well-known article titles that define what "on-topic" means.
DOMAIN_ANCHOR_TITLES = [
    "Environmental sustainability",
    "Sustainability",
    "Climate change",
    "Renewable energy",
    "Solar energy",
    "Wind power",
    "Carbon dioxide",
    "Greenhouse gas",
    "Deforestation",
    "Biodiversity",
    "Pollution",
    "Recycling",
    "Water conservation",
    "Carbon footprint",
    "Fossil fuel",
    "Global warming",
    "Ozone depletion",
    "Sustainable development",
    "Electric vehicle",
    "Energy efficiency",
    "Carbon capture and storage",
    "Sea level rise",
    "Coral reef",
    "Endangered species",
    "Air quality",
    "Soil conservation",
    "Organic farming",
    "Hydroelectric power",
    "Geothermal energy",
    "Biomass",
]


# === Wikipedia API ===

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "EcoStatKG-Builder/1.0 (research project)"}

# Categories to skip (maintenance, meta, non-topical)
SKIP_PATTERNS = [
    "articles", "pages", "cs1", "webarchive", "wikidata", "commons",
    "short description", "use dmy", "use mdy", "all stub", "wikipedia",
    "harv and sfn", "lacking", "needing", "cleanup", "accuracy disputes",
    "living people", "births", "deaths", "alumni", "template",
    "redirects", "disambiguation", "good articles", "featured articles",
    "by country", "by continent", "by year", "by decade", "by century",
    "organizations", "companies", "people in", "awards",
    # Block obvious drift paths found in dry-run analysis
    "ships sunk", "scuttled", "vessels", "shipwrecks", "naval",
    "epidemiology", "property law", "real estate", "stubs",
    "music", "films", "football", "cricket", "basketball",
    "biography", "politicians", "elections", "political parties",
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
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                print(f"    429 rate limit, waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue
            print(f"  Wiki HTTP error {e.code}")
            return None
        except Exception as e:
            print(f"  Wiki error: {e}")
            return None
    print("  Wiki API: all retries exhausted")
    return None


def get_category_members(category: str, cmtype: str = "page", limit: int = 500) -> list[str]:
    """Get article/subcategory titles from a Wikipedia category."""
    members = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": cmtype,
            "cmlimit": str(min(limit, 500)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = wiki_api_call(params)
        if not data or "query" not in data:
            break
        for m in data["query"].get("categorymembers", []):
            title = m.get("title", "")
            if cmtype == "subcat":
                title = title.replace("Category:", "")
            members.append(title)
        # Handle continuation
        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break
    return members


def should_skip_category(name: str) -> bool:
    """Check if a category name is maintenance/meta and should be skipped."""
    lower = name.lower()
    return any(p in lower for p in SKIP_PATTERNS)


# === Embedding helpers ===

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# === Main crawl logic ===

def crawl_from_root_categories(
    root_categories: list[str],
    max_depth: int = 3,
    max_articles: int = 30000,
) -> dict[str, dict]:
    """
    BFS through Wikipedia category tree starting from root categories.
    Returns dict: {article_title: {"depth": int, "category": str}}
    """
    print(f"\n{'='*60}")
    print("Phase 1: Crawling Wikipedia category tree")
    print(f"  Root categories: {len(root_categories)}")
    print(f"  Max depth: {max_depth}")
    print(f"  Max articles: {max_articles}")
    print(f"{'='*60}")

    articles = {}  # title -> {depth, category}
    visited_cats = set()
    queue = [(cat, 0) for cat in root_categories]  # (category_name, depth)
    queue_idx = 0
    api_calls = 0

    while queue_idx < len(queue) and len(articles) < max_articles:
        cat_name, depth = queue[queue_idx]
        queue_idx += 1

        if cat_name in visited_cats:
            continue
        visited_cats.add(cat_name)

        if should_skip_category(cat_name):
            continue

        # Get member articles
        time.sleep(0.4)  # ~150 req/min, safe margin
        members = get_category_members(cat_name, cmtype="page")
        api_calls += 1
        new_count = 0
        for title in members:
            if title not in articles:
                articles[title] = {"depth": depth, "category": cat_name}
                new_count += 1

        # Get subcategories (only if depth < max_depth)
        if depth < max_depth:
            time.sleep(0.4)
            subcats = get_category_members(cat_name, cmtype="subcat")
            api_calls += 1
            for sc in subcats:
                if sc not in visited_cats and not should_skip_category(sc):
                    queue.append((sc, depth + 1))

        # Progress reporting
        if queue_idx % 25 == 0:
            print(f"  [{queue_idx}/{len(queue)}] cats explored, "
                  f"{len(articles)} articles, "
                  f"depth={depth}, "
                  f"API calls={api_calls}")

        # Early exit if we have enough
        if len(articles) >= max_articles:
            print(f"  Reached {max_articles} articles limit, stopping crawl")
            break

    # Final stats
    print(f"\nCrawl complete:")
    print(f"  Categories explored: {len(visited_cats)}")
    print(f"  Categories in queue: {len(queue)}")
    print(f"  Articles found: {len(articles)}")
    print(f"  API calls: {api_calls}")

    # Depth distribution
    depth_dist = Counter(info["depth"] for info in articles.values())
    for d in sorted(depth_dist):
        print(f"  Depth {d}: {depth_dist[d]} articles")

    # Top categories by article count
    cat_counts = Counter(info["category"] for info in articles.values())
    print(f"\nTop 15 categories by article count:")
    for cat, count in cat_counts.most_common(15):
        print(f"  [{count:5d}] {cat}")

    return articles


def score_and_filter(
    client,
    articles: dict[str, dict],
    anchor_titles: list[str],
    threshold: float = 0.65,
    batch_size: int = 50,
) -> list[dict]:
    """
    Embed all articles, score by cosine similarity to domain centroid.
    """
    print(f"\n{'='*60}")
    print("Phase 2: Embedding & scoring")
    print(f"{'='*60}")

    # Step 1: Compute domain centroid from anchor titles
    print(f"\nEmbedding {len(anchor_titles)} domain anchor titles...")
    anchor_embeddings = embed_batch(
        client, anchor_titles,
        batch_size=batch_size,
        max_calls_per_minute=MAX_CALLS_PER_MINUTE,
    )
    centroid = np.array(anchor_embeddings).mean(axis=0)
    print(f"Domain centroid computed (dim={len(centroid)})")

    # Step 2: Embed all candidate article titles
    titles = list(articles.keys())
    print(f"\nEmbedding {len(titles)} candidate articles...")
    candidate_embeddings = embed_batch(
        client, titles,
        batch_size=batch_size,
        max_calls_per_minute=MAX_CALLS_PER_MINUTE,
    )

    # Step 3: Score each
    print(f"\nScoring (threshold >= {threshold})...")
    scored = []
    for i, title in enumerate(titles):
        emb = np.array(candidate_embeddings[i])
        score = cosine_similarity(emb, centroid)
        info = articles[title]
        scored.append({
            "title": title,
            "score": round(score, 4),
            "depth": info["depth"],
            "category": info["category"],
        })

    scored.sort(key=lambda x: -x["score"])

    accepted = [s for s in scored if s["score"] >= threshold]
    rejected = [s for s in scored if s["score"] < threshold]

    # Stats
    scores = [s["score"] for s in scored]
    print(f"\nScoring results:")
    print(f"  Total: {len(scored)}")
    print(f"  Accepted (>= {threshold}): {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Mean: {np.mean(scores):.4f}")

    print(f"\nTop 30 accepted:")
    for s in accepted[:30]:
        print(f"  {s['score']:.4f}  [d={s['depth']}] {s['title']}")

    if rejected:
        print(f"\nTop 10 rejected (closest to threshold):")
        for s in rejected[:10]:
            print(f"  {s['score']:.4f}  [d={s['depth']}] {s['title']}")

    return scored


def save_results(
    scored: list[dict],
    threshold: float,
    target: int,
):
    """Save expanded topic list and reports."""
    accepted = [s for s in scored if s["score"] >= threshold]

    if len(accepted) > target:
        accepted = accepted[:target]
        print(f"Capped to top {target} by score")

    # Save topic list
    expanded_path = DATA_DIR / "expanded_topics.txt"
    with open(expanded_path, "w", encoding="utf-8") as f:
        for s in accepted:
            f.write(s["title"] + "\n")
    print(f"\nSaved {len(accepted)} topics to {expanded_path.name}")

    # Also update seed_topics.txt to be the expanded list
    # (so downstream scripts pick it up automatically)
    seed_path = DATA_DIR / "seed_topics.txt"
    # Backup original seeds first
    backup_path = DATA_DIR / "original_seeds_backup.txt"
    if seed_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(seed_path, backup_path)
        print(f"Backed up original seeds to {backup_path.name}")

    with open(seed_path, "w", encoding="utf-8") as f:
        for s in accepted:
            f.write(s["title"] + "\n")
    print(f"Updated {seed_path.name} with {len(accepted)} topics")

    # Save detailed report
    scores_all = [s["score"] for s in scored]
    report = {
        "root_categories": ROOT_CATEGORIES,
        "anchor_titles": DOMAIN_ANCHOR_TITLES,
        "threshold": threshold,
        "target": target,
        "total_crawled": len(scored),
        "accepted": len(accepted),
        "rejected": len(scored) - len(accepted),
        "score_stats": {
            "min": float(min(scores_all)),
            "p25": float(np.percentile(scores_all, 25)),
            "median": float(np.median(scores_all)),
            "mean": float(np.mean(scores_all)),
            "p75": float(np.percentile(scores_all, 75)),
            "max": float(max(scores_all)),
        },
        "depth_distribution": dict(Counter(s["depth"] for s in accepted)),
        "category_distribution": dict(Counter(s["category"] for s in accepted).most_common(50)),
        "top_50": accepted[:50],
        "bottom_rejected_50": [s for s in scored if s["score"] < threshold][:50],
    }

    report_path = REPORTS_DIR / "topic_expansion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {report_path.name}")

    # Save all scores for analysis
    all_path = REPORTS_DIR / "all_candidate_scores.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"Saved all scores to {all_path.name}")

    return accepted


def main():
    parser = argparse.ArgumentParser(
        description="Expand topics via Wikipedia category crawl"
    )
    parser.add_argument("--target", type=int, default=10000,
                        help="Max number of topics to keep")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Cosine similarity threshold")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Max subcategory depth")
    parser.add_argument("--max-articles", type=int, default=30000,
                        help="Max raw articles to crawl before scoring")
    parser.add_argument("--dry-run", action="store_true",
                        help="Crawl only, skip embedding")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Embedding batch size")
    args = parser.parse_args()

    t_start = time.time()

    # Phase 1: Crawl
    articles = crawl_from_root_categories(
        ROOT_CATEGORIES,
        max_depth=args.max_depth,
        max_articles=args.max_articles,
    )

    if args.dry_run:
        print(f"\n[DRY RUN] {len(articles)} articles found. Skipping embedding.")
        dry_path = REPORTS_DIR / "dry_run_candidates.txt"
        with open(dry_path, "w", encoding="utf-8") as f:
            for title in sorted(articles.keys()):
                info = articles[title]
                f.write(f"{info['depth']}\t{title}\t{info['category']}\n")
        print(f"Saved to {dry_path.name}")
        elapsed = time.time() - t_start
        print(f"Time: {elapsed/60:.1f} min")
        return

    # Phase 2: Score
    client = create_client()
    scored = score_and_filter(
        client, articles, DOMAIN_ANCHOR_TITLES,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    # Phase 3: Save
    accepted = save_results(scored, args.threshold, args.target)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"TOPIC EXPANSION COMPLETE")
    print(f"  Root categories: {len(ROOT_CATEGORIES)}")
    print(f"  Articles crawled: {len(articles)}")
    print(f"  Articles accepted: {len(accepted)}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

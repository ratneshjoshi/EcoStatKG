"""
Step 1: Corpus Extraction
=========================
Fetches Wikipedia article text for the curated topic list (output of step 00).
Filters sentences containing statistical/numerical content.
Saves both per-topic corpus and flat sentence list for downstream extraction.

Features:
  - Checkpoint every 100 topics (resumes from last save on restart)
  - Rate-limited Wikipedia API (0.3s between requests)
  - Keeps both statistical sentences and full text per topic

Usage:
    python 01_corpus_extraction.py               # all topics
    python 01_corpus_extraction.py --resume       # resume from checkpoint
    python 01_corpus_extraction.py --limit 500    # process first 500 only
"""

import json
import re
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import wikipediaapi

from config import CORPUS_DIR, DATA_DIR, SEED_TOPICS_PATH


CHECKPOINT_EVERY = 100  # save progress every N topics
WIKI_DELAY = 0.3        # seconds between Wikipedia requests


def load_topics(path: Path = SEED_TOPICS_PATH) -> list[str]:
    """Load curated topics from file (output of step 00), one per line."""
    if not path.exists():
        # Try expanded_topics.txt as fallback
        alt = path.parent / "expanded_topics.txt"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"No topic file found at {path} or {alt}")
    with open(path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(topics)} topics from {path.name}")
    return topics


def load_checkpoint(output_dir: Path) -> dict:
    """Load checkpoint corpus if it exists."""
    ckpt = output_dir / "corpus_checkpoint.json"
    if ckpt.exists():
        with open(ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded checkpoint: {len(data)} topics already extracted")
        return data
    return {}


def save_checkpoint(corpus: dict, output_dir: Path):
    """Save intermediate checkpoint."""
    ckpt = output_dir / "corpus_checkpoint.json"
    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"  Checkpoint saved: {len(corpus)} topics")


def fetch_wikipedia_text(topic: str, wiki: wikipediaapi.Wikipedia) -> str | None:
    """Fetch the full text of a Wikipedia article."""
    page = wiki.page(topic)
    if not page.exists():
        return None
    return page.text


def has_numerical_content(sentence: str) -> bool:
    """
    Check if a sentence contains statistical/numerical content.
    Looks for: percentages, numbers with units, decimal numbers,
    dates with statistics, comparative quantities.
    """
    patterns = [
        r'\d+\.?\d*\s*%',           # Percentages: 13.1%, 90%
        r'\d+\.?\d*\s*(GW|MW|kW|TWh|MWh|GWh)',  # Energy units
        r'\d+\.?\d*\s*(million|billion|trillion|thousand)',  # Large numbers
        r'\d+\.?\d*\s*(tons?|tonnes?|kilograms?|kg)',  # Weight
        r'\d+\.?\d*\s*(hectares?|acres?|km²|square)',  # Area
        r'\d+\.?\d*\s*(gallons?|liters?|litres?|cubic)',  # Volume
        r'\d+\.?\d*\s*(per\s+capita|per\s+year|annually)',  # Rates
        r'(approximately|about|nearly|over|up\s+to|around)\s+\d+',  # Approximate values
        r'\$\s*\d+',                 # Dollar amounts
        r'\d{4}\s*(to|and|through)\s*\d{4}',  # Date ranges with context
        r'(increased?|decreased?|grew?|declined?|rose?|fell?)\s+(by\s+)?\d+',  # Changes
        r'\d+\.?\d*x\s',            # Multipliers: 4x, 2.5x
        r'\d+\.?\d*\s*-\s*fold',    # N-fold increases
    ]
    return any(re.search(p, sentence, re.IGNORECASE) for p in patterns)


def extract_statistical_sentences(text: str) -> list[str]:
    """
    Split text into sentences and keep only those with numerical/statistical content.
    Also filters out very short sentences (< 50 chars) and reference-heavy lines.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    statistical = []
    for sent in sentences:
        sent = sent.strip()
        # Skip short fragments
        if len(sent) < 50:
            continue
        # Skip reference/citation lines
        if sent.startswith("^") or "== " in sent or sent.startswith("ISBN"):
            continue
        # Keep if it has numerical content
        if has_numerical_content(sent):
            statistical.append(sent)

    return statistical


def build_corpus(
    topics: list[str],
    output_dir: Path = CORPUS_DIR,
    resume: bool = False,
) -> dict:
    """
    Main corpus construction pipeline.
    Fetches Wikipedia text, filters statistical sentences, checkpoints progress.
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="EcoStatKG-Research/1.0 (research project)",
        language="en"
    )

    # Resume from checkpoint if requested
    corpus = load_checkpoint(output_dir) if resume else {}
    already_done = set(corpus.keys())
    remaining = [t for t in topics if t not in already_done]

    if already_done:
        print(f"Skipping {len(already_done)} already-extracted topics")
    print(f"Processing {len(remaining)} remaining topics...")

    stats = {
        "topics_found": len([v for v in corpus.values() if v.get("sentence_count", 0) > 0]),
        "topics_missing": 0,
        "topics_no_stats": 0,
        "total_sentences": sum(v.get("sentence_count", 0) for v in corpus.values()),
    }
    new_count = 0

    for i, topic in enumerate(tqdm(remaining, desc="Extracting Wikipedia articles")):
        text = fetch_wikipedia_text(topic, wiki)
        if text is None:
            stats["topics_missing"] += 1
            time.sleep(WIKI_DELAY)
            continue

        sentences = extract_statistical_sentences(text)

        # Keep topic even if no statistical sentences (full_text useful for structural extraction)
        corpus[topic] = {
            "sentences": sentences,
            "full_text": text,
            "sentence_count": len(sentences),
            "text_length": len(text),
        }

        if sentences:
            stats["topics_found"] += 1
            stats["total_sentences"] += len(sentences)
        else:
            stats["topics_no_stats"] += 1

        new_count += 1
        time.sleep(WIKI_DELAY)

        # Checkpoint
        if new_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(corpus, output_dir)

    # Final save
    output_file = output_dir / "statistical_corpus.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    # Flat sentence list (for chunk-based RAG baseline)
    all_sentences = []
    for topic, data in corpus.items():
        for sent in data.get("sentences", []):
            all_sentences.append({"topic": topic, "sentence": sent})

    flat_file = output_dir / "all_statistical_sentences.json"
    with open(flat_file, "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, indent=2, ensure_ascii=False)

    # Clean up checkpoint
    ckpt = output_dir / "corpus_checkpoint.json"
    if ckpt.exists():
        ckpt.unlink()
        print("Removed checkpoint (extraction complete)")

    print(f"\n{'='*60}")
    print(f"CORPUS EXTRACTION COMPLETE")
    print(f"  Topics with stats:    {stats['topics_found']}")
    print(f"  Topics without stats: {stats['topics_no_stats']}")
    print(f"  Topics missing:       {stats['topics_missing']}")
    print(f"  Total sentences:      {stats['total_sentences']}")
    print(f"  Corpus file:          {output_file.name}")
    print(f"  Flat sentences:       {flat_file.name} ({len(all_sentences)} entries)")
    print(f"{'='*60}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wikipedia corpus")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N topics (0=all)")
    args = parser.parse_args()

    topics = load_topics()
    if args.limit > 0:
        topics = topics[:args.limit]
        print(f"Limited to first {args.limit} topics")

    build_corpus(topics, resume=args.resume)

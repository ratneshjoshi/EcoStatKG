"""
Step 03: Entity Normalization
=============================
Resolves entity surface-form variations in both LLM and structural triples
using Wikipedia redirect resolution and deterministic string normalization.

This is critical for KG quality:
  - "CO2 emissions" vs "Carbon dioxide emissions" vs "CO2 Emissions"
  - "Solar PV" vs "Solar photovoltaics" vs "Photovoltaic system"
  - "EU" vs "European Union"

Method:
  1. Collect all unique entity strings (subjects + objects) from both triple files
  2. Build a redirect map via Wikipedia API (batch queries)
  3. Apply deterministic normalization: strip whitespace, title-case, merge redirects
  4. Rewrite both triple files in-place with normalized entities
  5. Report: merges performed, entity count reduction

Usage:
    python 03_entity_normalization.py
    python 03_entity_normalization.py --dry-run         # report only, no file changes
    python 03_entity_normalization.py --skip-wikipedia   # skip API calls, string-only normalization
"""

import json
import re
import time
import argparse
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import httpx
from tqdm import tqdm

from config import DATA_DIR, TRIPLES_DIR, REPORTS_DIR


# ---------------------------------------------------------------------------
# 1. String normalization (deterministic, no API)
# ---------------------------------------------------------------------------

def normalize_string(s: str) -> str:
    """Deterministic string normalization for entity matching."""
    if not s or not isinstance(s, str):
        return s

    # Unicode normalize (NFC)
    s = unicodedata.normalize("NFC", s)

    # Strip leading/trailing whitespace
    s = s.strip()

    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)

    # Remove trailing periods (common in LLM output)
    s = s.rstrip(".")

    return s


def canonical_key(s: str) -> str:
    """Generate a canonical lookup key for grouping near-identical strings."""
    k = normalize_string(s)
    # Lowercase for comparison
    k = k.lower()
    # Remove articles
    k = re.sub(r"\b(the|a|an)\b", "", k)
    # Remove possessives
    k = k.replace("'s", "").replace("'s", "")
    # Strip non-alphanumeric except spaces
    k = re.sub(r"[^a-z0-9 ]", "", k)
    # Collapse whitespace again
    k = re.sub(r"\s+", " ", k).strip()
    return k


# ---------------------------------------------------------------------------
# 2. Wikipedia redirect resolution (API-based)
# ---------------------------------------------------------------------------

def fetch_redirects_batch(titles: list[str], session: httpx.Client) -> dict[str, str]:
    """
    Query Wikipedia API for redirect resolution.
    Returns {original_title: canonical_title} for titles that redirect.
    Titles that don't redirect are NOT included.
    Uses batch queries (up to 50 titles per request).
    """
    redirects = {}
    BATCH_SIZE = 50  # Wikipedia API limit

    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i:i + BATCH_SIZE]
        titles_param = "|".join(batch)

        try:
            resp = session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "titles": titles_param,
                    "redirects": 1,
                    "format": "json",
                    "formatversion": 2,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            # Process redirects
            for redir in data.get("query", {}).get("redirects", []):
                from_title = redir["from"]
                to_title = redir["to"]
                redirects[from_title] = to_title

            # Process normalized titles (e.g., first-letter capitalization)
            for norm in data.get("query", {}).get("normalized", []):
                from_title = norm["from"]
                to_title = norm["to"]
                # Only add if not already redirected
                if from_title not in redirects:
                    redirects[from_title] = to_title

        except Exception as e:
            tqdm.write(f"  Wikipedia API error for batch starting '{batch[0]}': {e}")

        # Respectful rate limiting
        time.sleep(0.2)

    return redirects


# ---------------------------------------------------------------------------
# 3. Build normalization map
# ---------------------------------------------------------------------------

def build_normalization_map(
    entities: set[str],
    skip_wikipedia: bool = False,
) -> dict[str, str]:
    """
    Build a mapping from original entity string to its normalized form.
    
    Steps:
      1. String normalization (whitespace, unicode, etc.)
      2. Wikipedia redirect resolution (optional)
      3. Canonical-key grouping: pick the most frequent form as representative
    
    Returns: {original_string: normalized_string}
    """
    norm_map = {}

    # --- Step 1: Basic string normalization ---
    string_normed = {}
    for e in entities:
        normed = normalize_string(e)
        string_normed[e] = normed
        if normed != e:
            norm_map[e] = normed

    # Unique normalized forms
    unique_normed = set(string_normed.values())
    print(f"  String normalization: {len(entities)} -> {len(unique_normed)} unique forms")

    # --- Step 2: Wikipedia redirect resolution ---
    wiki_redirects = {}
    if not skip_wikipedia:
        # Only query entities that look like proper titles (not pure numeric values)
        title_candidates = [
            e for e in unique_normed
            if len(e) > 3
            and not re.match(r"^[\d.,% $]+$", e)
            and not re.match(r"^\d", e)  # Skip "2,700 TWh in 2025" etc.
            and len(e) < 200  # Skip very long strings (these are values, not entities)
        ]
        print(f"  Wikipedia redirect query: {len(title_candidates)} title candidates")

        session = httpx.Client(verify=False, timeout=30.0)
        try:
            for i in tqdm(range(0, len(title_candidates), 50),
                          desc="Resolving redirects", unit="batch"):
                batch = title_candidates[i:i + 50]
                batch_redirects = fetch_redirects_batch(batch, session)
                wiki_redirects.update(batch_redirects)
        finally:
            session.close()

        print(f"  Wikipedia redirects found: {len(wiki_redirects)}")

        # Apply redirects on top of string normalization
        for original, string_norm in list(string_normed.items()):
            if string_norm in wiki_redirects:
                resolved = wiki_redirects[string_norm]
                norm_map[original] = resolved
                string_normed[original] = resolved

    # --- Step 3: Canonical-key grouping ---
    # Group all forms by canonical key (lowercase, no articles, no punctuation)
    key_groups = defaultdict(list)
    for e in set(string_normed.values()):
        ck = canonical_key(e)
        if ck:
            key_groups[ck].append(e)

    # For groups with multiple variants, pick the shortest as representative
    # (shortest is usually the most canonical: "Solar energy" over "Solar Energy Sources")
    merge_count = 0
    for ck, variants in key_groups.items():
        if len(variants) <= 1:
            continue
        # Pick the shortest variant (tie-break: alphabetical)
        representative = min(variants, key=lambda x: (len(x), x))
        for v in variants:
            if v != representative:
                merge_count += 1
                # Update norm_map for all originals that map to this variant
                for original, current in list(string_normed.items()):
                    if current == v:
                        norm_map[original] = representative

    print(f"  Canonical-key merges: {merge_count}")

    # Remove identity mappings
    norm_map = {k: v for k, v in norm_map.items() if k != v}

    return norm_map


# ---------------------------------------------------------------------------
# 4. Apply normalization to triple files
# ---------------------------------------------------------------------------

def normalize_triples_file(
    filepath: Path,
    norm_map: dict[str, str],
    fields: list[str] = ["subject", "object"],
) -> dict:
    """
    Read a JSONL triple file, apply normalization, write back.
    Returns stats about changes made.
    """
    lines = filepath.read_text(encoding="utf-8").splitlines()
    stats = {"total": 0, "modified": 0, "field_changes": Counter()}
    output_lines = []

    for line in lines:
        if not line.strip():
            continue
        triple = json.loads(line)
        stats["total"] += 1
        modified = False

        for field in fields:
            original = triple.get(field, "")
            if original in norm_map:
                triple[field] = norm_map[original]
                stats["field_changes"][field] += 1
                modified = True

        if modified:
            stats["modified"] += 1
        output_lines.append(json.dumps(triple, ensure_ascii=False))

    # Atomic write
    tmp_path = filepath.with_suffix(".jsonl.tmp")
    tmp_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    tmp_path.replace(filepath)

    return stats


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------

def deduplicate_triples(filepath: Path) -> dict:
    """
    Remove exact duplicate triples (same subject, relation, object).
    Returns stats.
    """
    lines = filepath.read_text(encoding="utf-8").splitlines()
    seen = set()
    unique_lines = []
    dup_count = 0

    for line in lines:
        if not line.strip():
            continue
        triple = json.loads(line)
        key = (triple["subject"], triple["relation"], triple["object"])
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique_lines.append(line)

    if dup_count > 0:
        tmp_path = filepath.with_suffix(".jsonl.tmp")
        tmp_path.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
        tmp_path.replace(filepath)

    return {"before": len(lines), "after": len(unique_lines), "duplicates_removed": dup_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entity normalization for KG triples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, do not modify files")
    parser.add_argument("--skip-wikipedia", action="store_true",
                        help="Skip Wikipedia API calls, string-only normalization")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 03: Entity Normalization")
    print("=" * 60)

    kg_path = TRIPLES_DIR / "kg_triples.jsonl"
    st_path = TRIPLES_DIR / "structural_triples.jsonl"

    # --- Collect all entities ---
    print("\n[1/5] Collecting entities from triple files...")
    all_entities = set()
    entity_counts = Counter()  # track frequency for reporting

    for fpath, label in [(kg_path, "LLM"), (st_path, "Structural")]:
        if not fpath.exists():
            print(f"  WARNING: {fpath.name} not found, skipping")
            continue
        count = 0
        for line in open(fpath, encoding="utf-8"):
            t = json.loads(line.strip())
            for field in ["subject", "object"]:
                val = t.get(field, "")
                if val:
                    all_entities.add(val)
                    entity_counts[val] += 1
            count += 1
        print(f"  {label}: {count:,} triples loaded")

    print(f"  Total unique entity strings: {len(all_entities):,}")

    # --- Build normalization map ---
    print("\n[2/5] Building normalization map...")
    norm_map = build_normalization_map(
        all_entities,
        skip_wikipedia=args.skip_wikipedia,
    )
    print(f"  Entities to normalize: {len(norm_map):,}")

    # --- Report top merges ---
    if norm_map:
        print("\n[3/5] Sample normalizations (first 30):")
        samples = list(norm_map.items())[:30]
        for original, normalized in samples:
            freq = entity_counts.get(original, 0)
            print(f"    '{original}' -> '{normalized}' (freq={freq})")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        # Save normalization map for inspection
        map_path = REPORTS_DIR / "normalization_map_preview.json"
        map_path.write_text(
            json.dumps(norm_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Normalization map saved to {map_path}")
        return

    # --- Apply normalization ---
    print("\n[3/5] Applying normalization...")
    results = {}
    for fpath, label in [(kg_path, "LLM"), (st_path, "Structural")]:
        if not fpath.exists():
            continue
        stats = normalize_triples_file(fpath, norm_map)
        results[label] = stats
        print(f"  {label}: {stats['modified']:,}/{stats['total']:,} triples modified")
        for field, count in stats["field_changes"].items():
            print(f"    {field} changes: {count:,}")

    # --- Deduplicate ---
    print("\n[4/5] Deduplicating triples...")
    dedup_results = {}
    for fpath, label in [(kg_path, "LLM"), (st_path, "Structural")]:
        if not fpath.exists():
            continue
        stats = deduplicate_triples(fpath)
        dedup_results[label] = stats
        print(f"  {label}: {stats['duplicates_removed']:,} duplicates removed "
              f"({stats['before']:,} -> {stats['after']:,})")

    # --- Final stats ---
    print("\n[5/5] Post-normalization statistics...")
    final_entities = set()
    final_counts = {"LLM": 0, "Structural": 0}
    for fpath, label in [(kg_path, "LLM"), (st_path, "Structural")]:
        if not fpath.exists():
            continue
        for line in open(fpath, encoding="utf-8"):
            t = json.loads(line.strip())
            final_entities.add(t.get("subject", ""))
            final_entities.add(t.get("object", ""))
            final_counts[label] += 1

    original_entity_count = len(all_entities)
    final_entity_count = len(final_entities)

    print(f"  Entities before: {original_entity_count:,}")
    print(f"  Entities after:  {final_entity_count:,}")
    print(f"  Reduction: {original_entity_count - final_entity_count:,} "
          f"({(1 - final_entity_count / original_entity_count) * 100:.1f}%)")
    print(f"  LLM triples:        {final_counts['LLM']:,}")
    print(f"  Structural triples: {final_counts['Structural']:,}")
    print(f"  Combined total:     {sum(final_counts.values()):,}")

    # --- Save report ---
    report = {
        "original_entity_count": original_entity_count,
        "final_entity_count": final_entity_count,
        "entity_reduction": original_entity_count - final_entity_count,
        "entity_reduction_pct": round((1 - final_entity_count / original_entity_count) * 100, 2),
        "normalization_map_size": len(norm_map),
        "wikipedia_redirects_used": not args.skip_wikipedia,
        "normalization_results": {
            label: {
                "total": r["total"],
                "modified": r["modified"],
                "field_changes": dict(r["field_changes"]),
            }
            for label, r in results.items()
        },
        "deduplication_results": dedup_results,
        "final_triple_counts": final_counts,
        "final_combined_total": sum(final_counts.values()),
    }

    report_path = REPORTS_DIR / "entity_normalization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Report saved to {report_path}")

    # Save full normalization map
    map_path = REPORTS_DIR / "normalization_map.json"
    map_path.write_text(
        json.dumps(norm_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Normalization map saved to {map_path}")

    print("\n" + "=" * 60)
    print("Entity normalization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

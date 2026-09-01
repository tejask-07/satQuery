"""
Text retriever for BigEarthNet.txt remote-sensing VQA dataset.

Lightweight, deterministic retrieval over BigEarthNet.txt.parquet using PyArrow.
Searches row groups without loading the entire 9.5M-row file into RAM.
Provides contextual few-shot demonstration examples for the P4 VLM.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Default path to BigEarthNet.txt.parquet metadata
DEFAULT_PARQUET_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "data",
        "metadata",
        "bigearthnet_txt",
        "BigEarthNet.txt.parquet",
    )
)

# Semantic domain keywords and corresponding categories
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "vegetation": [
        "vegetation",
        "forest",
        "crop",
        "farmland",
        "arable",
        "pasture",
        "tree",
        "green",
        "woodland",
        "agriculture",
        "ndvi",
    ],
    "urban": [
        "urban",
        "building",
        "city",
        "road",
        "settlement",
        "developed",
        "industrial",
        "commercial",
        "built-up",
        "ndbi",
        "fabric",
    ],
    "water": [
        "water",
        "lake",
        "river",
        "reservoir",
        "wetland",
        "marine",
        "inland waters",
        "ndwi",
        "water body",
        "ocean",
    ],
    "spatial": [
        "adjacent",
        "next to",
        "border",
        "near",
        "relative position",
        "boundary",
        "contact",
        "neighbor",
    ],
    "area": [
        "area",
        "cover",
        "coverage",
        "extent",
        "square meters",
        "hectares",
        "size",
    ],
    "count": [
        "how many",
        "count",
        "number of",
        "multiple",
        "quantity",
    ],
    "presence": [
        "is there",
        "are there",
        "presence",
        "contain",
        "appear",
        "identify",
        "detect",
    ],
    "remote_sensing": [
        "satellite",
        "land cover",
        "image",
        "earth observation",
        "scene",
        "patch",
    ],
}

CATEGORY_MAP: Dict[str, List[str]] = {
    "spatial": ["adjacency", "relative pos"],
    "area": ["area"],
    "count": ["count"],
    "presence": ["presence"],
}

# Stopwords to filter out from raw query tokenization
STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "in",
    "of",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "do",
    "does",
    "did",
    "would",
    "you",
    "can",
    "show",
    "between",
    "what",
    "from",
    "with",
    "that",
    "this",
    "these",
    "those",
    "please",
    "tell",
    "me",
    "about",
    "compare",
}

# Global in-memory cache for the primary row group table to enable sub-10ms queries
_CACHED_PARQUET_PATH: Optional[str] = None
_CACHED_PARQUET_FILE: Optional[pq.ParquetFile] = None
_CACHED_TABLES: Dict[int, Any] = {}


def _get_parquet_file(parquet_path: str) -> Optional[pq.ParquetFile]:
    """Get or instantiate a cached ParquetFile handle."""
    global _CACHED_PARQUET_PATH, _CACHED_PARQUET_FILE, _CACHED_TABLES

    if not os.path.isfile(parquet_path):
        logger.warning(
            f"[TEXT RETRIEVER] Parquet file not found at: {parquet_path}"
        )
        return None

    if _CACHED_PARQUET_FILE is not None and _CACHED_PARQUET_PATH == parquet_path:
        return _CACHED_PARQUET_FILE

    try:
        _CACHED_PARQUET_FILE = pq.ParquetFile(parquet_path)
        _CACHED_PARQUET_PATH = parquet_path
        _CACHED_TABLES.clear()
        return _CACHED_PARQUET_FILE
    except Exception as exc:
        logger.error(
            f"[TEXT RETRIEVER] Failed to open ParquetFile at {parquet_path}: {exc}"
        )
        return None


def _get_row_group_table(
    pf: pq.ParquetFile, row_group_idx: int
) -> Optional[Any]:
    """Retrieve a row group PyArrow table with caching for row group 0."""
    global _CACHED_TABLES

    if row_group_idx in _CACHED_TABLES:
        return _CACHED_TABLES[row_group_idx]

    try:
        table = pf.read_row_group(
            row_group_idx,
            columns=[
                "input",
                "output",
                "type",
                "category",
                "s1_name",
                "patch_id",
            ],
        )
        # Cache row group 0 for instant repeat retrieval (occupies ~32MB RAM)
        if row_group_idx == 0:
            _CACHED_TABLES[row_group_idx] = table
        return table
    except Exception as exc:
        logger.warning(
            f"[TEXT RETRIEVER] Failed to read row group {row_group_idx}: {exc}"
        )
        return None


def extract_query_features(query: str) -> Dict[str, Any]:
    """
    Extract search terms, target categories, and domain associations from query text.
    """
    q_lower = query.lower()
    raw_tokens = re.findall(r"\b[a-z]{3,}\b", q_lower)
    content_tokens = [t for t in raw_tokens if t not in STOPWORDS]

    target_categories: Set[str] = set()
    matched_domains: Set[str] = set()
    domain_kws_to_add: List[str] = []

    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in q_lower:
                matched_domains.add(domain)
                domain_kws_to_add.append(kw)
                if domain in CATEGORY_MAP:
                    target_categories.update(CATEGORY_MAP[domain])

    # Prioritize user query's own content tokens, then append domain keywords
    active_keywords: List[str] = list(dict.fromkeys(content_tokens + domain_kws_to_add))
    # Cap keywords at 6 to ensure blazing-fast substring evaluation (<50ms)
    active_keywords = active_keywords[:6]

    return {
        "query_lower": q_lower,
        "tokens": content_tokens,
        "keywords": active_keywords,
        "target_categories": list(target_categories),
        "matched_domains": list(matched_domains),
    }


def retrieve_examples(
    question: str,
    max_examples: int = 3,
    parquet_path: Optional[str] = None,
    max_row_groups_to_scan: int = 2,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant BigEarthNet remote-sensing question/answer examples.

    Args:
        question: User query or investigation question.
        max_examples: Maximum number of examples to return (default 3, range 1-5).
        parquet_path: Path to BigEarthNet.txt.parquet file. Defaults to DEFAULT_PARQUET_PATH.
        max_row_groups_to_scan: Maximum number of row groups to search before returning.

    Returns:
        List of dictionaries with input, output, type, category, s1_name, patch_id.
        Returns [] if file is missing, question is empty, or no matches found.
    """
    if not question or not question.strip():
        return []

    path = parquet_path or DEFAULT_PARQUET_PATH
    pf = _get_parquet_file(path)
    if pf is None:
        return []

    features = extract_query_features(question)
    active_kws = features["keywords"]
    target_cats = set(features["target_categories"])
    tokens = features["tokens"]

    if not active_kws and not target_cats:
        return []

    scored_candidates: List[Dict[str, Any]] = []
    seen_inputs: Set[str] = set()

    total_rgs = min(pf.num_row_groups, max_row_groups_to_scan)

    for rg_idx in range(total_rgs):
        table = _get_row_group_table(pf, rg_idx)
        if table is None or len(table) == 0:
            continue

        inp_col = pc.utf8_lower(table["input"])
        cat_col = pc.utf8_lower(table["category"])

        masks = []
        for kw in active_kws:
            masks.append(pc.match_substring(inp_col, kw))
        for cat in target_cats:
            masks.append(pc.equal(cat_col, cat))

        if not masks:
            continue

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = pc.or_(combined_mask, m)

        filtered = table.filter(combined_mask)
        if len(filtered) == 0:
            continue

        inputs = filtered["input"].to_pylist()
        outputs = filtered["output"].to_pylist()
        types = filtered["type"].to_pylist()
        categories = filtered["category"].to_pylist()
        s1_names = filtered["s1_name"].to_pylist()
        patch_ids = filtered["patch_id"].to_pylist()

        num_filtered = min(600, len(filtered))
        for i in range(num_filtered):
            inp_text = inputs[i]
            if inp_text in seen_inputs:
                continue
            seen_inputs.add(inp_text)

            inp_text_lower = inp_text.lower()
            cat = categories[i]
            score = 0.0

            # 1. Keyword overlap scoring
            for kw in active_kws:
                if kw in inp_text_lower:
                    score += 3.0 if len(kw) > 4 else 2.0

            # 2. Category match bonus
            if cat in target_cats:
                score += 4.0

            # 3. Exact word / phrase match bonus
            for token in tokens:
                if re.search(r"\b" + re.escape(token) + r"\b", inp_text_lower):
                    score += 2.5

            # 4. Multi-word phrase matches from query
            for i_tok in range(len(tokens) - 1):
                bigram = f"{tokens[i_tok]} {tokens[i_tok + 1]}"
                if bigram in inp_text_lower:
                    score += 5.0

            if score > 0:
                scored_candidates.append(
                    {
                        "score": score,
                        "input": inp_text,
                        "question": inp_text,
                        "output": outputs[i],
                        "answer": outputs[i],
                        "type": types[i],
                        "category": cat,
                        "s1_name": s1_names[i],
                        "patch_id": patch_ids[i],
                    }
                )

        # Early exit if we already have sufficient matches from current row group
        if len(scored_candidates) >= max_examples * 2:
            break

    if not scored_candidates:
        return []

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Return top max_examples
    return scored_candidates[:max_examples]


def format_examples_for_prompt(examples: List[Dict[str, Any]]) -> str:
    """
    Format retrieved BigEarthNet examples into a clean demonstration prompt block.
    """
    if not examples:
        return ""

    lines = [
        "BIGEARTHNET REMOTE-SENSING EXAMPLES",
        "===================================",
        "These examples come from the BigEarthNet image-text dataset.",
        "They are examples of remote-sensing question/answer patterns.",
        "They are NOT measurements for the current user imagery.",
        "Do NOT copy their numerical values or answers as facts about the current image.",
        "",
    ]

    for idx, ex in enumerate(examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append(f"Question: {ex['input']}")
        lines.append(f"Answer: {ex['output']}")
        lines.append(f"Category: {ex['category']}")
        lines.append(f"Type: {ex['type']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import time

    print("=== Testing BigEarthNet Text Retriever ===")
    test_queries = [
        "Show vegetation change",
        "Are forests adjacent to farmland?",
        "How much area does pasture cover?",
        "Show urban development",
        "Show water change",
    ]

    for q in test_queries:
        t0 = time.time()
        results = retrieve_examples(q, max_examples=3)
        dt = (time.time() - t0) * 1000
        print(f"\nQuery: '{q}' (found {len(results)} in {dt:.1f}ms)")
        for idx, r in enumerate(results, 1):
            print(f"  [{idx}] Category: {r['category']} | Type: {r['type']} | Score: {r['score']}")
            print(f"      Q: {r['input']}")
            print(f"      A: {r['output']}")

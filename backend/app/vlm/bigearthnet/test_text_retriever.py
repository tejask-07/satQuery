"""
Test suite for BigEarthNet.txt remote-sensing text retriever.

Validates:
1. Retrieval on standard benchmark queries (vegetation, spatial, area, urban, water).
2. Proper schema and structure of returned examples.
3. Speed and absence of ArrowMemoryError.
"""

import sys
import os
import time

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from app.vlm.bigearthnet.text_retriever import (
    retrieve_examples,
    format_examples_for_prompt,
)

BENCHMARK_QUERIES = [
    "Show vegetation change",
    "Are forests adjacent to farmland?",
    "How much area does pasture cover?",
    "Show urban development",
    "Show water change",
]


def test_text_retriever_benchmark():
    """Verify all benchmark queries return valid, structured examples."""
    print("\n=======================================================")
    print("BIGEARTHNET.TXT RETRIEVER BENCHMARK TESTS")
    print("=======================================================")

    for query in BENCHMARK_QUERIES:
        start_time = time.time()
        results = retrieve_examples(query, max_examples=3)
        duration_ms = (time.time() - start_time) * 1000

        print(f"\nQUERY: '{query}'")
        print(f"NUMBER OF RETRIEVED EXAMPLES: {len(results)} (retrieved in {duration_ms:.1f}ms)")
        assert len(results) > 0, f"Expected at least 1 example for query: {query}"
        assert len(results) <= 3

        for i, example in enumerate(results, start=1):
            category = example.get("category")
            q_type = example.get("type")
            question = example.get("input") or example.get("question")
            answer = example.get("output") or example.get("answer")

            assert category, "Missing category"
            assert q_type, "Missing type"
            assert question, "Missing question"
            assert answer is not None, "Missing answer"

            # Print required fields
            print(f"  Example {i}:")
            print(f"    Category / Type: {category} / {q_type}")
            print(f"    Short Question:  {question[:90]}{'...' if len(question) > 90 else ''}")
            print(f"    Short Answer:    {answer[:60]}{'...' if len(str(answer)) > 60 else ''}")

        formatted_block = format_examples_for_prompt(results)
        assert "BIGEARTHNET REMOTE-SENSING EXAMPLES" in formatted_block
        assert "Question:" in formatted_block
        assert "Answer:" in formatted_block


def test_empty_query_returns_empty_list():
    """Empty or whitespace queries should return an empty list without error."""
    assert retrieve_examples("") == []
    assert retrieve_examples("   ") == []


def test_invalid_path_returns_empty_list():
    """Non-existent parquet file path should return [] gracefully."""
    assert retrieve_examples("forest change", parquet_path="non_existent.parquet") == []


if __name__ == "__main__":
    test_text_retriever_benchmark()
    test_empty_query_returns_empty_list()
    test_invalid_path_returns_empty_list()
    print("\n[ALL TEXT RETRIEVER TESTS PASSED]")

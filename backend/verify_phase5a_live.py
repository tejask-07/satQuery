"""
Live Verification Script for Phase 5A: Multi-Index Evidence Calculation.

Query: "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
"""

from pathlib import Path
from pprint import pprint

from app.api.routes_query import process_query
from app.schemas.query import QueryRequest


def main():
    query = "Compare urban change between 2021 and 2025 for AOI [16.40, 48.20, 16.41, 48.21]"
    print("\n========================================================")
    print("RUNNING PHASE 5A LIVE QUERY:")
    print(f"{query}")
    print("========================================================\n")

    req = QueryRequest(query=query)
    result = process_query(req)

    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence}")
    print(f"Primary Metric: {result.statistics.get('metric')}")

    print("\n--------------------------------------------------------")
    print("MULTI-INDEX EVIDENCE (PHASE 5A):")
    print("--------------------------------------------------------")
    ev = result.multi_index_evidence
    assert ev is not None, "multi_index_evidence must be present in response!"

    print(f"Target: {ev.get('target')}")
    print(f"Primary Hypothesis: {ev.get('primary_hypothesis')}")
    print(f"Urban Expansion Support Score: {ev.get('urban_expansion_support')}")

    print("\nComponent Evidence Scores (Bounded [0.0, 1.0]):")
    for comp, score in ev.get("component_evidence", {}).items():
        print(f"  * {comp}: {score}")

    print("\nInspectable Evidence Signals:")
    for sig_name, sig_data in ev.get("signals", {}).items():
        print(f"\n  [{sig_name.upper()} SIGNAL]")
        print(f"    Name: {sig_data.get('display_name')}")
        print(f"    Direction: {sig_data.get('direction')}")
        print(f"    Raw Magnitude: {sig_data.get('raw_magnitude'):.4f}")
        print(f"    Normalized Strength: {sig_data.get('normalized_strength'):.4f}")
        print(f"    Support State: {sig_data.get('support_state')}")
        print(f"    Support Score: {sig_data.get('support_score')}")
        print(f"    Valid: {sig_data.get('valid')}")
        print(f"    Interpretation: {sig_data.get('interpretation')}")

    print("\nCounter-Hypothesis Evaluation:")
    counter = ev.get("counter_hypothesis", {})
    for k, v in counter.items():
        print(f"  * {k}: {v}")

    print("\nMetadata & Thresholds:")
    pprint(ev.get("metadata"))

    print("\n--------------------------------------------------------")
    print("PHASE 4 LAYER INTEGRITY CHECK:")
    print("--------------------------------------------------------")
    assert result.layer_package is not None
    print(f"Before layers: {list(result.layer_package['before'].keys())}")
    print(f"After layers: {list(result.layer_package['after'].keys())}")
    print(f"Change layers: {list(result.layer_package['change'].keys())}")
    print(f"Quality layers: {list(result.layer_package['quality'].keys())}")

    print("\nSUCCESS: Phase 5A Multi-Index Evidence calculation is complete and verified!")


if __name__ == "__main__":
    main()

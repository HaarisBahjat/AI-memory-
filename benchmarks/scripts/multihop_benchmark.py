"""
multihop_benchmark.py — Phase 9.4

Dedicated evaluation of MULTI_HOP queries only.
"""
import os
import sys
import json
import asyncio
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from benchmarks.scripts.statistical_analysis import find_latest_per_query_file

def compute_multihop_metrics(per_query_results: list[dict]) -> dict:
    mh_queries = [r for r in per_query_results if r.get("category") == "MULTI_HOP"]
    n = len(mh_queries)
    if n == 0:
        return {"error": "No MULTI_HOP queries found", "n": 0}

    f1_vals       = [r.get("f1_at_5", 0.0) for r in mh_queries]
    recall_vals   = [r.get("recall_at_5", 0.0) for r in mh_queries]
    hit_vals      = [r.get("hit_rate_at_5", 0.0) for r in mh_queries]

    both_retrieved = 0
    partial_retrieved = 0
    zero_retrieved = 0
    mean_facts_retrieved = 0.0

    for r in mh_queries:
        gt  = set(r.get("ground_truth_ids", []))
        ret = set(r.get("retrieved_ids", [])[:5])
        found = len(gt & ret)
        mean_facts_retrieved += found
        if found == len(gt) and len(gt) > 0:
            both_retrieved += 1
        elif found > 0:
            partial_retrieved += 1
        else:
            zero_retrieved += 1

    mean_facts_retrieved /= n

    return {
        "n_multihop_queries": n,
        "f1_at_5_mean": round(sum(f1_vals) / n, 4),
        "recall_at_5_mean": round(sum(recall_vals) / n, 4),
        "hit_rate_at_5_mean": round(sum(hit_vals) / n, 4),
        "both_facts_retrieved_rate": round(both_retrieved / n, 4),
        "partial_facts_retrieved_rate": round(partial_retrieved / n, 4),
        "zero_facts_retrieved_rate": round(zero_retrieved / n, 4),
        "mean_facts_retrieved_per_query": round(mean_facts_retrieved, 4),
    }


def run_multihop_benchmark():
    print("\n" + "=" * 60)
    print("MULTI-HOP BENCHMARK — Phase 9.4")
    print("=" * 60)

    configs = [
        "baseline_c_vector",
        "baseline_d_recency_decay",
        "proposed_full",
    ]

    comparison = {}
    for cfg in configs:
        pq_file = find_latest_per_query_file(cfg)
        if not pq_file:
            print(f"  [{cfg}] No per-query file found")
            continue
        with open(pq_file) as f:
            pq_results = json.load(f)

        mh_metrics = compute_multihop_metrics(pq_results)
        comparison[cfg] = mh_metrics

        print(f"\n[{cfg}]")
        print(f"  N: {mh_metrics['n_multihop_queries']}")
        if mh_metrics['n_multihop_queries'] > 0:
            print(f"  F1@5:                  {mh_metrics['f1_at_5_mean']}")
            print(f"  Both Facts Retrieved:  {mh_metrics['both_facts_retrieved_rate']*100:.1f}%")
            print(f"  Partial Retrieval:     {mh_metrics['partial_facts_retrieved_rate']*100:.1f}%")
            print(f"  Zero Retrieval:        {mh_metrics['zero_facts_retrieved_rate']*100:.1f}%")
            print(f"  Mean Facts Retrieved:  {mh_metrics['mean_facts_retrieved_per_query']}")
        
    return comparison

if __name__ == "__main__":
    run_multihop_benchmark()

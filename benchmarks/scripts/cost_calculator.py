"""
cost_calculator.py — v2.0  Phase 9.4

Separates API costs from infrastructure costs, ensuring accurate measurement 
of benchmark execution and production estimation.
"""
import os
import sys
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── API Token Prices (text-embedding-3-small & gpt-4o-mini) ──────────────────
EMBEDDING_COST_PER_1K = 0.00002     # $0.02 per 1M tokens
INPUT_COST_PER_1K     = 0.00015     # $0.15 per 1M tokens
OUTPUT_COST_PER_1K    = 0.00060     # $0.60 per 1M tokens


def estimate_benchmark_cost(num_queries=400, num_memories=250):
    """Calculate the API cost of running the benchmark suite."""
    
    # 1. Embedding generation (run once during db seed)
    # Average memory content is ~20 tokens
    embedding_tokens = num_memories * 20
    embed_cost = (embedding_tokens / 1000) * EMBEDDING_COST_PER_1K
    
    # 2. Answer benchmark (RAG generation)
    # Average query: 15 tokens. Average context (top 5): 100 tokens. Output: 40 tokens.
    input_tokens = num_queries * 115
    output_tokens = num_queries * 40
    gen_cost = (input_tokens / 1000) * INPUT_COST_PER_1K + (output_tokens / 1000) * OUTPUT_COST_PER_1K
    
    # 3. LLM Judge
    # Input: 15 (query) + 40 (answer) + 15 (ground truth) + 100 (rubric) = 170 tokens
    # Output: 5 tokens (score + short rationale)
    judge_input = num_queries * 170
    judge_output = num_queries * 5
    judge_cost = (judge_input / 1000) * INPUT_COST_PER_1K + (judge_output / 1000) * OUTPUT_COST_PER_1K
    
    total_api_cost = embed_cost + gen_cost + judge_cost
    
    return {
        "benchmark_api_cost": {
            "embedding_cost_usd": round(embed_cost, 6),
            "generation_cost_usd": round(gen_cost, 6),
            "llm_judge_cost_usd": round(judge_cost, 6),
            "total_api_cost_usd": round(total_api_cost, 6)
        },
        "infrastructure": {
            "type": "Local workstation (no cloud cost)",
            "database": "PostgreSQL 16.x + pgvector",
            "compute": "CPU-bound retrieval",
            "ram_usage_estimate": "16 GB"
        }
    }


def estimate_production_cost_per_100_queries():
    """Estimate the runtime cost of 100 production queries."""
    
    # Per query: Embed the query (15 tokens)
    query_embed_cost = (1500 / 1000) * EMBEDDING_COST_PER_1K
    
    # Per query: Generate answer (115 in, 40 out)
    query_gen_cost = (11500 / 1000) * INPUT_COST_PER_1K + (4000 / 1000) * OUTPUT_COST_PER_1K
    
    total = query_embed_cost + query_gen_cost
    
    return {
        "per_100_queries_api_cost_usd": round(total, 6),
        "note": "Excludes nightly consolidation costs (which run asynchronously)"
    }


def run_cost_analysis():
    print("=" * 60)
    print("COST ANALYSIS — Phase 9.4")
    print("=" * 60)
    
    benchmark_costs = estimate_benchmark_cost()
    prod_costs = estimate_production_cost_per_100_queries()
    
    print("\n[Benchmark Execution Cost (API)]")
    for k, v in benchmark_costs["benchmark_api_cost"].items():
        print(f"  {k}: ${v}")
        
    print("\n[Infrastructure]")
    for k, v in benchmark_costs["infrastructure"].items():
        print(f"  {k}: {v}")
        
    print("\n[Production Runtime Cost (API)]")
    print(f"  Per 100 queries: ${prod_costs['per_100_queries_api_cost_usd']}")
    print(f"  Note: {prod_costs['note']}")
    
    # Save
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cost_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "benchmark_execution": benchmark_costs,
            "production_estimate": prod_costs
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_cost_analysis()

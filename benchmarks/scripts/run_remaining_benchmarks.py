"""
Run only the remaining benchmark configs (skipping A and B which already completed).
"""
import os
import sys
import asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(BASE_DIR))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import the run_benchmark function
from benchmarks.scripts.rag_benchmark import run_benchmark

CONFIGS = [
    "baseline_c_vector",
    "baseline_d_recency_decay",
    "baseline_e_consolidated",
    "proposed_full",
    "no_temporal",
    "no_graph",
    "no_consolidation",
]

async def main():
    for config in CONFIGS:
        config_path = os.path.join(workspace_root, "benchmarks", "configs", f"{config}.yaml")
        if os.path.exists(config_path):
            print(f"\n{'='*50}")
            print(f"Running: {config}")
            print('='*50)
            await run_benchmark(config, debug=False)
        else:
            print(f"SKIP (missing): {config}")

if __name__ == "__main__":
    asyncio.run(main())

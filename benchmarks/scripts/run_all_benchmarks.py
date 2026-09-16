import os
import sys
import asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(BASE_DIR))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

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
    import uuid
    from benchmarks.scripts.generate_dataset import generate_dataset
    from benchmarks.scripts.rag_benchmark import seed_database
    
    run_group = f"final_run_{uuid.uuid4().hex[:8]}"
    os.environ["BENCHMARK_RESULTS_DIR"] = os.path.join(workspace_root, "benchmarks", "results", run_group)
    
    print("="*50)
    print("STEP 1/2: Generating Dataset and Seeding Database")
    print("="*50)
    generate_dataset(num_users=50, seed=42)
    await seed_database()
    
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

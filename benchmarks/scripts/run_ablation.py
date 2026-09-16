import os
import sys
import json
import asyncio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
    
from benchmarks.scripts.rag_benchmark import run_benchmark

CONFIGS_TO_RUN = [
    "full_system",
    "no_temporal",
    "no_graph",
    "no_consolidation"
]

async def main():
    print("========================================")
    print("Starting Ablation Studies")
    print("========================================")
    
    for config in CONFIGS_TO_RUN:
        await run_benchmark(config, debug=False)
        
    print("\n========================================")
    print("Ablation Studies Completed.")
    print("Parsing Results...")
    print("========================================")
    
    results_dir = os.path.join(BASE_DIR, "results")
    runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")], reverse=True)
    
    metrics = {}
    for config in CONFIGS_TO_RUN:
        for run in runs:
            candidate_file = os.path.join(results_dir, run, f"retrieval_{config}.json")
            if os.path.exists(candidate_file):
                with open(candidate_file, "r") as f:
                    metrics[config] = json.load(f)["f1_at_5"]
                break
                
    if "full_system" not in metrics:
        print("Missing full_system metrics.")
        return
        
    full_f1 = metrics["full_system"]
    
    print("\n| System Configuration | F1@5 | ∆ from Full System |")
    print("|----------------------|------|--------------------|")
    print(f"| **Full Proposed System** | **{full_f1:.4f}** | **0.00** |")
    
    for config in CONFIGS_TO_RUN:
        if config == "full_system": continue
        f1 = metrics.get(config, 0.0)
        delta = f1 - full_f1
        print(f"| Ablation: {config} | {f1:.4f} | {delta:+.4f} |")
        
if __name__ == "__main__":
    asyncio.run(main())

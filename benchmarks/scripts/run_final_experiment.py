import os
import sys
import asyncio
import uuid
import json
import subprocess
import platform
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

def run_script(script_path):
    res = subprocess.run(
        [sys.executable, script_path],
        env={**os.environ, "PYTHONPATH": "."},
        cwd=workspace_root
    )
    if res.returncode != 0:
        print(f"Script {script_path} failed!")
        sys.exit(1)

# Final confirmatory ablations: Full vs Full-minus-one + Vector baseline
CONFIGS = [
    "baseline_c_vector",
    "proposed_full",
    "no_temporal",
    "no_graph",
    "no_consolidation",
]

async def main():
    print("="*60)
    print("FINAL PUBLICATION EXPERIMENT PIPELINE")
    print("="*60)

    # 1. Generate unique run ID and directories
    run_id = f"final_run_{uuid.uuid4().hex[:8]}"
    res_dir = os.path.join(BASE_DIR, "results", run_id)
    os.makedirs(res_dir, exist_ok=True)
    os.environ["BENCHMARK_RESULTS_DIR"] = res_dir

    # 2. Reproducibility manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "dev_users": 40,
        "test_users": 10,
        "configs": CONFIGS,
        "os_name": platform.system() + " " + platform.release(),
        "python_version": platform.python_version()
    }

    # 3. Regenerate deterministic dataset and seed DB
    print("\n[1/7] Generating dataset & Seeding database")
    from benchmarks.scripts.generate_dataset import generate_dataset
    from benchmarks.scripts.rag_benchmark import seed_database
    generate_dataset(num_users=50, seed=42)
    await seed_database()

    # 3.5 Integrity and Leakage
    print("\n[2/7] Validating Dataset Integrity")
    run_script("benchmarks/scripts/dataset_integrity.py")

    print("\n[3/7] Running Leakage Audit")
    run_script("benchmarks/scripts/leakage_audit.py")

    # 4. Threshold Sweep on DEV split
    print("\n[4/7] Running threshold sweep on DEV split")
    from benchmarks.scripts.threshold_sweep import main as run_threshold_sweep
    run_script("benchmarks/scripts/threshold_sweep.py")
        
    manifest["selected_threshold"] = 0.40 # Hardcoded after visual check of DEV

    # 5. Execute benchmarks on TEST split
    print("\n[5/7] Executing metrics on TEST split")
    for config in CONFIGS:
        print(f"  -> {config}")
        res = subprocess.run(
            [sys.executable, "benchmarks/scripts/rag_benchmark.py", "--config", config],
            env={**os.environ, "PYTHONPATH": ".", "BENCHMARK_SPLIT": "TEST"},
            cwd=workspace_root
        )
        if res.returncode != 0:
            print(f"Benchmark failed for {config}!")
            sys.exit(1)

    # 6. Statistical Analysis
    print("\n[6/7] Running Statistical Analysis")
    from benchmarks.scripts.statistical_analysis import run_statistics
    run_statistics()

    # Save manifest
    with open(os.path.join(res_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 7. Generate Error Analysis (for proposed_full)
    print("\n[7/7] Generating Error Analysis and Reports")
    try:
        with open(os.path.join(res_dir, "retrieval_proposed_full_per_query.json"), encoding="utf-8") as f:
            per_query = json.load(f)
        
        errors = []
        for q in per_query:
            if q.get("f1_at_5", 1.0) is not None and q.get("f1_at_5", 1.0) < 1.0:
                # Answerable with less than perfect F1
                errors.append({
                    "query_id": q["query_id"],
                    "user_id": q["user_id"],
                    "category": q["category"],
                    "f1_at_5": q["f1_at_5"],
                    "failure_type": "retrieval_miss",
                    "retrieved": [str(x) for x in q.get("retrieved_memory_ids", [])],
                    "expected": [str(x) for x in q.get("ground_truth_memory_ids", [])]
                })
            elif q.get("category") == "NO_ANSWER" and not q.get("abstention_correct", True):
                errors.append({
                    "query_id": q["query_id"],
                    "user_id": q["user_id"],
                    "category": q["category"],
                    "subtype": q.get("no_answer_type"),
                    "failure_type": "abstention_failure",
                    "retrieved": [str(x) for x in q.get("retrieved_memory_ids", [])]
                })
        
        with open(os.path.join(res_dir, "error_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
    except Exception as e:
        print(f"Failed to generate error analysis: {e}")

    # Generate Final Report
    run_script("benchmarks/scripts/generate_final_report.py")

    # Run Final Self-Audit Consistency Checker
    print("\nRunning Final Self-Audit Checker...")
    run_script("benchmarks/scripts/final_self_audit.py")

    print("\n" + "="*60)
    print(f"EXPERIMENT COMPLETE. Results in {res_dir}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

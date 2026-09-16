"""
threshold_sweep.py
Run a parameter sweep for `similarity_threshold` to justify the selection.
Evaluates ONLY on DEV split (user_idx <= 40) to prevent test leakage.
"""
import os
import sys
import yaml
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# We use baseline_c_vector for the sweep since it represents standard vector search.
CONFIG_FILE = os.path.join(BASE_DIR, "configs", "baseline_c_vector.yaml")

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

def main():
    print("==================================================")
    print("THRESHOLD SWEEP (DEV SPLIT ONLY)")
    print("==================================================")

    # Read original config
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    # rag_benchmark.py automatically handles DEV filtering via BENCHMARK_SPLIT=DEV

    results = []
    
    try:
        for t in THRESHOLDS:
            print(f"\nEvaluating threshold: {t}")
            config["retrieval"]["similarity_threshold"] = t
            with open(CONFIG_FILE, "w") as f:
                yaml.dump(config, f)
            
            run_id = f"run_{t}_sweep"
            sweep_res_dir = os.path.join(BASE_DIR, "results", run_id)
            os.makedirs(sweep_res_dir, exist_ok=True)
            
            res = subprocess.run(
                [sys.executable, "benchmarks/scripts/rag_benchmark.py", "--config", "baseline_c_vector"],
                env={**os.environ, "PYTHONPATH": ".", "BENCHMARK_RESULTS_DIR": sweep_res_dir, "BENCHMARK_SPLIT": "DEV"},
                cwd=workspace_root,
                capture_output=True,
                text=True
            )
            
            if res.returncode != 0:
                print("rag_benchmark.py failed!")
                print(res.stderr)
                break
                
            
            res_file = os.path.join(sweep_res_dir, "retrieval_baseline_c_vector.json")
            
            with open(res_file) as f:
                data = json.load(f)
                
            agg = data["aggregate"]
            no_ans = data["no_answer_subtype_breakdown"]
            
            total_no_answer = 0
            total_abstentions_correct = 0
            for type_name, metrics in no_ans.items():
                total_no_answer += metrics["n"]
                total_abstentions_correct += metrics["abstention_correct"]
                
            abstention_acc = total_abstentions_correct / total_no_answer if total_no_answer > 0 else 0
            recall_ans = data["category_breakdown"]["FACTUAL"]["recall_at_5"]  # proxy for answerable recall
            
            results.append({
                "threshold": t,
                "f1_at_5": agg.get("f1_at_5", 0),
                "abstention_acc": abstention_acc,
                "factual_recall": recall_ans
            })
            print(f"  -> F1@5: {agg.get('f1_at_5', 0):.4f} | Abstention Acc: {abstention_acc:.4f} | Factual Recall: {recall_ans:.4f}")
            
    finally:
        # Restore config to 0.40
        config["retrieval"]["similarity_threshold"] = 0.40
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

    print("\n==================================================")
    print("THRESHOLD SWEEP COMPLETE")
    print(f"{'Threshold':<15} | {'Abstention Acc':<15} | {'Factual Recall':<15} | {'F1@5':<15}")
    for r in results:
        print(f"{r['threshold']:<15.2f} | {r['abstention_acc']:<15.4f} | {r['factual_recall']:<15.4f} | {r['f1_at_5']:<15.4f}")
    
    print("\nSelect the threshold that maximizes Abstention Acc without destroying Factual Recall.")

if __name__ == "__main__":
    main()

"""
Run all benchmark configs sequentially (skip dataset generation and seeding
since they are already done). Captures stdout/stderr to run_log.txt.
"""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    "baseline_a",
    "baseline_b",
    "baseline_c_vector",
    "baseline_d_recency_decay",
    "baseline_e_consolidated",
    "baseline_f_temporal_validity",
    "proposed_full",
    "no_temporal",
    "no_graph",
    "no_consolidation",
]

from datetime import datetime
import json
import hashlib

run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
res_dir = os.path.join(BASE_DIR, "benchmarks", "results", run_id)
os.makedirs(res_dir, exist_ok=True)

# Generate manifest
queries_path = os.path.join(BASE_DIR, "benchmarks", "dataset", "synthetic", "queries.json")
try:
    with open(queries_path, "rb") as f:
        ds_hash = hashlib.md5(f.read()).hexdigest()
except:
    ds_hash = "unknown"

manifest = {
    "experiment_id": run_id,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "dataset_hash": ds_hash,
    "models": {
        "embedding": "text-embedding-3-small",
        "chat": "gpt-4o-mini"
    }
}
with open(os.path.join(res_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

log_path = os.path.join(res_dir, "full_run_log.txt")

env_vars = {**os.environ, "PYTHONPATH": ".", "BENCHMARK_RESULTS_DIR": res_dir}

with open(log_path, "w", encoding="utf-8") as logf:
    for cfg in CONFIGS:
        cfg_file = os.path.join(BASE_DIR, "benchmarks", "configs", f"{cfg}.yaml")
        if not os.path.exists(cfg_file):
            msg = f"[SKIP] {cfg}: config file not found\n"
            print(msg, end="")
            logf.write(msg)
            continue

        print(f"\n>> Running: {cfg}")
        logf.write(f"\n{'='*60}\n>> Config: {cfg}\n{'='*60}\n")
        logf.flush()

        result = subprocess.run(
            [sys.executable, "benchmarks/scripts/rag_benchmark.py", "--config", cfg],
            env=env_vars,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Only write non-SQLAlchemy INFO lines to the log for readability
        for line in result.stdout.splitlines():
            if "INFO sqlalchemy" not in line:
                logf.write(line + "\n")
        if result.returncode != 0:
            logf.write(f"[ERROR] exit code {result.returncode}\n")
            logf.write(result.stderr[-3000:])
            print(f"  [FAILED] exit code {result.returncode}")
        else:
            print(f"  [OK]")
        logf.flush()

        logf.write("\n=== Supplementary Benchmarks ===\n")
        
        SUPPLEMENTARY = [
            ["benchmarks/scripts/consolidation_benchmark.py"],
            ["benchmarks/scripts/statistical_analysis.py"],
        ]
        
        for cmd in SUPPLEMENTARY:
            print(f"\n>> Running: {cmd[0]}")
            logf.write(f"\n>> {cmd[0]}\n")
            logf.flush()
            r = subprocess.run(
                [sys.executable] + cmd,
                env=env_vars,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in r.stdout.splitlines():
                if "INFO sqlalchemy" not in line:
                    logf.write(line + "\n")
            if r.returncode != 0:
                logf.write(f"[ERROR] exit code {r.returncode}\n{r.stderr[-2000:]}\n")
                print(f"  [FAILED]")
            else:
                print(f"  [OK]")
            logf.flush()

print(f"\nDone. Full log: {log_path}")

"""
run_multi_seed.py — Phase 9.4

Run the deterministic retrieval benchmark across multiple seeds to measure variance.

For each seed:
  1. Regenerate the synthetic dataset (no LLM calls)
  2. Re-seed the database
  3. Run retrieval benchmarks for Baseline C, D, and Full System
  4. Collect per-query F1@5 scores

Reports:
  - Mean ± std for F1@5, Recall@5, Hit@5 across seeds
  - Per-seed aggregate metrics

Outputs:
  benchmarks/results/multi_seed_summary.json
"""
import os
import sys
import json
import asyncio
import statistics
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

SEEDS = [42, 123, 456, 789, 2026]
CONFIGS = ["baseline_c_vector", "baseline_d_recency_decay", "proposed_full"]
METRICS = ["f1_at_5", "recall_at_5", "hit_rate_at_5", "precision_at_5"]


def run_generate_dataset(seed: int):
    """Regenerate dataset for a given seed (subprocess call to keep memory clean)."""
    import subprocess
    script = os.path.join(BASE_DIR, "scripts", "generate_dataset.py")
    result = subprocess.run(
        [sys.executable, script, "--seed", str(seed), "--users", "50"],
        capture_output=True, text=True, cwd=workspace_root
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate_dataset.py failed for seed {seed}:\n{result.stderr}")
    print(f"  Generated dataset for seed={seed}")


async def run_seed(seed: int) -> dict:
    """Run full benchmark for a single seed. Returns per-config aggregate metrics."""
    print(f"\n{'='*60}")
    print(f"SEED: {seed}")
    print(f"{'='*60}")

    # 1. Regenerate dataset
    run_generate_dataset(seed)

    # 2. Re-seed database and run benchmarks
    from benchmarks.scripts.rag_benchmark import seed_database, run_benchmark

    await seed_database()

    seed_results = {"seed": seed}
    for cfg in CONFIGS:
        cfg_path = os.path.join(BASE_DIR, "configs", f"{cfg}.yaml")
        if not os.path.exists(cfg_path):
            print(f"  Skipping {cfg}: config not found")
            continue
        result = await run_benchmark(cfg)
        if result and "aggregate" in result:
            seed_results[cfg] = {
                m: result["aggregate"].get(m, 0.0) for m in METRICS
            }
        elif result:
            seed_results[cfg] = {
                m: result.get(m, 0.0) for m in METRICS
            }

    return seed_results


async def run_multi_seed_benchmark():
    print("=" * 60)
    print("MULTI-SEED ROBUSTNESS BENCHMARK — Phase 9.4")
    print(f"Seeds: {SEEDS}")
    print(f"Configs: {CONFIGS}")
    print("=" * 60)

    all_seed_results = []
    for seed in SEEDS:
        try:
            result = await run_seed(seed)
            all_seed_results.append(result)
        except Exception as e:
            print(f"  ERROR for seed {seed}: {e}")
            all_seed_results.append({"seed": seed, "error": str(e)})

    # ── Aggregate across seeds ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)

    summary = {}
    for cfg in CONFIGS:
        cfg_vals = {m: [] for m in METRICS}
        for sr in all_seed_results:
            if cfg in sr and "error" not in sr:
                for m in METRICS:
                    cfg_vals[m].append(sr[cfg].get(m, 0.0))

        cfg_summary = {}
        for m in METRICS:
            vals = cfg_vals[m]
            if vals:
                mean = statistics.mean(vals)
                std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
                cfg_summary[m] = {
                    "mean":  round(mean, 4),
                    "std":   round(std, 4),
                    "min":   round(min(vals), 4),
                    "max":   round(max(vals), 4),
                    "n_seeds": len(vals),
                    "values": [round(v, 4) for v in vals],
                }
        summary[cfg] = cfg_summary
        print(f"\n  [{cfg}]")
        for m, ms in cfg_summary.items():
            print(f"    {m}: {ms['mean']:.4f} ± {ms['std']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seeds": SEEDS,
        "configs": CONFIGS,
        "per_seed_results": all_seed_results,
        "aggregate_across_seeds": summary,
    }
    out_path = os.path.join(out_dir, "multi_seed_summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_multi_seed_benchmark())

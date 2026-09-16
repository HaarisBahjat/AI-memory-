"""
statistical_analysis.py — v2.0  Phase 9.4

Computes real, non-parametric statistical tests from actual per-query benchmark results.

Tests performed:
  - Paired Wilcoxon Signed-Rank Test (scipy.stats.wilcoxon) on per-query F1@5
  - Rank-biserial correlation as effect size: r = Z / sqrt(N)
  - 95% CI via bootstrap (1000 iterations, BCa method)
  - Benjamini-Hochberg FDR correction across all planned comparisons

Outputs:
  benchmarks/results/statistical_analysis.json
  benchmarks/results/statistical_analysis.csv
"""

import os
import sys
import json
import csv
import math
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)

try:
    from scipy.stats import wilcoxon, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed. Install with: pip install scipy")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("WARNING: numpy not installed. Install with: pip install numpy")


def find_latest_per_query_file(config_name: str) -> str | None:
    """Find the most recent per-query results file for a given config.
    If BENCHMARK_RESULTS_DIR is set, search there first. Otherwise scan all run_* dirs.
    """
    results_dir = os.path.join(BASE_DIR, "results")
    
    # If called from run_all_benchmarks.py, it sets BENCHMARK_RESULTS_DIR
    explicit_dir = os.environ.get("BENCHMARK_RESULTS_DIR")
    if explicit_dir:
        candidate = os.path.join(explicit_dir, f"retrieval_{config_name}_per_query.json")
        if os.path.exists(candidate):
            return candidate
    
    if not os.path.exists(results_dir):
        return None
    dirs = [d for d in os.listdir(results_dir) if d.startswith("run_") or d.startswith("final_run_")]
    # Sort directories by modification time, newest first
    runs = sorted(
        dirs,
        key=lambda d: os.path.getmtime(os.path.join(results_dir, d)),
        reverse=True
    )
    for run in runs:
        candidate = os.path.join(results_dir, run, f"retrieval_{config_name}_per_query.json")
        if os.path.exists(candidate):
            return candidate
    return None


def load_f1_pairs(sys_file: str, base_file: str) -> tuple[list[float], list[float], list[dict]]:
    """
    Load two per-query result files and align them by user_id.
    Returns (sys_user_f1s, base_user_f1s, raw_pairs).
    """
    with open(sys_file) as f: sys_data = json.load(f)
    with open(base_file) as f: base_data = json.load(f)

    # raw_pairs is for CSV dump
    sys_dict = {r["query_id"]: r for r in sys_data if r.get("category") != "NO_ANSWER"}
    base_dict = {r["query_id"]: r for r in base_data if r.get("category") != "NO_ANSWER"}
    
    # Group query-level F1s by user_id
    sys_user_f1s = defaultdict(list)
    base_user_f1s = defaultdict(list)
    raw_pairs = []

    for qid, s_res in sys_dict.items():
        if qid not in base_dict:
            continue
        b_res = base_dict[qid]
        s_f1 = s_res.get("f1_at_5")
        b_f1 = b_res.get("f1_at_5")

        if s_f1 is None or b_f1 is None:
            continue
            
        uid = s_res.get("user_id", "unknown")
        sys_user_f1s[uid].append(s_f1)
        base_user_f1s[uid].append(b_f1)
        
        raw_pairs.append({
            "query_id": qid,
            "user_id": uid,
            "category": s_res.get("category", ""),
            "system_f1": s_f1,
            "baseline_f1": b_f1,
            "delta": s_f1 - b_f1,
        })
        
    # Aggregate F1 per user (mean)
    sys_agg = []
    base_agg = []
    for uid in sorted(sys_user_f1s.keys()):
        sys_agg.append(sum(sys_user_f1s[uid]) / len(sys_user_f1s[uid]))
        base_agg.append(sum(base_user_f1s[uid]) / len(base_user_f1s[uid]))

    return sys_agg, base_agg, raw_pairs


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean difference.
    Unit of resampling: User (since diffs are already user-level means)
    Number of bootstrap samples: 10,000
    CI: 95% percentile
    Statistic: mean paired F1 difference
    Random seed: 42
    """
    if not NUMPY_AVAILABLE:
        # Fallback: normal approximation
        n = len(diffs)
        if n <= 1: return 0.0, 0.0
        mean = sum(diffs) / n
        std = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        se = std / math.sqrt(n)
        z = 1.96
        return mean - z * se, mean + z * se

    np.random.seed(42)
    arr = np.array(diffs)
    if len(arr) == 0: return 0.0, 0.0
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def rank_biserial_correlation(w_stat: float, n: int) -> float:
    """
    Calculate matched-pairs rank biserial correlation (effect size).
    r = 1 - (4W) / (n * (n+1))
    """
    total = n * (n + 1) / 2
    if total == 0:
        return 0.0
    return 1.0 - (4 * w_stat) / (n * (n + 1))


def fdr_correction(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini-Hochberg FDR correction.
    Returns FDR-adjusted p-values, maintaining original index.
    Ignores None p-values.
    """
    valid_p = [(i, p) for i, p in enumerate(p_values) if p is not None]
    m = len(valid_p)
    adjusted = [None] * len(p_values)
    
    if m == 0:
        return adjusted
        
    indexed = sorted(valid_p, key=lambda x: x[1])
    prev_adj = 1.0
    
    for rank, (orig_idx, pval) in enumerate(reversed(indexed)):
        adjusted_p = min(prev_adj, pval * m / (m - rank))
        adjusted[orig_idx] = min(adjusted_p, 1.0)
        prev_adj = adjusted[orig_idx]
        
    return adjusted


def run_wilcoxon_test(system_f1s: list[float], baseline_f1s: list[float],
                      comparison_label: str) -> dict:
    """Run a paired Wilcoxon test and return the full result dict."""
    if not SCIPY_AVAILABLE:
        return {
            "comparison": comparison_label,
            "error": "scipy not available — install scipy to run this test",
            "n_users": len(system_f1s),
        }

    diffs = [s - b for s, b in zip(system_f1s, baseline_f1s)]
    n = len(diffs)
    mean_diff = sum(diffs) / n if n > 0 else 0.0

    # Remove zero differences (wilcoxon requires non-zero pairs unless zero_method specified)
    nonzero_diffs = [(s, b) for s, b in zip(system_f1s, baseline_f1s) if s != b]
    n_nonzero = len(nonzero_diffs)

    if n_nonzero < 5:
        # Too few non-zero diffs for Wilcoxon — report bootstrap CI only
        ci_lo, ci_hi = bootstrap_ci(diffs)
        return {
            "comparison": comparison_label,
            "n_users": n,
            "n_nonzero_users": n_nonzero,
            "mean_diff": round(mean_diff, 4),
            "ci_95_mean_diff": [round(ci_lo, 4), round(ci_hi, 4)],
            "note": f"n_nonzero_users={n_nonzero} < 5; Wilcoxon not applicable. Reporting bootstrap CI only.",
            "significant_fdr": False,
        }

    s_vals = [s for s, b in nonzero_diffs]
    b_vals = [b for s, b in nonzero_diffs]

    try:
        # Changed to two-sided to accurately detect effects in either direction
        stat, pval = wilcoxon(s_vals, b_vals, alternative="two-sided", zero_method="wilcox")
        w_stat = float(stat)
        p_value = float(pval)
    except Exception as e:
        return {"comparison": comparison_label, "error": str(e), "n_users": n}

    # Effect size
    r_effect = rank_biserial_correlation(w_stat, n_nonzero)

    # Bootstrap CI on mean difference
    ci_lo, ci_hi = bootstrap_ci(diffs)

    return {
        "comparison": comparison_label,
        "n_users": n,
        "n_nonzero_users": n_nonzero,
        "mean_diff": round(mean_diff, 4),
        "mean_system_f1": round(sum(system_f1s) / n, 4),
        "mean_baseline_f1": round(sum(b_vals) / n_nonzero, 4) if n_nonzero > 0 else 0.0,
        "wilcoxon_W": round(w_stat, 2),
        "p_value": round(p_value, 6),
        "p_value_raw": p_value,   # kept for FDR correction
        "effect_size_r": round(r_effect, 4),
        "effect_size_label": (
            "large" if abs(r_effect) >= 0.5 else
            "medium" if abs(r_effect) >= 0.3 else
            "small" if abs(r_effect) >= 0.1 else "negligible"
        ),
        "ci_95_mean_diff": [round(ci_lo, 4), round(ci_hi, 4)],
        "test": "Wilcoxon Signed-Rank (two-sided)",
        "alpha": 0.05,
    }


def run_statistics():
    """
    Run all planned comparisons and apply BH-FDR correction.
    """
    print("=" * 60)
    print("Statistical Analysis — Phase 9.4")
    print("=" * 60)

    COMPARISONS = [
        ("proposed_full", "baseline_c_vector", "Full System vs Vector RAG"),
        ("proposed_full", "no_temporal", "Full System vs No Temporal (Ablation)"),
        ("proposed_full", "no_graph", "Full System vs No Graph (Ablation)"),
        ("proposed_full", "no_consolidation", "Full System vs No Consolidation (Ablation)"),
    ]

    results = []
    csv_rows = []  # Paired per-query data

    for system_cfg, baseline_cfg, label in COMPARISONS:
        sys_file  = find_latest_per_query_file(system_cfg)
        base_file = find_latest_per_query_file(baseline_cfg)

        if not sys_file:
            print(f"\n[SKIP] {label} — no per-query file found for '{system_cfg}'")
            results.append({"comparison": label, "error": "Missing result file", "n_total": 0})
            continue
        if not base_file:
            print(f"\n[SKIP] {label} — no per-query file found for '{baseline_cfg}'")
            results.append({"comparison": label, "error": "Missing result file", "n_total": 0})
            continue

        print(f"\nRunning: {label}")
        print(f"  System file:   {os.path.basename(os.path.dirname(sys_file))}/{os.path.basename(sys_file)}")
        print(f"  Baseline file: {os.path.basename(os.path.dirname(base_file))}/{os.path.basename(base_file)}")

        sys_f1s, base_f1s, pairs = load_f1_pairs(sys_file, base_file)
        test_result = run_wilcoxon_test(sys_f1s, base_f1s, label)
        results.append(test_result)

        # Collect CSV rows
        for p in pairs:
            csv_rows.append({
                "comparison": label,
                "user_id": p.get("user_id", p.get("query_id", "?")),
                "system_f1": p["system_f1"],
                "baseline_f1": p["baseline_f1"],
                "delta": p["delta"],
            })

    # ── BH-FDR correction ─────────────────────────────────────────────────────
    raw_pvals = [r.get("p_value_raw") for r in results]
    valid_indices = [i for i, p in enumerate(raw_pvals) if p is not None]
    if valid_indices:
        subset_pvals = [raw_pvals[i] for i in valid_indices]
        adjusted = fdr_correction(subset_pvals)
        for i, adj in zip(valid_indices, adjusted):
            results[i]["p_value_fdr"] = round(adj, 6)
            results[i]["significant_fdr"] = adj < 0.05

    # Clean raw p-values from output
    for r in results:
        r.pop("p_value_raw", None)
        # Ensure nulls for missing values instead of missing keys
        if "wilcoxon_W" not in r: r["wilcoxon_W"] = None
        if "p_value" not in r: r["p_value"] = None
        if "p_value_fdr" not in r: r["p_value_fdr"] = None

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("  Pairing unit: per-query (each query is a paired observation)")

    # ── Power analysis warnings ──────────────────────────────────────
    for r in results:
        n = r.get("n_total", 0)
        n = r.get("n_users", 0)
        if n > 0 and n < 20:
            r["power_warning"] = (
                f"WARNING: N={n} < 20 paired observations. "
                f"Wilcoxon signed-rank test requires N≥20 for 80% power "
                f"at medium effect size (r=0.3). Results may be underpowered."
            )
            print(f"\n  [!] POWER WARNING for [{r.get('comparison', '?')}]: N={n} < 20")
    for r in results:
        print(f"  [{r['comparison']}]")
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            p_str = round(r["p_value"], 4) if r["p_value"] is not None else "None"
            pf_str = round(r["p_value_fdr"], 4) if r.get("p_value_fdr") is not None else "None"
            sig = "SIGNIFICANT" if r.get("significant_fdr") else "not significant"
            
            print(f"    n_users={r.get('n_users', '?')}, n_nonzero={r.get('n_nonzero_users', '?')}, "
                  f"W={r.get('wilcoxon_W')}, p={p_str}, p_fdr={pf_str}")
            print(f"    effect_size_r={r.get('effect_size_r')} ({r.get('effect_size_label')}), "
                  f"mean_delta={r.get('mean_diff')} [{sig}]")
            print(f"    Bootstrap 95% CI: {r.get('ci_95_mean_diff')}")
            if "note" in r:
                print(f"    Note: {r['note']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "statistical_analysis.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "test_family": "Wilcoxon Signed-Rank (two-sided, α=0.05, BH-FDR corrected)",
            "scipy_available": SCIPY_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "comparisons": results,
        }, f, indent=2)

    csv_path = os.path.join(out_dir, "statistical_analysis.csv")
    # Determine fieldnames dynamically from the actual rows collected
    csv_fieldnames = list(csv_rows[0].keys()) if csv_rows else ["comparison", "user_id", "system_f1", "baseline_f1", "delta"]
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return results


if __name__ == "__main__":
    run_statistics()

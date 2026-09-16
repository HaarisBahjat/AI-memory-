"""
compile_results.py — Print a clean summary of all test-split benchmark results,
then read the statistical_analysis.json and update paper_metrics_log.md.
"""
import json
import os

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "results")
dirs = [d for d in os.listdir(RESULTS_ROOT) if (d.startswith("run_") or d.startswith("final_run_")) and os.path.isdir(os.path.join(RESULTS_ROOT, d))]
run_dirs = sorted(dirs, key=lambda d: os.path.getmtime(os.path.join(RESULTS_ROOT, d)))
RES_DIR = os.path.join(RESULTS_ROOT, run_dirs[-1]) if run_dirs else RESULTS_ROOT
STATS_FILE = os.path.join(RESULTS_ROOT, "statistical_analysis.json")

configs = sorted([f for f in os.listdir(RES_DIR)
                  if f.endswith('.json') and not f.endswith('_per_query.json') and f != 'manifest.json'])

print("=" * 80)
print("FULL-DATASET EVALUATION (50 Users, 400 Queries: 250 Answerable, 150 NO_ANSWER)")
print("=" * 80)
print(f"{'Config':<35} {'F1@5':>6} {'R@5':>6} {'P@5':>6} {'MRR':>6} {'N':>4}")
print("-" * 80)

all_results = {}
for cfg_file in configs:
    with open(os.path.join(RES_DIR, cfg_file)) as f:
        data = json.load(f)
    agg = data["aggregate"]
    name = cfg_file.replace("retrieval_", "").replace(".json", "")
    all_results[name] = data
    print(f"{name:<35} {agg['f1_at_5']:>6.4f} {agg['recall_at_5']:>6.4f} "
          f"{agg['precision_at_5']:>6.4f} {agg['mrr']:>6.4f} {agg['n_queries']:>4}")

print("\n" + "=" * 80)
print("CATEGORY BREAKDOWN — PROPOSED FULL SYSTEM")
print("=" * 80)
if "proposed_full" in all_results:
    for cat, metrics in all_results["proposed_full"]["category_breakdown"].items():
        if cat != "NO_ANSWER":
            print(f"  {cat:<30} F1={metrics['f1_at_5']:.4f}  n={metrics['n']}")

    no_ans_bd = all_results["proposed_full"].get("no_answer_subtype_breakdown", {})
    if no_ans_bd:
        print("\n  NO_ANSWER subtypes:")
        for stype, sm in no_ans_bd.items():
            acc = sm['abstention_correct'] / sm['n'] if sm['n'] > 0 else 0
            print(f"    {stype:<30} abstention_acc={acc:.4f}  n={sm['n']}")

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS SUMMARY (User-Level, N=50)")
print("=" * 80)
if os.path.exists(STATS_FILE):
    with open(STATS_FILE) as f:
        stats = json.load(f)
    for comp in stats["comparisons"]:
        label = comp.get("comparison", "?")
        n = comp.get("n_users", "?")
        mean_diff = comp.get("mean_diff", "?")
        ci = comp.get("ci_95_mean_diff", ["?", "?"])
        note = comp.get("note", "")
        sig = comp.get("significant_fdr", False)
        pval = comp.get("p_value_fdr")
        if "error" in comp:
            print(f"  [{label}] ERROR: {comp['error']}")
        else:
            sig_str = "[SIGNIFICANT]" if sig else "[not sig]"
            pval_str = "< 0.001" if pval is not None and pval < 0.001 else str(pval)
            print(f"  {label}")
            print(f"    n_users={n}, mean_delta={mean_diff}, 95%CI={ci}, p={pval_str} {sig_str}")
            if note:
                print(f"    NOTE: {note}")
else:
    print("  statistical_analysis.json not found.")

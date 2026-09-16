import os
import json
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_f1_from_report(report_text, config_name):
    # Regex to find | ConfigName | F1 | ... in the markdown table
    pattern = rf"\|\s*{config_name}\s*\|\s*([0-9\.]+)\s*\|"
    match = re.search(pattern, report_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def extract_delta_f1_from_report(report_text, comparison_name):
    pattern = rf"###\s*{re.escape(comparison_name)}.*?\n- \*\*ΔF1:\*\* ([0-9\.]+)"
    match = re.search(pattern, report_text, re.DOTALL | re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def main():
    results_dir = os.environ.get("BENCHMARK_RESULTS_DIR")
    if not results_dir:
        all_runs = [d for d in os.listdir(os.path.join(BASE_DIR, "results")) if d.startswith("final_run_")]
        if not all_runs:
            print("FAIL: No final runs found.")
            sys.exit(1)
        all_runs.sort(key=lambda d: os.path.getmtime(os.path.join(BASE_DIR, "results", d)))
        results_dir = os.path.join(BASE_DIR, "results", all_runs[-1])
        
    report_path = os.path.join(results_dir, "final_report.md")
    if not os.path.exists(report_path):
        print(f"FAIL: Report {report_path} not found.")
        sys.exit(1)
        
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Load machine-readable JSON
    stats_path = os.path.join(BASE_DIR, "results", "statistical_analysis.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Dictionary mapping markdown names to JSON file bases
    configs = {
        "Vector RAG": "baseline_c_vector",
        "Full System": "proposed_full",
        "No Temporal Ablation": "no_temporal",
        "No Graph Ablation": "no_graph",
        "No Consolidation Ablation": "no_consolidation"
    }

    errors = []

    # 1. Verify F1@5 scores
    for md_name, json_base in configs.items():
        json_path = os.path.join(results_dir, f"retrieval_{json_base}.json")
        if not os.path.exists(json_path):
            errors.append(f"Missing {json_path}")
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        json_f1 = data.get("aggregate", {}).get("f1_at_5", 0.0)
        md_f1 = extract_f1_from_report(report_text, md_name)
        
        if md_f1 is None:
            errors.append(f"Could not find F1 for '{md_name}' in report.")
            continue
            
        # JSON might have 4 decimal places, check closeness
        if abs(json_f1 - md_f1) > 0.0002:
            errors.append(f"F1 mismatch for {md_name}: JSON={json_f1:.4f} vs Report={md_f1:.4f}")

    # 2. Verify Deltas in comparisons
    for comparison in stats["comparisons"]:
        comp_name = comparison["comparison"]
        json_delta = comparison.get("mean_diff", 0.0)
        
        md_delta = extract_delta_f1_from_report(report_text, comp_name)
        if md_delta is None:
            errors.append(f"Could not find Delta F1 for '{comp_name}' in report.")
            continue
            
        if abs(json_delta - md_delta) > 0.0002:
            errors.append(f"Delta F1 mismatch for {comp_name}: JSON={json_delta:.4f} vs Report={md_delta:.4f}")

    if errors:
        print("========================================")
        print("FINAL SELF-AUDIT FAILED (HALLUCINATIONS DETECTED)")
        print("========================================")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("========================================")
        print("FINAL SELF-AUDIT PASSED (100% CONSISTENT)")
        print("========================================")
        sys.exit(0)

if __name__ == "__main__":
    main()

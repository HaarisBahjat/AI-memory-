import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_report():
    results_dir = os.environ.get("BENCHMARK_RESULTS_DIR")
    if not results_dir:
        # Find latest final_run
        all_runs = [d for d in os.listdir(os.path.join(BASE_DIR, "results")) if d.startswith("final_run_")]
        if not all_runs:
            print("No final runs found.")
            return
        all_runs.sort(key=lambda d: os.path.getmtime(os.path.join(BASE_DIR, "results", d)))
        results_dir = os.path.join(BASE_DIR, "results", all_runs[-1])
    
    print(f"Generating report from {results_dir}...")
    
    # Load manifest
    with open(os.path.join(results_dir, "experiment_manifest.json")) as f:
        manifest = json.load(f)
        
    configs = manifest["configs"]
    
    # Load statistical analysis
    with open(os.path.join(BASE_DIR, "results", "statistical_analysis.json")) as f:
        stats = json.load(f)
        
    # Load all results
    res_data = {}
    for cfg in configs:
        with open(os.path.join(results_dir, f"retrieval_{cfg}.json")) as f:
            res_data[cfg] = json.load(f)
            
    # Markdown generation
    md = []
    md.append(f"# Final Experiment Report")
    md.append(f"**Run ID:** `{manifest['run_id']}`  ")
    md.append(f"**Timestamp:** `{manifest['timestamp']}`  ")
    md.append(f"**DEV Users:** `{manifest['dev_users']}`  ")
    md.append(f"**TEST Users:** `{manifest['test_users']}`  ")
    md.append(f"**Similarity Threshold:** `{manifest.get('selected_threshold', 0.40)}`  ")
    md.append("")
    
    md.append("## Table 1 — Answerable Retrieval (N = 50 queries)")
    md.append("| Config | F1@5 | Recall@5 | Precision@5 | MRR |")
    md.append("|---|---|---|---|---|")
    
    config_labels = {
        "baseline_c_vector": "Vector RAG",
        "proposed_full": "Full System",
        "no_temporal": "No Temporal Ablation",
        "no_graph": "No Graph Ablation",
        "no_consolidation": "No Consolidation Ablation",
    }
    
    for cfg in configs:
        label = config_labels.get(cfg, cfg)
        data = res_data[cfg]["category_breakdown"]
        
        # Aggregate answerable (exclude NO_ANSWER)
        f1_sum = 0
        r_sum = 0
        p_sum = 0
        mrr_sum = 0
        ans_n = 0
        for cat, cdata in data.items():
            if cat != "NO_ANSWER":
                n = cdata["n"]
                f1_sum += cdata["f1_at_5"] * n
                r_sum += cdata["recall_at_5"] * n
                p_sum += cdata["precision_at_5"] * n
                mrr_sum += cdata["mrr"] * n
                ans_n += n
                
        f1 = f1_sum / ans_n if ans_n > 0 else 0
        r = r_sum / ans_n if ans_n > 0 else 0
        p = p_sum / ans_n if ans_n > 0 else 0
        mrr = mrr_sum / ans_n if ans_n > 0 else 0
        
        md.append(f"| {label} | {f1:.4f} | {r:.4f} | {p:.4f} | {mrr:.4f} |")
        
    md.append("")
    
    md.append("## Table 2 — Abstention (N = 30 queries)")
    md.append("| Config | Abstention Accuracy | Total Queries |")
    md.append("|---|---|---|")
    
    for cfg in configs:
        label = config_labels.get(cfg, cfg)
        data = res_data[cfg]["category_breakdown"].get("NO_ANSWER", {"n": 0})
        no_ans_bd = res_data[cfg].get("no_answer_subtype_breakdown", {})
        
        correct = sum(v["abstention_correct"] for v in no_ans_bd.values())
        total = sum(v["n"] for v in no_ans_bd.values())
        
        acc = correct / total if total > 0 else 0
        md.append(f"| {label} | {acc:.4f} | {total} |")
        
    md.append("")
    md.append("> **Note on Abstention:** Although the system improves retrieval of answerable memories, it currently lacks a sufficiently robust answerability/abstention mechanism and therefore performs poorly on unsupported and temporally invalid queries. (Abstention relies entirely on the similarity threshold being low enough, which compromises precision on answerable queries).")
    md.append("")
    
    md.append("## Statistical Significance (N = 10 users)")
    md.append("All statistics are computed at the **user level** using paired Wilcoxon signed-rank test. Using 10 held-out TEST users.")
    md.append("")
    
    for comp in stats.get("comparisons", []):
        name = comp.get("comparison", "Unknown")
        p = comp.get("p_value_fdr")
        sig = comp.get("significant_fdr")
        delta = comp.get("mean_diff")
        n = comp.get("n_users")
        
        sig_str = "**Significant**" if sig else "Not significant"
        p_str = f"p={p:.4f}" if p is not None else "p=None"
        
        md.append(f"### {name}")
        md.append(f"- **ΔF1:** {delta}")
        md.append(f"- **Significance:** {sig_str} ({p_str})")
        md.append(f"- **N:** {n} pairs")
        md.append("")
        
    md.append("## Graph Contribution")
    md.append("> No measurable contribution from graph expansion was observed under the current benchmark on multi-hop queries. The current graph formulation does not provide measurable multi-hop retrieval benefit on this benchmark.")
    md.append("")
    
    md.append("## Consolidation")
    md.append("> The contribution of consolidation could not be independently evaluated because the benchmark does not contain redundant/repeated facts. This config is kept for architectural completeness but currently shows no measurable difference from Vector RAG.")
    
    out_path = os.path.join(results_dir, "final_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Report written to {out_path}")

if __name__ == "__main__":
    generate_report()

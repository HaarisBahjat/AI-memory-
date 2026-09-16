import os, json

results_dir = "benchmarks/results"
runs = sorted(
    [d for d in os.listdir(results_dir)
     if d.startswith("run_") and os.path.isdir(os.path.join(results_dir, d))],
    reverse=True
)

print(f"{'Config':40s} {'F1@5':>7} {'R@5':>7} {'MRR':>7} {'n':>5} {'Outdated':>10} {'GT'}")
print("-" * 100)

for run in runs[:9]:
    run_path = os.path.join(results_dir, run)
    for fname in os.listdir(run_path):
        if fname.endswith(".json") and "per_query" not in fname:
            data = json.load(open(os.path.join(run_path, fname)))
            agg = data.get("aggregate", data)
            cfg = fname.replace("retrieval_", "").replace(".json", "")
            f1      = agg.get("f1_at_5", 0)
            recall  = agg.get("recall_at_5", 0)
            mrr     = agg.get("mrr", 0)
            n       = agg.get("n_queries", 0)
            ofr     = agg.get("outdated_fact_retrieval_rate")
            gt      = agg.get("ground_truth_type", "?")
            ofr_str = f"{ofr:.4f}" if ofr is not None else "N/A"
            print(f"{cfg:40s} {f1:7.4f} {recall:7.4f} {mrr:7.4f} {n:5d} {ofr_str:>10} {gt}")

print()
print("Per-category breakdown for key configs:")
for run in runs[:9]:
    run_path = os.path.join(results_dir, run)
    for fname in os.listdir(run_path):
        if fname.endswith(".json") and "per_query" not in fname:
            data = json.load(open(os.path.join(run_path, fname)))
            if "category_breakdown" in data:
                cfg = fname.replace("retrieval_", "").replace(".json", "")
                bk = data["category_breakdown"]
                print(f"\n  [{cfg}]")
                for cat, metrics in sorted(bk.items()):
                    print(f"    {cat:25s} F1={metrics.get('f1_at_5',0):.4f}  R={metrics.get('recall_at_5',0):.4f}  n={metrics.get('n',0)}")
                na_bk = data.get("no_answer_subtype_breakdown", {})
                if na_bk:
                    print(f"    NO_ANSWER subtypes:")
                    for ntype, ns in na_bk.items():
                        correct = ns.get("abstention_correct", 0)
                        total   = ns.get("n", 0)
                        print(f"      {ntype:35s}  correct={correct}/{total}")

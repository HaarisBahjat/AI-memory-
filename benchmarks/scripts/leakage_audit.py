import os
import json
import numpy as np

def jaccard_similarity(s1, s2):
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset", "synthetic")
    
    with open(os.path.join(dataset_dir, "queries.json")) as f:
        queries = json.load(f)["queries"]
        
    with open(os.path.join(dataset_dir, "memories.json")) as f:
        memories = {m["id"]: m["content"] for m in json.load(f)["memories"]}
        
    with open(os.path.join(dataset_dir, "episodes.json")) as f:
        episodes = {e["id"]: e["content"] for e in json.load(f)["episodes"]}
        
    overlaps = []
    high_overlap_count = 0
    
    for q in queries:
        query_text = q["query"]
        
        # Check against GT memories
        gt_texts = []
        for mid in q.get("ground_truth_memory_ids", []):
            if mid in memories:
                gt_texts.append(memories[mid])
                
        for eid in q.get("expected_episode_ids", []):
            if eid in episodes:
                gt_texts.append(episodes[eid])
                
        if not gt_texts:
            continue
            
        max_overlap = max(jaccard_similarity(query_text, text) for text in gt_texts)
        overlaps.append(max_overlap)
        
        if max_overlap > 0.5:
            high_overlap_count += 1
            
    report = {
        "mean_overlap": 0,
        "median_overlap": 0,
        "percent_high": 0,
        "high_overlap_count": high_overlap_count,
        "total_valid_queries": len(overlaps),
        "status": "PASS"
    }

    if overlaps:
        report["mean_overlap"] = float(np.mean(overlaps))
        report["median_overlap"] = float(np.median(overlaps))
        report["percent_high"] = float((high_overlap_count / len(overlaps)) * 100)
        
        if report["percent_high"] > 0:
            report["status"] = "FAIL"
            
        print("========================================")
        print("DATA LEAKAGE AUDIT (Lexical Overlap)")
        print("========================================")
        print(f"Mean Jaccard Overlap:   {report['mean_overlap']:.4f}")
        print(f"Median Jaccard Overlap: {report['median_overlap']:.4f}")
        print(f"% Queries > 0.5 overlap: {report['percent_high']:.2f}%")
        print("========================================")
    else:
        print("No valid queries found with GT.")
        
    out_dir = os.environ.get("BENCHMARK_RESULTS_DIR", os.path.join(base_dir, "results"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "leakage_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["status"] == "FAIL":
        print(f"Leakage detected. Aborting. Report written to {out_file}")
        sys.exit(1)
    else:
        print(f"Leakage audit passed. Report written to {out_file}")

if __name__ == "__main__":
    import sys
    main()

import json
import os
import hashlib
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "synthetic")

def main():
    print("=" * 50)
    print("EXPERIMENT VALIDATION AUDIT")
    print("=" * 50)

    # 1. Dataset checks
    with open(os.path.join(DATA_DIR, "queries.json")) as f:
        queries = json.load(f)["queries"]
        
    with open(os.path.join(DATA_DIR, "memories.json")) as f:
        memories = json.load(f)["memories"]
        
    users = set(q["user_id"] for q in queries)
    print(f"[DATASET] Total Queries: {len(queries)}")
    print(f"[DATASET] Total Users: {len(users)}")
    
    # Query breakdown
    cats = Counter(q["category"] for q in queries)
    print(f"[DATASET] Categories: {dict(cats)}")
    
    no_answer = sum(1 for q in queries if q["category"] == "NO_ANSWER")
    answerable = len(queries) - no_answer
    print(f"[DATASET] Answerable: {answerable}, NO_ANSWER: {no_answer}")
    
    # User breakdown
    queries_per_user = Counter(q["user_id"] for q in queries)
    assert all(count == 8 for count in queries_per_user.values()), "Not all users have 8 queries!"
    print(f"[DATASET] Verified: All {len(users)} users have exactly 8 queries each.")
    
    # 2. Leakage check
    # Ensure memory UUIDs are unique and tied to exactly one user
    mem_users = {}
    leakage_found = False
    for m in memories:
        if m["id"] in mem_users and mem_users[m["id"]] != m["user_id"]:
            print(f"[LEAKAGE] Memory {m['id']} belongs to multiple users!")
            leakage_found = True
        mem_users[m["id"]] = m["user_id"]
    if not leakage_found:
        print(f"[LEAKAGE] Verified: No memory leakage across users.")
        
    # 3. DEV/TEST Threshold Check
    print(f"\n[AUDIT] Analyzing answer_benchmark.py for Threshold Sweep...")
    answer_bench_path = os.path.join(BASE_DIR, "scripts", "answer_benchmark.py")
    if os.path.exists(answer_bench_path):
        with open(answer_bench_path) as f:
            answer_content = f.read()
            if "gold_queries =" in answer_content:
                print("[DEV/TEST] Found 'gold_queries' filter.")
            if "sample_size" in answer_content:
                print("[DEV/TEST] Uses 'sample_size' parameter.")
    else:
        print("[DEV/TEST] answer_benchmark.py not found.")
        
    # 4. Results check
    results_dir = os.path.join(BASE_DIR, "results")
    if not os.path.exists(results_dir):
        print("[RESULTS] No results dir.")
        return
        
    dirs = [d for d in os.listdir(results_dir) if (d.startswith("run_") or d.startswith("final_run_")) and os.path.isdir(os.path.join(results_dir, d))]
    runs = sorted(dirs, key=lambda d: os.path.getmtime(os.path.join(results_dir, d)))
    if runs:
        latest = runs[-1]
        print(f"\n[RESULTS] Latest run: {latest}")
        sample_file = os.path.join(results_dir, latest, "retrieval_proposed_full_per_query.json")
        if os.path.exists(sample_file):
            with open(sample_file) as f:
                res = json.load(f)
                users_eval = set(r.get("user_id", "") for r in res)
                print(f"[EVALUATED] Users in results: {len(users_eval)}")
                print(f"[EVALUATED] Queries in results: {len(res)}")
                
                # Check MRR for NO_ANSWER
                no_ans = [r for r in res if r.get("category") == "NO_ANSWER"]
                if no_ans:
                    mrrs = [r.get("mrr") for r in no_ans if r.get("mrr") is not None]
                    print(f"[MRR] NO_ANSWER average MRR: {sum(mrrs)/len(mrrs) if mrrs else 0}")
                    print(f"[MRR] Note: NO_ANSWER queries usually have empty ground_truth_ids.")
    
if __name__ == "__main__":
    main()

import os
import json
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "synthetic")

def main():
    report = {
        "status": "PASS",
        "errors": [],
        "counts": {}
    }
    
    def log_error(msg):
        report["status"] = "FAIL"
        report["errors"].append(msg)
        print(f"FAIL: {msg}")

    # Load datasets
    try:
        with open(os.path.join(DATASET_DIR, "episodes.json"), encoding="utf-8") as f:
            episodes = json.load(f)["episodes"]
        with open(os.path.join(DATASET_DIR, "memories.json"), encoding="utf-8") as f:
            memories = json.load(f)["memories"]
        with open(os.path.join(DATASET_DIR, "queries.json"), encoding="utf-8") as f:
            queries = json.load(f)["queries"]
    except Exception as e:
        log_error(f"Failed to load dataset files: {e}")
        sys.exit(1)

    # Validate Counts
    user_ids = set()
    query_categories = {}
    
    episode_ids = set()
    for e in episodes:
        user_ids.add(e["user_id"])
        if e["id"] in episode_ids:
            log_error(f"Duplicate episode ID: {e['id']}")
        episode_ids.add(e["id"])
        
    memory_ids = set()
    memory_user_map = {}
    for m in memories:
        user_ids.add(m["user_id"])
        if m["id"] in memory_ids:
            log_error(f"Duplicate memory ID: {m['id']}")
        memory_ids.add(m["id"])
        memory_user_map[m["id"]] = m["user_id"]

    query_ids = set()
    queries_per_user = {}
    for q in queries:
        user_ids.add(q["user_id"])
        if q["id"] in query_ids:
            log_error(f"Duplicate query ID: {q['id']}")
        query_ids.add(q["id"])
        
        q_user = q["user_id"]
        queries_per_user[q_user] = queries_per_user.get(q_user, 0) + 1
        
        cat = q["category"]
        if cat == "NO_ANSWER":
            cat = q.get("no_answer_type", "NO_ANSWER")
        query_categories[cat] = query_categories.get(cat, 0) + 1
        
        # Check cross-user leakage
        for mid in q.get("ground_truth_memory_ids", []):
            if mid not in memory_user_map:
                log_error(f"Query {q['id']} references non-existent memory {mid}")
            elif memory_user_map[mid] != q_user:
                log_error(f"CROSS-USER LEAKAGE: Query {q['id']} for {q_user} references memory {mid} from {memory_user_map[mid]}")

    report["counts"]["users"] = len(user_ids)
    report["counts"]["episodes"] = len(episodes)
    report["counts"]["memories"] = len(memories)
    report["counts"]["queries"] = len(queries)
    report["counts"]["query_categories"] = query_categories
    
    if len(user_ids) != 50:
        log_error(f"Expected 50 users, found {len(user_ids)}")
    if len(queries) != 400:
        log_error(f"Expected 400 queries, found {len(queries)}")
        
    for user, qcount in queries_per_user.items():
        if qcount != 8:
            log_error(f"User {user} has {qcount} queries instead of 8")

    expected_cats = {
        "FACTUAL": 50,
        "TEMPORAL": 50,
        "HISTORICAL": 50,
        "CONTRADICTION": 50,
        "MULTI_HOP": 50,
        "TYPE_A_UNKNOWN": 50,
        "TYPE_B_SIMILAR_UNSUPPORTED": 50,
        "TYPE_C_TEMPORALLY_INVALID": 50
    }
    
    for cat, expected in expected_cats.items():
        actual = query_categories.get(cat, 0)
        if actual != expected:
            log_error(f"Category {cat} has {actual} queries, expected {expected}")

    # Determine explicit dir
    out_dir = os.environ.get("BENCHMARK_RESULTS_DIR", os.path.join(BASE_DIR, "results"))
    out_file = os.path.join(out_dir, "dataset_validation_report.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["status"] == "FAIL":
        print(f"Dataset integrity validation FAILED. Report written to {out_file}")
        sys.exit(1)
    else:
        print(f"Dataset integrity validation PASSED. Report written to {out_file}")
        sys.exit(0)

if __name__ == "__main__":
    main()

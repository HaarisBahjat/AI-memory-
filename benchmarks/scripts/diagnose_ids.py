"""Diagnose why retrieved IDs never match ground truth IDs."""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "dataset", "synthetic")

memories = json.load(open(os.path.join(data_dir, "memories.json")))["memories"]
queries = json.load(open(os.path.join(data_dir, "queries.json")))["queries"]

# Pick user_041's FACTUAL query
factual_q = [q for q in queries if q["user_id"] == "user_041" and q["category"] == "FACTUAL"][0]
print("=== FACTUAL QUERY (user_041) ===")
print(f"Query: {factual_q['query']}")
print(f"Ground truth memory IDs: {factual_q['ground_truth_memory_ids']}")

gt_id = factual_q["ground_truth_memory_ids"][0]
matching = [m for m in memories if m["id"] == gt_id]
print(f"Memory found in dataset JSON: {len(matching) > 0}")
if matching:
    print(f"  Content: {matching[0]['content']}")
    print(f"  User: {matching[0]['user_id']}")

# All memories for user_041
user_mems = [m for m in memories if m["user_id"] == "user_041"]
print(f"\nUser 041 has {len(user_mems)} memories in dataset:")
for m in user_mems:
    print(f"  {m['id']} | {m['content'][:80]}")

# Now check per-query results to see what was retrieved
results_file = os.path.join(BASE_DIR, "results", "run_20260903_002550", "retrieval_proposed_full_per_query.json")
if os.path.exists(results_file):
    pq = json.load(open(results_file))
    user41_factual = [r for r in pq if r["user_id"] == "user_041" and r["category"] == "FACTUAL"][0]
    print(f"\n=== RETRIEVED ===")
    print(f"Retrieved IDs: {user41_factual['retrieved_ids']}")
    print(f"Ground truth IDs: {user41_factual['ground_truth_ids']}")
    print(f"Candidate memory IDs: {user41_factual['candidate_memory_ids']}")
    print(f"Similarity scores: {user41_factual.get('raw_similarity_scores', {})}")
    
    # Check if ground truth ID is in any retrieved results
    gt_in_retrieved = gt_id in set(user41_factual['retrieved_ids'])
    gt_in_candidates = gt_id in set(user41_factual['candidate_memory_ids'])
    print(f"\nGT in retrieved: {gt_in_retrieved}")
    print(f"GT in candidates: {gt_in_candidates}")
    
    # Check if the IDs are from a different dataset generation
    retrieved_id = user41_factual['retrieved_ids'][0] if user41_factual['retrieved_ids'] else None
    if retrieved_id:
        ret_in_dataset = any(m["id"] == retrieved_id for m in memories)
        ret_in_user = any(m["id"] == retrieved_id for m in user_mems)
        print(f"\nRetrieved ID {retrieved_id}")
        print(f"  Found in dataset memories: {ret_in_dataset}")
        print(f"  Found in user_041 memories: {ret_in_user}")
        if ret_in_dataset:
            ret_mem = [m for m in memories if m["id"] == retrieved_id][0]
            print(f"  Content: {ret_mem['content']}")
            print(f"  User: {ret_mem['user_id']}")

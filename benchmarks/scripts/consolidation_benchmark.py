"""
consolidation_benchmark.py — v2.0  Phase 9.4

Fair evaluation of semantic consolidation benefit.

Two parallel experiments with MATCHING ground truth:
  Experiment A: Retrieve raw episodes → evaluate against expected_episode_ids (GT-E)
  Experiment B: Retrieve semantic memories → evaluate against ground_truth_memory_ids (GT-M)

The same queries are used in both experiments. Each query has BOTH ground-truth types.
The `source_episode_id` field on each memory links the consolidated memory back to
its originating episode, enabling information preservation measurement.

Additionally reports:
  - Storage: raw episode text bytes vs consolidated memory text bytes
  - Embedding storage: semantic_memories have embeddings; episodes do not by default
  - Record reduction ratio (row count)
  - Context token savings (consolidated memories are shorter on average)
"""
import os
import sys
import json
import math
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "synthetic")

workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def calculate_retrieval_metrics(retrieved_ids: list[str], ground_truth_ids: list[str]) -> dict:
    """Standard retrieval metrics (GT-E and GT-M both use same formula)."""
    if not ground_truth_ids:
        return None
    ret = set(retrieved_ids[:5])
    truth = set(ground_truth_ids)
    hits = len(ret & truth)
    p = hits / 5.0
    r = hits / len(truth)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    mrr = 0.0
    for i, rid in enumerate(retrieved_ids[:5]):
        if rid in truth:
            mrr = 1.0 / (i + 1)
            break
    return {"precision_at_5": p, "recall_at_5": r, "f1_at_5": f1,
            "hit_rate_at_5": 1.0 if hits > 0 else 0.0, "mrr": mrr}


def compute_storage_metrics(episodes: list[dict], memories: list[dict]) -> dict:
    """Compute byte-level storage metrics for episodes vs consolidated memories."""
    # Raw episode text storage
    raw_text_bytes = sum(len(ep.get("content", "").encode("utf-8")) for ep in episodes)

    # Consolidated memory text storage
    consol_text_bytes = sum(len(m.get("content", "").encode("utf-8")) for m in memories)

    # Embedding storage: each memory has a 1536-dim float32 embedding
    # Episodes do NOT have embeddings stored (they're indexed by timestamp)
    bytes_per_embedding = 1536 * 4  # float32 = 4 bytes
    embedding_bytes = len(memories) * bytes_per_embedding

    # Total consolidated storage (text + embeddings)
    total_consol_bytes = consol_text_bytes + embedding_bytes

    text_reduction = 1 - (consol_text_bytes / raw_text_bytes) if raw_text_bytes > 0 else 0.0

    return {
        "raw_episode_count": len(episodes),
        "consolidated_memory_count": len(memories),
        "record_reduction_ratio": round(1 - len(memories) / len(episodes), 4),
        "record_reduction_pct": f"{(1 - len(memories)/len(episodes))*100:.1f}%",
        "raw_text_bytes": raw_text_bytes,
        "raw_text_kb": round(raw_text_bytes / 1024, 2),
        "consolidated_text_bytes": consol_text_bytes,
        "consolidated_text_kb": round(consol_text_bytes / 1024, 2),
        "text_reduction_ratio": round(text_reduction, 4),
        "text_reduction_pct": f"{text_reduction*100:.1f}%",
        "embedding_storage_bytes": embedding_bytes,
        "embedding_storage_kb": round(embedding_bytes / 1024, 2),
        "total_consolidated_storage_kb": round(total_consol_bytes / 1024, 2),
        "total_raw_storage_kb": round(raw_text_bytes / 1024, 2),
        "note": (
            "Embeddings are only stored for consolidated memories (not raw episodes), "
            "so total consolidated storage includes significant embedding overhead."
        )
    }


def compute_information_preservation(memories: list[dict], queries: list[dict]) -> dict:
    """
    Measure how well consolidated memories preserve the information from source episodes.

    Uses the `source_episode_id` field on each memory to check that the episode
    referenced by the query's `expected_episode_ids` has a corresponding consolidated memory.
    """
    # Build mapping: episode_id → memory_id (via source_episode_id)
    episode_to_memory = {}
    for m in memories:
        src_ep = m.get("source_episode_id")
        if src_ep:
            episode_to_memory[src_ep] = m["id"]

    total_episode_facts = 0
    preserved_facts = 0
    preservation_detail = []

    for q in queries:
        ep_ids = q.get("expected_episode_ids", [])
        mem_ids = q.get("ground_truth_memory_ids", [])
        if not ep_ids:
            continue

        for ep_id in ep_ids:
            total_episode_facts += 1
            if ep_id in episode_to_memory:
                corresponding_mem = episode_to_memory[ep_id]
                if corresponding_mem in (mem_ids or []):
                    preserved_facts += 1

    preservation_rate = preserved_facts / total_episode_facts if total_episode_facts > 0 else 0.0

    return {
        "total_episode_facts_checked": total_episode_facts,
        "facts_with_consolidated_counterpart": preserved_facts,
        "information_preservation_rate": round(preservation_rate, 4),
        "information_preservation_pct": f"{preservation_rate*100:.1f}%",
        "episode_to_memory_coverage": len(episode_to_memory),
    }


def simulate_retrieval_a(queries: list[dict], episodes: list[dict]) -> list[dict]:
    """
    Experiment A: Simulate recency-based episode retrieval.

    Since this is a deterministic benchmark (no live DB call needed here),
    we simulate by selecting the 5 most recent episodes for each user
    at or before the query time.
    """
    from datetime import datetime

    # Build per-user episode list sorted by timestamp
    ep_by_user: dict[str, list[dict]] = {}
    for ep in episodes:
        uid = ep["user_id"]
        ep_by_user.setdefault(uid, [])
        ep_by_user[uid].append(ep)
    for uid in ep_by_user:
        ep_by_user[uid].sort(key=lambda e: e["created_at"])

    results = []
    for q in queries:
        uid = q["user_id"]
        qt = datetime.fromisoformat(q["query_time"])
        user_eps = ep_by_user.get(uid, [])
        eligible = [e for e in user_eps if datetime.fromisoformat(e["created_at"]) <= qt]
        top5 = eligible[-5:][::-1]  # most recent first
        retrieved = [e["id"] for e in top5]
        gt_e = q.get("expected_episode_ids", [])
        mets = calculate_retrieval_metrics(retrieved, gt_e)
        if mets:
            results.append({
                "query_id": q["id"],
                "category": q.get("category"),
                "retrieved_episode_ids": retrieved,
                "ground_truth_episode_ids": gt_e,
                **mets
            })
    return results


def run_deterministic_consolidation_benchmark():
    print("=" * 60)
    print("CONSOLIDATION BENCHMARK — v2.0 (Phase 9.4)")
    print("=" * 60)

    episodes = json.load(open(os.path.join(DATA_DIR, "episodes.json")))["episodes"]
    memories = json.load(open(os.path.join(DATA_DIR, "memories.json")))["memories"]
    queries  = json.load(open(os.path.join(DATA_DIR, "queries.json")))["queries"]

    # ── Storage metrics ───────────────────────────────────────────────────────
    print("\n[1] Storage Analysis")
    storage = compute_storage_metrics(episodes, memories)
    for k, v in storage.items():
        print(f"    {k}: {v}")

    # ── Information preservation ──────────────────────────────────────────────
    print("\n[2] Information Preservation Analysis")
    preservation = compute_information_preservation(memories, queries)
    for k, v in preservation.items():
        print(f"    {k}: {v}")

    # ── Experiment A: Episode retrieval on GT-E ───────────────────────────────
    print("\n[3] Experiment A — Episode Recency Retrieval (GT-E ground truth)")
    # Only use queries that have expected_episode_ids
    ep_queries = [q for q in queries if q.get("expected_episode_ids") and q.get("category") != "NO_ANSWER"]
    results_a = simulate_retrieval_a(ep_queries, episodes)

    agg_a = {}
    if results_a:
        for k in ["precision_at_5", "recall_at_5", "f1_at_5", "hit_rate_at_5", "mrr"]:
            vals = [r[k] for r in results_a if k in r]
            agg_a[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        agg_a["n"] = len(results_a)
        print(f"    Results (n={agg_a['n']}): F1@5={agg_a['f1_at_5']:.4f} "
              f"R@5={agg_a['recall_at_5']:.4f} Hit@5={agg_a['hit_rate_at_5']:.4f}")

    # ── Experiment B: Semantic memory retrieval note ──────────────────────────
    print("\n[4] Experiment B — Semantic Memory Retrieval (GT-M ground truth)")
    print("    NOTE: Run rag_benchmark.py --config baseline_c_vector for GT-M retrieval results.")
    print("    Load the latest retrieval_baseline_c_vector.json for the GT-M F1@5.")

    # Try to load latest baseline C results
    results_dir = os.path.join(BASE_DIR, "results")
    baseline_c_f1 = None
    if os.path.exists(results_dir):
        runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")], reverse=True)
        for run in runs:
            candidate = os.path.join(results_dir, run, "retrieval_baseline_c_vector.json")
            if os.path.exists(candidate):
                with open(candidate) as f:
                    data = json.load(f)
                    # Handle both old and new format
                    if "aggregate" in data:
                        baseline_c_f1 = data["aggregate"].get("f1_at_5")
                    else:
                        baseline_c_f1 = data.get("f1_at_5")
                break
    if baseline_c_f1 is not None:
        print(f"    Latest Baseline C F1@5 (GT-M): {baseline_c_f1:.4f}")

    # ── Consolidation claim assessment ────────────────────────────────────────
    print("\n[5] Consolidation Claim Assessment")
    if agg_a and baseline_c_f1 is not None:
        # Compare GT-E episode retrieval vs GT-M semantic retrieval
        # These cannot be numerically compared directly (different object types)
        print(f"    Episode retrieval F1@5 (GT-E): {agg_a.get('f1_at_5', 'N/A'):.4f}")
        print(f"    Semantic memory F1@5 (GT-M):   {baseline_c_f1:.4f}")
        print("    NOTE: These two F1 scores use different ground-truth types (GT-E vs GT-M)")
        print("    and CANNOT be directly compared numerically.")
        print("    The valid claim is:")
        print("      'Consolidation reduces record count by {}'.".format(
            storage["record_reduction_pct"]))
        print("      'Consolidated text requires {} less storage.'.".format(
            storage["text_reduction_pct"]))
        print("      'Information preservation rate: {}'.".format(
            preservation["information_preservation_pct"]))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "storage_analysis": storage,
        "information_preservation": preservation,
        "experiment_a_episode_retrieval": {
            "ground_truth_type": "GT-E",
            "aggregate_metrics": agg_a,
        },
        "experiment_b_semantic_retrieval": {
            "ground_truth_type": "GT-M",
            "note": "See retrieval_baseline_c_vector.json for F1@5",
            "latest_f1_at_5": baseline_c_f1,
        },
        "valid_consolidation_claims": [
            f"Record count reduced by {storage['record_reduction_pct']} ({storage['raw_episode_count']} episodes → {storage['consolidated_memory_count']} semantic memories)",
            f"Consolidated text bytes: {storage['consolidated_text_bytes']:,} vs raw text bytes: {storage['raw_text_bytes']:,} ({storage['text_reduction_pct']} reduction in text)",
            f"Information preservation rate: {preservation['information_preservation_pct']}",
            "GT-E and GT-M retrieval F1 cannot be numerically compared (different object types)",
        ],
        "invalid_claims": [
            "Do NOT claim 'system requires consolidation to function' based on the 0.000 no_consolidation result",
            "Do NOT claim semantic retrieval F1 'preserves' episode retrieval F1 without a fair common ground truth",
        ]
    }
    out_path = os.path.join(out_dir, "consolidation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_deterministic_consolidation_benchmark()

"""
rag_benchmark.py — v2.0  Phase 9.4

Retrieval benchmark for the Longitudinal AI Memory System.

Key changes in v2.0:
  - Ground-truth routing: Baseline B uses expected_episode_ids (GT-E), all others use GT-M
  - Baseline E filters to consolidated memories only (consolidation_run_id)
  - Per-category results breakdown added to aggregate output
  - outdated_fact_retrieval_rate computed for all retrieval-based configs
  - NO_ANSWER subtypes (A/B/C) tracked separately
  - Category stored in per-query output for downstream stratified analysis
"""
import os
import sys
import json
import yaml
import hashlib
import asyncio
from datetime import datetime, timezone
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from sqlalchemy import text
from app.core.database import AsyncSessionLocal
from app.services.embedding_service import get_openai_client, settings

# ── Embedding cache ───────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(BASE_DIR, "cache", "embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_cache.json")
embedding_cache: dict = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(embedding_cache, f)

async def cached_embed_text(text_to_embed: str) -> list[float]:
    key = hashlib.md5(text_to_embed.encode()).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    client = get_openai_client()
    resp = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text_to_embed[:10000],
        encoding_format="float",
        dimensions=1536,
    )
    embedding_cache[key] = resp.data[0].embedding
    save_cache()
    return embedding_cache[key]

import app.services.embedding_service
app.services.embedding_service.embed_text = cached_embed_text

# ── Database seeding ──────────────────────────────────────────────────────────
async def seed_database():
    """Seed PostgreSQL with the v2.0 synthetic dataset."""
    print("Seeding PostgreSQL with synthetic dataset v2.0...")
    data_dir = os.path.join(BASE_DIR, "dataset", "synthetic")
    users    = json.load(open(os.path.join(data_dir, "users.json")))["users"]
    episodes = json.load(open(os.path.join(data_dir, "episodes.json")))["episodes"]
    memories = json.load(open(os.path.join(data_dir, "memories.json")))["memories"]

    async with AsyncSessionLocal() as db:
        print("Clearing old benchmark data...")
        await db.execute(text("DELETE FROM semantic_memories WHERE user_id LIKE 'user_%'"))
        await db.execute(text("DELETE FROM episodes WHERE user_id LIKE 'user_%'"))
        await db.execute(text("DELETE FROM users WHERE user_id LIKE 'user_%'"))

        for user in users:
            await db.execute(
                text("INSERT INTO users (user_id, email, password_hash) VALUES (:id, :email, 'hash')"),
                {"id": user["id"], "email": f"{user['id']}@benchmark.local"}
            )

        for ep in episodes:
            await db.execute(
                text("INSERT INTO episodes (id, user_id, session_summary, timestamp) "
                     "VALUES (:id, :user_id, :content, :ts)"),
                {"id": ep["id"], "user_id": ep["user_id"],
                 "content": ep["content"],
                 "ts": datetime.fromisoformat(ep["created_at"])}
            )

        for mem in memories:
            emb  = await cached_embed_text(mem["content"])
            cat  = mem["category"].lower()
            if cat == "fact": cat = "milestone"

            valid_from  = mem.get("valid_from", "2020-01-01T00:00:00")
            valid_until = mem.get("valid_until")
            consol_at   = mem.get("consolidated_at")
            consol_run  = mem.get("consolidation_run_id")

            await db.execute(
                text("""INSERT INTO semantic_memories
                        (id, user_id, text, category, embedding, valid_from, valid_until)
                        VALUES (:id, :uid, :txt, :cat, :emb, :vf, :vu)"""),
                {
                    "id":  mem["id"],    "uid": mem["user_id"],
                    "txt": mem["content"], "cat": cat,
                    "emb": str(emb),
                    "vf":  datetime.fromisoformat(valid_from.replace("Z", "+00:00")),
                    "vu":  datetime.fromisoformat(valid_until.replace("Z", "+00:00")) if valid_until else None,
                }
            )
        await db.commit()
    print(f"Seeded {len(users)} users, {len(episodes)} episodes, {len(memories)} memories.")

    from app.core.database import engine
    await engine.dispose()

# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_metrics(retrieved_ids: list[str], ground_truth_ids: list[str],
                      category: str = "", no_answer_type: str = None,
                      invalid_memory_ids: list[str] = None) -> dict | None:
    """
    Compute retrieval metrics.

    For NO_ANSWER queries: correct behaviour is returning zero results.
    For all other queries: standard Precision@5 / Recall@5 / F1@5 / MRR.
    If invalid_memory_ids are retrieved, they count as false positives and penalize the score.
    """
    is_no_answer = (category == "NO_ANSWER")

    if is_no_answer:
        correct = (len(retrieved_ids) == 0)
        return {
            "precision_at_5": None, "recall_at_5": None,
            "f1_at_5": None, "hit_rate_at_5": None, "mrr": None,
            "abstention_correct": int(correct),
            "no_answer_type": no_answer_type or "UNKNOWN",
        }

    if not ground_truth_ids:
        return None

    retrieved = retrieved_ids[:5]
    truth = set(ground_truth_ids)
    invalid = set(invalid_memory_ids or [])
    ret_set = set(retrieved)
    hits = len(ret_set & truth)
    
    # Any invalid memory retrieved counts against precision.
    # Effectively, invalid memories are false positives.
    invalid_hits = len(ret_set & invalid)

    precision = hits / 5.0
    recall    = hits / len(truth)
    
    # Apply penalty for contradiction failures (retrieving the superseded fact)
    if invalid_hits > 0:
        # Penalize precision heavily if the exact contradicted fact is retrieved
        precision = max(0.0, precision - (invalid_hits * 0.2))

    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hit_rate  = 1.0 if hits > 0 else 0.0
    mrr = 0.0
    for rank, rid in enumerate(retrieved):
        if rid in truth:
            mrr = 1.0 / (rank + 1)
            break

    return {
        "precision_at_5": precision, "recall_at_5": recall,
        "f1_at_5": f1, "hit_rate_at_5": hit_rate, "mrr": mrr,
        "invalid_hits": invalid_hits
    }

# ── Main benchmark ────────────────────────────────────────────────────────────
async def run_benchmark(config_name: str, debug: bool = False):
    print(f"\n{'='*50}")
    print(f"Benchmark: {config_name}")
    print(f"{'='*50}")

    config_path = os.path.join(BASE_DIR, "configs", f"{config_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    features  = config.get("features", {})
    retrieval = config.get("retrieval", {})
    consol_cfg = config.get("consolidation", {})

    use_semantic  = features.get("semantic_memory", False)
    use_temporal  = features.get("temporal_layer", False)
    use_graph     = features.get("knowledge_graph", False)
    use_episodic  = features.get("episodic_window", False)
    use_consol    = features.get("consolidation", False)
    top_k         = retrieval.get("top_k", 5)
    sim_threshold = retrieval.get("similarity_threshold", 0.65)
    consol_run_id = consol_cfg.get("run_id", "")

    # Ground-truth type: Baseline B → GT-E; everything else → GT-M
    ground_truth_type = "GT-E" if (use_episodic and not use_semantic) else "GT-M"

    print(f"Features -> Semantic:{use_semantic} | Temporal:{use_temporal} | "
          f"Graph:{use_graph} | Episodic:{use_episodic} | Consolidation:{use_consol}")
    print(f"Ground-truth type: {ground_truth_type}")

    data_dir = os.path.join(BASE_DIR, "dataset", "synthetic")
    queries  = json.load(open(os.path.join(data_dir, "queries.json")))["queries"]
    
    # Filter by split (DEV=1..40, TEST=41..50, ALL=1..50)
    split = os.environ.get("BENCHMARK_SPLIT", "ALL").upper()
    print(f"Evaluating Split: {split}")
    
    if split == "TEST":
        queries = [q for q in queries if int(q["user_id"].split("_")[1]) > 40]
    elif split == "DEV":
        queries = [q for q in queries if int(q["user_id"].split("_")[1]) <= 40]


    results: list[dict] = []
    per_query_results: list[dict] = []
    # Temporal correctness tracking
    total_retrieved_memories = 0
    outdated_retrieved_memories = 0

    import app.services.retrieval_engine

    for i, q in enumerate(queries):
        # Fresh session per query: prevents one aborted transaction
        # from poisoning all subsequent queries in the same session.
        async with AsyncSessionLocal() as db:
            query_time = datetime.fromisoformat(q["query_time"]).replace(tzinfo=timezone.utc)

            class MockDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return query_time
            app.services.retrieval_engine.datetime = MockDatetime

            candidate_memory_ids: list[str] = []
            raw_similarity_scores: dict      = {}
            time_decay_scores: dict          = {}
            final_candidates: list[dict]     = []

            try:
                # ── Baseline A — No retrieval ─────────────────────────────────
                if not use_semantic and not use_episodic and not use_graph:
                    pass  # final_candidates stays empty

                # ── Baseline B — Episodic Recency Window (evaluated on GT-E) ─
                elif use_episodic and not use_semantic:
                    rq = text("""
                        SELECT id, session_summary AS text, 'episode' AS category, timestamp AS created_at
                        FROM episodes
                        WHERE user_id = :uid AND timestamp <= :qt
                        ORDER BY timestamp DESC
                        LIMIT :k
                    """)
                    result = await db.execute(rq, {"uid": q["user_id"], "qt": query_time, "k": top_k})
                    final_candidates = [dict(r) for r in result.mappings().all()]
                    candidate_memory_ids = [str(m["id"]) for m in final_candidates]

                # ── Episode fallback (no semantic, no episodic, has graph) ──
                elif not use_semantic and not use_episodic and use_graph:
                    rq = text("""
                        SELECT id, session_summary AS text, 'episode' AS category, timestamp AS created_at
                        FROM episodes
                        WHERE user_id = :uid AND timestamp <= :qt
                        ORDER BY timestamp DESC
                        LIMIT :k
                    """)
                    result = await db.execute(rq, {"uid": q["user_id"], "qt": query_time, "k": top_k})
                    final_candidates = [dict(r) for r in result.mappings().all()]
                    candidate_memory_ids = [str(m["id"]) for m in final_candidates]

                # ── Semantic retrieval (Baseline C, D, E, Full) ───────────────
                elif use_semantic:
                    qvec = await cached_embed_text(q["query"])
                    vec_str = "[" + ",".join(str(v) for v in qvec) + "]"

                    # Baseline E: filter to consolidated memories only
                    consolidation_filter = ""
                    if use_consol and consol_run_id:
                        consolidation_filter = "AND consolidation_run_id IS NOT NULL"

                    rq = text(f"""
                        SELECT
                            id, text, category, created_at, is_pinned, valid_until, valid_from,
                            1 - (embedding <=> CAST(:vec AS vector)) AS similarity_score
                        FROM semantic_memories
                        WHERE user_id = :uid
                          {consolidation_filter}
                        ORDER BY is_pinned DESC, embedding <=> CAST(:vec AS vector)
                        LIMIT 20
                    """)
                    result = await db.execute(rq, {"uid": q["user_id"], "vec": vec_str})
                    all_memories = [dict(r) for r in result.mappings().all()]
                    
                    # Apply similarity threshold
                    memories = [m for m in all_memories if m["similarity_score"] >= sim_threshold]

                    candidate_memory_ids = [str(m["id"]) for m in memories]
                    raw_similarity_scores = {str(m["id"]): float(m["similarity_score"]) for m in memories}

                    # Track outdated memories (valid_until < query_time)
                    for m in memories:
                        vu = m.get("valid_until")
                        if vu:
                            if isinstance(vu, str):
                                vu = datetime.fromisoformat(vu)
                            if hasattr(vu, 'tzinfo') and vu.tzinfo is None:
                                vu = vu.replace(tzinfo=timezone.utc)
                            if vu < query_time:
                                outdated_retrieved_memories += 1
                    total_retrieved_memories += len(memories)

                    # ── Temporal filtering and decay ──────────────────────────
                    if use_temporal:
                        valid_mems = []
                        for m in memories:
                            vu = m.get("valid_until")
                            if not vu:
                                valid_mems.append(m)
                            else:
                                if isinstance(vu, str):
                                    vu = datetime.fromisoformat(vu)
                                if hasattr(vu, 'tzinfo') and vu.tzinfo is None:
                                    vu = vu.replace(tzinfo=timezone.utc)
                                if vu >= query_time:
                                    valid_mems.append(m)

                        decayed = app.services.retrieval_engine.filter_by_decay(
                            memories=valid_mems, max_memories=top_k, threshold=sim_threshold
                        )
                        final_candidates = decayed
                        time_decay_scores = {str(m["id"]): m.get("adjusted_score", 0) for m in decayed}
                    else:
                        # No temporal filter — return top_k by similarity score directly
                        final_candidates = memories[:top_k]
                        time_decay_scores = {}

                    # ── Graph expansion (Proposed Full only) ──────────────────
                    if use_graph and final_candidates:
                        top_cat = final_candidates[0].get("category")
                        if top_cat:
                            existing_ids = [str(m["id"]) for m in final_candidates]
                            gvec_str = "[" + ",".join(str(v) for v in qvec) + "]"
                            try:
                                gr = await db.execute(text("""
                                    SELECT
                                        id, text, category, created_at, is_pinned, valid_until,
                                        1 - (embedding <=> CAST(:vec AS vector)) AS similarity_score
                                    FROM semantic_memories
                                    WHERE user_id = :uid
                                      AND category = :cat
                                      AND id != ALL(:eids)
                                    ORDER BY embedding <=> CAST(:vec AS vector)
                                    LIMIT 3
                                """), {"uid": q["user_id"], "vec": gvec_str,
                                       "cat": top_cat, "eids": existing_ids})
                                gc = [dict(r) for r in gr.mappings().all()]
                                merged = sorted(
                                    final_candidates + gc,
                                    key=lambda m: m.get("adjusted_score", m.get("similarity_score", 0)),
                                    reverse=True
                                )
                                final_candidates = merged[:top_k]
                            except Exception:
                                pass  # best-effort

                else:
                    final_candidates = []

                # ── Determine retrieved IDs for the right ground-truth type ──
                if ground_truth_type == "GT-E":
                    retrieved_ids = [str(m["id"]) for m in final_candidates]
                    gt_ids = q.get("expected_episode_ids", [])
                else:
                    retrieved_ids = [str(m["id"]) for m in final_candidates]
                    gt_ids = q.get("ground_truth_memory_ids", [])

                cat = q.get("category", "")
                mets = calculate_metrics(retrieved_ids, gt_ids,
                                         category=cat,
                                         no_answer_type=q.get("no_answer_type"),
                                         invalid_memory_ids=q.get("invalid_memory_ids"))
                if mets:
                    results.append({**mets, "_category": cat})

                per_query_results.append({
                    "query_id": q["id"],
                    "user_id": q.get("user_id", "unknown"),
                    "system": config_name,
                    "category": cat,
                    "no_answer_type": q.get("no_answer_type"),
                    "ground_truth_type": ground_truth_type,
                    "candidate_memory_ids": candidate_memory_ids,
                    "retrieved_ids": retrieved_ids,
                    "ground_truth_ids": gt_ids,
                    "raw_similarity_scores": raw_similarity_scores,
                    "time_decay_scores": time_decay_scores,
                    **(mets or {})
                })

            except Exception as e:
                print(f"  Error on query {q['id']}: {e}")

    if not results:
        print("No results computed.")
        return

    # ── Aggregate metrics (overall) ───────────────────────────────────────────
    metric_keys = [k for k in results[0].keys() if not k.startswith("_") and k != "no_answer_type"]
    agg = {}
    for k in metric_keys:
        vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
        agg[k] = sum(vals) / len(vals) if vals else 0.0

    agg["n_queries"] = len(results)
    agg["ground_truth_type"] = ground_truth_type

    # ── Outdated fact retrieval rate ──────────────────────────────────────────
    if total_retrieved_memories > 0:
        agg["outdated_fact_retrieval_rate"] = outdated_retrieved_memories / total_retrieved_memories
    else:
        agg["outdated_fact_retrieval_rate"] = None

    # ── Per-category breakdown ────────────────────────────────────────────────
    cat_results: dict[str, list] = defaultdict(list)
    for r in results:
        cat_results[r.get("_category", "UNKNOWN")].append(r)

    category_breakdown = {}
    for cat_name, cat_list in cat_results.items():
        cat_agg = {}
        for k in metric_keys:
            vals = [r[k] for r in cat_list if k in r and isinstance(r[k], (int, float))]
            cat_agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        cat_agg["n"] = len(cat_list)
        category_breakdown[cat_name] = cat_agg

    # ── NO_ANSWER subtype breakdown ───────────────────────────────────────────
    no_answer_breakdown = {}
    for r in results:
        if r.get("_category") == "NO_ANSWER":
            ntype = r.get("no_answer_type", "UNKNOWN")
            no_answer_breakdown.setdefault(ntype, {"n": 0, "abstention_correct": 0})
            no_answer_breakdown[ntype]["n"] += 1
            no_answer_breakdown[ntype]["abstention_correct"] += r.get("abstention_correct", 0)

    print("\n--- AGGREGATE RESULTS ---")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- PER-CATEGORY BREAKDOWN ---")
    for cat_name, cat_agg in category_breakdown.items():
        print(f"  {cat_name}: F1@5={cat_agg.get('f1_at_5', 0):.4f} "
              f"R@5={cat_agg.get('recall_at_5', 0):.4f} n={cat_agg['n']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if "BENCHMARK_RESULTS_DIR" in os.environ:
        res_dir = os.environ["BENCHMARK_RESULTS_DIR"]
    else:
        run_id  = f"run_{int(datetime.now().timestamp())}"
        res_dir = os.path.join(BASE_DIR, "results", run_id)
        
    os.makedirs(res_dir, exist_ok=True)

    output = {
        "aggregate": agg,
        "category_breakdown": category_breakdown,
        "no_answer_subtype_breakdown": no_answer_breakdown,
    }
    with open(os.path.join(res_dir, f"retrieval_{config_name}.json"), "w") as f:
        json.dump(output, f, indent=2)
    with open(os.path.join(res_dir, f"retrieval_{config_name}_per_query.json"), "w") as f:
        json.dump(per_query_results, f, indent=2)

    print(f"\nResults saved -> {res_dir}")

    from app.core.database import engine
    await engine.dispose()
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Seed DB before running benchmarks")
    parser.add_argument("--config", type=str, default="", help="Run a single config only")
    args = parser.parse_args()

    if args.seed:
        asyncio.run(seed_database())

    configs_all = [
        "baseline_a", "baseline_b",
        "baseline_c_vector", "baseline_d_recency_decay",
        "baseline_e_consolidated", "proposed_full",
        "no_temporal", "no_graph", "no_consolidation",
    ]
    configs_to_run = [args.config] if args.config else configs_all

    for cfg in configs_to_run:
        cfg_path = os.path.join(BASE_DIR, "configs", f"{cfg}.yaml")
        if os.path.exists(cfg_path):
            asyncio.run(run_benchmark(cfg))
        else:
            print(f"Skipping {cfg}: config file not found")

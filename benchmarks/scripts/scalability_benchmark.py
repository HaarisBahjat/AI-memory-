"""
scalability_benchmark.py — v2.0  Phase 9.4

Real latency measurements using actual PostgreSQL + pgvector queries.

Scales tested: 100, 1000, 5000, 10000 (local hardware limit)
Methodology:
  - Insert synthetic 1536-dim embedding vectors into semantic_memories
  - Run 20 warm-up queries, then 100 timed queries per scale
  - Measure ONLY retrieval latency (embedding + vector search + temporal filter)
  - End-to-end latency (including LLM generation) is NOT measured here
  - Report exact P50/P95/P99 from measured timings (no extrapolation for measured scales)
  - Curve fitting for extrapolation to 100k is clearly labeled as PROJECTION

Honest claims:
  - Only measured values are reported as benchmark results
  - Extrapolated values are labeled "EXTRAPOLATED (log-linear fit)" in the output
  - The hypothesis is revised to: "Retrieval latency remains below X ms at P95 up to 10k scale"

Outputs:
  benchmarks/results/latency_vs_memory_size.csv
  benchmarks/results/scalability_analysis.json
"""
import os
import sys
import json
import time
import math
import asyncio
import statistics
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from sqlalchemy import text
from app.core.database import AsyncSessionLocal
from app.services.embedding_service import get_openai_client, settings

# ── Test configuration ────────────────────────────────────────────────────────
SCALES       = [100, 1000, 5000, 10000]   # Real measurements (local hardware cap)
WARMUP_RUNS  = 20
TIMED_RUNS   = 100
TEST_USER_ID = "scalability_test_user"
DECAY_LAMBDA = 0.005


async def get_test_embedding() -> list[float]:
    """Get a representative query embedding (cached)."""
    cache_path = os.path.join(BASE_DIR, "cache", "scalability_query_emb.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    client = get_openai_client()
    resp = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input="What helps the user manage their health condition?",
        encoding_format="float", dimensions=1536
    )
    emb = resp.data[0].embedding
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(emb, f)
    return emb


def synthetic_embedding(seed: int) -> list[float]:
    """Generate a deterministic fake embedding without an API call."""
    import hashlib
    h = hashlib.md5(str(seed).encode()).digest()
    # Build a 1536-dim normalised vector from repeated hash
    raw = []
    s = str(seed)
    while len(raw) < 1536:
        h = hashlib.md5(s.encode()).digest()
        raw.extend([(b - 128) / 128.0 for b in h])
        s += "x"
    v = raw[:1536]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


async def setup_scale(db, n: int):
    """Insert n synthetic memories for the test user."""
    # Clear previous test data
    await db.execute(text(
        "DELETE FROM semantic_memories WHERE user_id = :uid"
    ), {"uid": TEST_USER_ID})
    await db.execute(text(
        "DELETE FROM users WHERE user_id = :uid"
    ), {"uid": TEST_USER_ID})
    await db.execute(text(
        "INSERT INTO users (user_id, email, password_hash) VALUES (:uid, :email, 'hash')"
    ), {"uid": TEST_USER_ID, "email": "scalability@benchmark.local"})

    print(f"  Inserting {n} synthetic memories...", end="", flush=True)
    batch_size = 200
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for j in range(start, end):
            emb = synthetic_embedding(j)
            vec_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
            await db.execute(text("""
                INSERT INTO semantic_memories
                    (id, user_id, text, category, embedding, valid_from)
                VALUES (gen_random_uuid(), :uid, :txt, 'milestone', CAST(:emb AS vector), NOW())
            """), {"uid": TEST_USER_ID, "txt": f"Synthetic memory {j}", "emb": vec_str})
        await db.commit()
        print(".", end="", flush=True)
    print(f" done ({n} rows)")


async def run_retrieval_query(db, query_vec: list[float]) -> float:
    """
    Run one retrieval query and return elapsed time in milliseconds.
    Measures: embedding already done → vector search + temporal filter.
    """
    vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"

    t0 = time.perf_counter()
    await db.execute(text("""
        SELECT id, text,
            1 - (embedding <-> CAST(:vec AS vector)) AS similarity_score
        FROM semantic_memories
        WHERE user_id = :uid
          AND (valid_until IS NULL OR valid_until > NOW())
        ORDER BY embedding <-> CAST(:vec AS vector)
        LIMIT 5
    """), {"vec": vec_str, "uid": TEST_USER_ID})
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0   # convert to ms


async def get_table_size_kb(db) -> float:
    """Get the actual PostgreSQL table size in KB."""
    result = await db.execute(text("""
        SELECT pg_total_relation_size('semantic_memories') / 1024.0 AS size_kb
    """))
    row = result.fetchone()
    return float(row[0]) if row else 0.0


async def benchmark_scale(n: int, query_vec: list[float]) -> dict:
    """Benchmark one scale level: insert n rows, run warm-up + timed queries."""
    print(f"\n{'='*50}")
    print(f"Scale: {n:,} memories")
    print(f"{'='*50}")

    async with AsyncSessionLocal() as db:
        await setup_scale(db, n)

        # Get table size after insertion
        size_kb = await get_table_size_kb(db)

        # Warm-up
        print(f"  Warm-up: {WARMUP_RUNS} queries...")
        for _ in range(WARMUP_RUNS):
            await run_retrieval_query(db, query_vec)

        # Timed runs
        print(f"  Timed:  {TIMED_RUNS} queries...", end="", flush=True)
        timings = []
        for _ in range(TIMED_RUNS):
            ms = await run_retrieval_query(db, query_vec)
            timings.append(ms)
        print(" done")

    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[int(0.95 * len(timings))]
    p99 = timings[int(0.99 * len(timings))]
    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings)

    result = {
        "scale": n,
        "n_timed_queries": TIMED_RUNS,
        "n_warmup_queries": WARMUP_RUNS,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(mean, 3),
        "stdev_ms": round(stdev, 3),
        "min_ms": round(timings[0], 3),
        "max_ms": round(timings[-1], 3),
        "table_size_kb": round(size_kb, 2),
        "measured": True,
        "latency_type": "RETRIEVAL_ONLY (vector search + temporal filter; excludes embedding generation and LLM generation)",
    }

    print(f"  P50={p50:.2f}ms  P95={p95:.2f}ms  P99={p99:.2f}ms  Table={size_kb:.0f}KB")
    return result


def log_linear_extrapolate(scales: list[int], p95_vals: list[float],
                            target: int) -> float | None:
    """
    Fit a log-linear model P95 = a * log(n) + b and extrapolate to target.
    Returns None if insufficient data.
    """
    if len(scales) < 2:
        return None
    try:
        log_scales = [math.log(s) for s in scales]
        n = len(log_scales)
        mean_x = sum(log_scales) / n
        mean_y = sum(p95_vals) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_scales, p95_vals))
        denom = sum((x - mean_x) ** 2 for x in log_scales)
        a = num / denom if denom != 0 else 0
        b = mean_y - a * mean_x
        return a * math.log(target) + b
    except Exception:
        return None


async def run_scalability_benchmark():
    print("=" * 60)
    print("SCALABILITY BENCHMARK — v2.0 (Phase 9.4)")
    print("=" * 60)
    print(f"Scales: {SCALES}")
    print(f"Timed runs per scale: {TIMED_RUNS}")
    print("Latency type: RETRIEVAL ONLY (not end-to-end)")
    print()

    query_vec = await get_test_embedding()

    all_results = []
    for n in SCALES:
        result = await benchmark_scale(n, query_vec)
        all_results.append(result)

    # Clean up test data
    async with AsyncSessionLocal() as db:
        await db.execute(text("DELETE FROM semantic_memories WHERE user_id = :uid"),
                         {"uid": TEST_USER_ID})
        await db.execute(text("DELETE FROM users WHERE user_id = :uid"),
                         {"uid": TEST_USER_ID})
        await db.commit()
    print("\nCleaned up test data.")

    # ── Log-linear extrapolation to 100k (clearly labeled) ───────────────────
    measured_scales = [r["scale"] for r in all_results]
    measured_p95 = [r["p95_ms"] for r in all_results]
    extrapolation_targets = [50000, 100000]
    extrapolated = []
    for target in extrapolation_targets:
        est_p95 = log_linear_extrapolate(measured_scales, measured_p95, target)
        if est_p95:
            extrapolated.append({
                "scale": target,
                "p95_ms_extrapolated": round(max(est_p95, 0), 3),
                "measured": False,
                "method": "log-linear fit on measured scales",
                "warning": "EXTRAPOLATED — not a measured value. Do not cite as benchmark result.",
            })
            print(f"  Extrapolated P95 at {target:,}: {est_p95:.2f}ms (log-linear projection)")

    # ── Hypothesis assessment ──────────────────────────────────────────────────
    max_p95 = max(r["p95_ms"] for r in all_results)
    max_scale = max(r["scale"] for r in all_results)
    hypothesis = (
        f"Retrieval subsystem P95 latency remains below {math.ceil(max_p95 + 5)}ms "
        f"at up to {max_scale:,} semantic memories on the test hardware."
    )

    print(f"\n  Revised hypothesis: {hypothesis}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "methodology": {
            "latency_type": "retrieval_only",
            "includes": ["vector search", "temporal filter (valid_until check)"],
            "excludes": ["query embedding generation", "LLM generation", "network latency"],
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "scales_measured": SCALES,
        },
        "measured_results": all_results,
        "extrapolated_results": extrapolated,
        "hypothesis": hypothesis,
        "end_to_end_note": (
            "End-to-end latency (retrieval + LLM generation) is NOT measured in this benchmark. "
            "LLM generation adds 500-2000ms depending on model and output length. "
            "Sub-100ms claims apply to the retrieval subsystem only."
        ),
    }

    json_path = os.path.join(out_dir, "scalability_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = os.path.join(out_dir, "latency_vs_memory_size.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scale", "measured", "p50_ms", "p95_ms", "p99_ms",
            "mean_ms", "stdev_ms", "table_size_kb"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        for r in extrapolated:
            writer.writerow({
                "scale": r["scale"], "measured": False,
                "p50_ms": "", "p95_ms": r["p95_ms_extrapolated"],
                "p99_ms": "", "mean_ms": "", "stdev_ms": "", "table_size_kb": "",
            })

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    asyncio.run(run_scalability_benchmark())

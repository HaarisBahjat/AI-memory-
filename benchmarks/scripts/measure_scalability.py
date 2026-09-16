import os
import sys
import asyncio
import time
import numpy as np
from sqlalchemy import text
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from app.core.database import AsyncSessionLocal

SCALES = [100, 1000, 5000, 10000]
WARMUP_QUERIES = 5
TIMING_QUERIES = 20

async def clear_dummy_memories():
    async with AsyncSessionLocal() as db:
        await db.execute(text("DELETE FROM semantic_memories WHERE user_id = 'dummy_scale'"))
        await db.execute(text("DELETE FROM users WHERE user_id = 'dummy_scale'"))
        await db.commit()

async def insert_dummy_memories(count: int):
    print(f"Inserting {count} dummy memories...")
    async with AsyncSessionLocal() as db:
        await db.execute(text("INSERT INTO users (user_id, email, password_hash) VALUES ('dummy_scale', 'dummy@example.com', 'dummy') ON CONFLICT DO NOTHING"))
        await db.commit()
        # Insert in batches
        batch_size = 1000
        for i in range(0, count, batch_size):
            chunk = min(batch_size, count - i)
            values = []
            for j in range(chunk):
                vec = np.random.randn(1536)
                vec = vec / np.linalg.norm(vec)
                import uuid
                values.append({
                    "id": str(uuid.uuid4()),
                    "uid": "dummy_scale",
                    "txt": f"Dummy fact {i+j}",
                    "cat": "factual",
                    "emb": f"[{','.join(map(str, vec))}]"
                })
            
            await db.execute(
                text("""
                    INSERT INTO semantic_memories (id, user_id, text, category, embedding)
                    VALUES (:id, :uid, :txt, :cat, CAST(:emb AS vector))
                """),
                values
            )
        await db.commit()

async def run_queries(num_queries: int) -> list[float]:
    latencies = []
    async with AsyncSessionLocal() as db:
        for _ in range(num_queries):
            vec = np.random.randn(1536)
            vec = vec / np.linalg.norm(vec)
            vec_str = f"[{','.join(map(str, vec))}]"
            
            t0 = time.perf_counter()
            await db.execute(
                text("""
                    SELECT id FROM semantic_memories
                    WHERE user_id = 'dummy_scale'
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT 5
                """),
                {"vec": vec_str}
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
    return latencies

async def main():
    print("========================================")
    print("SCALABILITY LATENCY BENCHMARK (pgvector)")
    print("========================================")
    
    results = {}
    
    for scale in SCALES:
        await clear_dummy_memories()
        await insert_dummy_memories(scale)
        
        # Warmup
        await run_queries(WARMUP_QUERIES)
        
        # Timing
        lats = await run_queries(TIMING_QUERIES)
        
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        p99 = np.percentile(lats, 99)
        
        results[scale] = {"p50": p50, "p95": p95, "p99": p99}
        print(f"Scale {scale:5d} -> p50: {p50:5.1f}ms, p95: {p95:5.1f}ms, p99: {p99:5.1f}ms")
        
    await clear_dummy_memories()
    print("\nScalability measurements complete.")
    
if __name__ == "__main__":
    asyncio.run(main())

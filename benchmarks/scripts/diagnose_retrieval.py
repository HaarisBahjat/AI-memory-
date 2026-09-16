"""
Quick diagnostic: run a single semantic retrieval query manually and see what comes back.
Also tests the embedding endpoint is working.
"""
import asyncio, json, os, hashlib, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(BASE_DIR))

from sqlalchemy import text
from app.core.database import AsyncSessionLocal
from app.services.embedding_service import get_openai_client, settings

CACHE_FILE = os.path.join(BASE_DIR, "cache", "embeddings", "embeddings_cache.json")
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

async def cached_embed(t):
    k = hashlib.md5(t.encode()).hexdigest()
    if k in cache:
        return cache[k]
    client = get_openai_client()
    resp = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL, input=t[:10000],
        encoding_format="float", dimensions=1536
    )
    cache[k] = resp.data[0].embedding
    json.dump(cache, open(CACHE_FILE, "w"))
    return cache[k]

async def run():
    queries = json.load(open(os.path.join(BASE_DIR, "dataset", "synthetic", "queries.json")))["queries"]
    q = next(x for x in queries if x["category"] == "FACTUAL")
    print(f"Query: {q['query']}")
    print(f"User: {q['user_id']}")
    print(f"GT IDs: {q['ground_truth_memory_ids']}")

    vec = await cached_embed(q["query"])
    print(f"Embedding dims: {len(vec)}")

    vec_str = "[" + ",".join(str(v) for v in vec) + "]"

    async with AsyncSessionLocal() as db:
        # Check user memories exist
        r = await db.execute(text("SELECT id, text FROM semantic_memories WHERE user_id=:uid"), {"uid": q["user_id"]})
        mems = r.fetchall()
        print(f"\nAll memories for {q['user_id']}:")
        for m in mems:
            print(f"  {m[0]} | {m[1][:50]}")

        # Test vector search
        r2 = await db.execute(text("""
            SELECT id, text,
                1 - (embedding <-> CAST(:vec AS vector)) AS score
            FROM semantic_memories
            WHERE user_id = :uid
            ORDER BY embedding <-> CAST(:vec AS vector)
            LIMIT 5
        """), {"uid": q["user_id"], "vec": vec_str})
        results = r2.fetchall()
        print(f"\nVector search results:")
        for row in results:
            print(f"  {row[0]} | score={row[2]:.4f} | {str(row[1])[:50]}")

        print(f"\nGT IDs: {q['ground_truth_memory_ids']}")
        top5_ids = [str(r[0]) for r in results]
        print(f"Top5 IDs: {top5_ids}")
        hits = set(top5_ids) & set(q['ground_truth_memory_ids'])
        print(f"Hits: {hits}")
        print(f"F1@5 would be: {2*(len(hits)/5)*(len(hits)/len(q['ground_truth_memory_ids']))/(len(hits)/5+len(hits)/len(q['ground_truth_memory_ids'])) if hits else 0:.4f}")

asyncio.run(run())

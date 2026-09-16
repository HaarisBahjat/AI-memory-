import asyncio
from sqlalchemy import text
from app.core.database import AsyncSessionLocal
import json

async def check():
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("SELECT COUNT(*) FROM semantic_memories WHERE user_id LIKE 'user_%'"))
        mem_count = r.scalar()
        print("Memories in DB:", mem_count)

        r2 = await db.execute(text("SELECT COUNT(*) FROM episodes WHERE user_id LIKE 'user_%'"))
        ep_count = r2.scalar()
        print("Episodes in DB:", ep_count)

        r3 = await db.execute(text("SELECT id, text, user_id FROM semantic_memories WHERE user_id='user_001' LIMIT 3"))
        rows = r3.fetchall()
        for row in rows:
            print("  Memory ID:", row[0], "|", str(row[1])[:60])

        # Check if GT memory IDs are in DB
        import os
        qfile = os.path.join("benchmarks", "dataset", "synthetic", "queries.json")
        queries = json.load(open(qfile))["queries"]
        factual = [q for q in queries if q.get("category") == "FACTUAL"][:3]
        print("\nChecking GT IDs from 3 FACTUAL queries:")
        for q in factual:
            gt_ids = q.get("ground_truth_memory_ids", [])
            print(f"  Query user: {q['user_id']}, GT IDs: {gt_ids}")
            for gid in gt_ids:
                r4 = await db.execute(text("SELECT id FROM semantic_memories WHERE id=:gid"), {"gid": gid})
                found = r4.fetchone()
                print(f"    {gid} -> {'FOUND' if found else 'MISSING'}")

        # Check columns on semantic_memories
        r5 = await db.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='semantic_memories' ORDER BY ordinal_position"
        ))
        cols = [row[0] for row in r5.fetchall()]
        print("\nsemantic_memories columns:", cols)

asyncio.run(check())

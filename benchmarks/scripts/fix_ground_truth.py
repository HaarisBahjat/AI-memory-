import os
import sys
import json
import asyncio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from sqlalchemy import text
from app.core.database import AsyncSessionLocal

async def main():
    print("Fixing ground truth IDs for the new DB...")
    
    data_dir = os.path.join(BASE_DIR, "dataset", "synthetic")
    with open(os.path.join(data_dir, "memories.json")) as f:
        memories = json.load(f)["memories"]
        
    with open(os.path.join(data_dir, "queries.json")) as f:
        queries = json.load(f)["queries"]

    # Load all DB memories
    db_memories = []
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT id, user_id, text FROM semantic_memories"))
        db_memories = [dict(r) for r in res.mappings().all()]

    # Map old memory ID to new memory ID by exact text match
    # Deduplication might have slightly altered texts or merged them.
    # But since we monkeypatched with EXACT text, they should match, 
    # except for those merged (which would match the surviving one's text).
    # Wait, if A and B are similar, A is inserted, B is merged into A.
    # The text of A remains.
    
    old_to_new = {}
    for old_m in memories:
        user_id = old_m["user_id"]
        content = old_m["content"]
        
        # Find DB memory for this user with exact text
        match = next((db_m for db_m in db_memories if db_m["user_id"] == user_id and db_m["text"] == content), None)
        if match:
            old_to_new[old_m["id"]] = str(match["id"])
        else:
            # If not exact match, it was deduplicated into an existing one!
            # We must find the closest vector, but since we don't have embeddings loaded here...
            # We can use the DB similarity search!
            async with AsyncSessionLocal() as db:
                from app.services.embedding_service import embed_text
                # Note: this calls OpenAI if not cached, but should be cached.
                vec = await embed_text(content)
                res = await db.execute(
                    text("SELECT id FROM semantic_memories WHERE user_id = :uid ORDER BY embedding <-> CAST(:vec AS vector) LIMIT 1"),
                    {"uid": user_id, "vec": str(vec)}
                )
                row = res.mappings().first()
                if row:
                    old_to_new[old_m["id"]] = str(row["id"])

    # Now rewrite queries
    for q in queries:
        new_gts = []
        for old_id in q.get("ground_truth_memory_ids", []):
            if old_id in old_to_new:
                new_gts.append(old_to_new[old_id])
        q["ground_truth_memory_ids"] = new_gts
        
        # Also rewrite invalid_memory_ids
        new_inv = []
        for old_id in q.get("invalid_memory_ids", []):
            if old_id in old_to_new:
                new_inv.append(old_to_new[old_id])
        q["invalid_memory_ids"] = new_inv

    with open(os.path.join(data_dir, "queries.json"), "w") as f:
        json.dump({"queries": queries}, f, indent=2)

    print("Queries updated successfully.")

if __name__ == "__main__":
    asyncio.run(main())

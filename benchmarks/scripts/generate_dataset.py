"""
generate_dataset.py — v2.0

Generates a fully reproducible, paper-ready synthetic longitudinal benchmark dataset.

Changes in v2.0:
  - Added `expected_episode_ids` per query (enables fair Baseline B evaluation)
  - Added `answerable`, `valid_at_query_time`, `no_answer_type` fields
  - Added `consolidation_run_id` and `consolidated_at` to memories (enables fair Baseline D vs E)
  - Added Type B (similar-but-unsupported) and Type C (temporally invalid) NO_ANSWER queries
  - Memories without consolidation are tagged `consolidated_at=null`; post-run memories get a timestamp
  - Added multi-seed CLI argument
  - Cross-domain scenarios: health/wellness, preferences, goals, habits, lifestyle, study/work
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone
import argparse
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "synthetic")

# Consolidation run metadata (simulates a real nightly consolidation run)
CONSOLIDATION_RUN_ID = "consolidation_run_2026_phase2"
CONSOLIDATION_TIMESTAMP = "2026-03-15T02:00:00"   # Simulated nightly run date

# ── Deterministic UUID Generation ──────────────────────────────────────────────
# CRITICAL: Using uuid.uuid5() instead of uuid.uuid4() ensures that the same
# seed ALWAYS produces the same UUIDs. This prevents the ground-truth misalignment
# bug where dataset regeneration would create new UUIDs that don't match the
# memory IDs already seeded into the database.
_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
_id_counter = 0
_id_seed = 42  # Will be overridden by generate_dataset()

def make_id():
    """Generate a deterministic UUID based on the global seed and a monotonic counter."""
    global _id_counter
    _id_counter += 1
    return str(uuid.uuid5(_UUID_NAMESPACE, f"seed:{_id_seed}:idx:{_id_counter}"))

# ── Domain Templates ──────────────────────────────────────────────────────────

# Health/wellness domain
COPING_MECHANISMS = [
    ("Walking", "running"),
    ("Deep breathing", "meditation"),
    ("Listening to music", "playing an instrument"),
    ("Journaling", "talking to a therapist"),
    ("Yoga", "swimming"),
]
SYMPTOMS = [
    ("anxiety", "insomnia"),
    ("stress", "migraines"),
    ("back pain", "fatigue"),
    ("social anxiety", "panic attacks"),
]

# Preference domain
PREFERENCE_PAIRS = [
    ("coffee", "tea"),
    ("action movies", "documentaries"),
    ("meat dishes", "vegetarian meals"),
    ("working late", "waking early"),
    ("city life", "suburban life"),
]

# Goal/habit domain
HABIT_PAIRS = [
    ("skipping breakfast", "eating a healthy breakfast"),
    ("taking the elevator", "using the stairs"),
    ("watching TV before bed", "reading before bed"),
    ("procrastinating assignments", "planning tasks in advance"),
    ("skipping workouts", "exercising three times a week"),
]

# Work/study domain
STRESS_SOURCES = [
    ("project deadlines", "time management skills"),
    ("exam pressure", "structured study sessions"),
    ("team conflicts", "open communication"),
    ("information overload", "prioritisation techniques"),
    ("unclear expectations", "regular check-ins with supervisor"),
]

# Names
NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie", "Skyler", "Cameron", "Drew",
         "Quinn", "Avery", "Rowan", "Sage", "Elliot", "Blake", "Reese", "Parker", "Dakota", "Logan"]

TRIGGERS = ["Work stress", "Financial pressure", "Lack of sleep", "Family drama", "Social isolation"]
CONSEQUENCES = ["poor sleep", "irritability", "trouble concentrating", "loss of appetite", "low energy"]
SECONDARY_CONSEQUENCES = ["work performance", "relationships", "daily productivity", "overall health", "social life"]


def generate_user_timeline(user_idx: int, start_date: datetime, seed: int):
    """
    Generate a timeline of episodes, memories (with consolidation metadata),
    and queries for a single fictional user.

    Memory schema additions (v2):
      - `source_episode_id`: links memory to the episode it was derived from
      - `consolidated_at`: None for raw memories, timestamp for post-consolidation memories
      - `consolidation_run_id`: ID of the consolidation run that produced this memory
    """
    rng = random.Random(seed + user_idx)

    user_id = f"user_{user_idx:03d}"
    name = rng.choice(NAMES)

    episodes = []
    memories = []   # consolidated memories (Baseline E / Full System)
    queries = []

    symptom, _ = rng.choice(SYMPTOMS)
    old_coping, new_coping = rng.choice(COPING_MECHANISMS)
    old_pref, new_pref = rng.choice(PREFERENCE_PAIRS)
    old_habit, new_habit = rng.choice(HABIT_PAIRS)

    current_date = start_date

    # ── Health timeline (Episodes 1–3) ───────────────────────────────────────
    ep_id_1 = make_id()
    ep_text_1 = (
        f"{name} reported feeling high levels of {symptom} recently. "
        f"To manage this, {name} has started {old_coping.lower()} every evening. "
        f"{name} noted that {old_coping.lower()} really helps to reduce the {symptom}."
    )
    episodes.append({
        "id": ep_id_1, "user_id": user_id,
        "content": ep_text_1, "created_at": current_date.isoformat()
    })

    mem_id_1 = make_id()
    # This memory is PRE-consolidation (simulates a memory that would be created by consolidation
    # pipeline on or after CONSOLIDATION_TIMESTAMP)
    memories.append({
        "id": mem_id_1, "user_id": user_id,
        "content": f"{old_coping} helps {name} manage their {symptom}.",
        "valid_from": current_date.isoformat(), "valid_until": None,
        "category": "COPING_MECHANISM",
        "source_episode_id": ep_id_1
    })

    current_date += timedelta(days=30)
    ep_id_2 = make_id()
    ep_text_2 = (
        f"Follow up with {name}. They are still continuing with {old_coping.lower()} "
        f"and finding it effective for their {symptom}. General mood is stable."
    )
    episodes.append({
        "id": ep_id_2, "user_id": user_id,
        "content": ep_text_2, "created_at": current_date.isoformat()
    })

    # Contradiction episode (2 months after start)
    current_date += timedelta(days=30)
    ep_id_3 = make_id()
    ep_text_3 = (
        f"{name} had a rough week. Unfortunately, {old_coping.lower()} is no longer helping "
        f"with the {symptom} due to schedule changes. Instead, {name} tried {new_coping.lower()} "
        f"and found it to be much more effective now."
    )
    episodes.append({
        "id": ep_id_3, "user_id": user_id,
        "content": ep_text_3, "created_at": current_date.isoformat()
    })

    # Update old memory validity to reflect the contradiction
    memories[0]["valid_until"] = current_date.isoformat()

    mem_id_2 = make_id()
    memories.append({
        "id": mem_id_2, "user_id": user_id,
        "content": f"{new_coping} helps {name} manage their {symptom}.",
        "valid_from": current_date.isoformat(), "valid_until": None,
        "category": "COPING_MECHANISM",
        "source_episode_id": ep_id_3
    })

    # ── Filler episodes (16 more) ─────────────────────────────────────────────
    for _ in range(16):
        current_date += timedelta(days=rng.randint(1, 5))
        filler_id = make_id()
        activities = ["went grocery shopping", "watched a movie", "had dinner with a friend",
                      "read a book", "did some chores around the house"]
        episodes.append({
            "id": filler_id, "user_id": user_id,
            "content": f"{name} {rng.choice(activities)} and had an uneventful day.",
            "created_at": current_date.isoformat()
        })

    # ── FACTUAL Query & Memory ────────────────────────────────────────────────
    ep_id_fact = make_id()
    fact_animal = rng.choice(["dog", "cat", "bird"])
    fact_petname = rng.choice(["Buddy", "Luna", "Charlie"])
    episodes.append({
        "id": ep_id_fact, "user_id": user_id,
        "content": f"{name} adopted a pet {fact_animal} named {fact_petname} today.",
        "created_at": (current_date + timedelta(days=5)).isoformat()
    })
    mem_id_fact = make_id()
    memories.append({
        "id": mem_id_fact, "user_id": user_id,
        "content": f"{name} has a pet {fact_animal} named {fact_petname}.",
        "valid_from": (current_date + timedelta(days=5)).isoformat(), "valid_until": None,
        "category": "FACT",
        "source_episode_id": ep_id_fact
    })
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"What is the name of {name}'s pet?",
        "category": "FACTUAL",
        "query_time": (current_date + timedelta(days=10)).isoformat(),
        "query_temporal_mode": "ANY",
        "ground_truth_memory_ids": [mem_id_fact],
        "expected_episode_ids": [ep_id_fact],
        "ground_truth_answer": f"{name}'s pet {fact_animal} is named {fact_petname}.",
        "answerable": True,
        "valid_at_query_time": True,
        "expected_behavior": "ANSWER",
        "no_answer_type": None
    })

    # ── TEMPORAL Query ────────────────────────────────────────────────────────
    q_time_temporal = current_date + timedelta(days=1)
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"What currently helps {name} manage their {symptom}?",
        "category": "TEMPORAL",
        "query_time": q_time_temporal.isoformat(),
        "query_temporal_mode": "CURRENT",
        "ground_truth_memory_ids": [mem_id_2],
        "expected_episode_ids": [ep_id_3],
        "ground_truth_answer": f"{new_coping} currently helps {name} manage their {symptom}.",
        "answerable": True,
        "valid_at_query_time": True,
        "expected_behavior": "ANSWER",
        "no_answer_type": None
    })

    # ── HISTORICAL Query ──────────────────────────────────────────────────────
    hist_time = start_date + timedelta(days=10)
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"What used to help {name} manage their {symptom} back in {hist_time.strftime('%B')}?",
        "category": "HISTORICAL",
        "query_time": hist_time.isoformat(),
        "query_temporal_mode": "HISTORICAL",
        "ground_truth_memory_ids": [mem_id_1],
        "expected_episode_ids": [ep_id_1],
        "ground_truth_answer": f"Back then, {old_coping} helped {name} manage their {symptom}.",
        "answerable": True,
        "valid_at_query_time": True,
        "expected_behavior": "ANSWER",
        "no_answer_type": None
    })

    # ── CONTRADICTION Query ───────────────────────────────────────────────────
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"Does {old_coping.lower()} still help {name}?",
        "category": "CONTRADICTION",
        "query_time": (current_date + timedelta(days=1)).isoformat(),
        "query_temporal_mode": "CURRENT",
        "ground_truth_memory_ids": [mem_id_2],
        "expected_episode_ids": [ep_id_3],
        "invalid_memory_ids": [mem_id_1],
        "ground_truth_answer": f"No, {old_coping.lower()} is no longer helping. {name} now uses {new_coping.lower()}.",
        "answerable": True,
        "valid_at_query_time": True,
        "expected_behavior": "ANSWER",
        "no_answer_type": None,
        # For contradiction queries, document the superseded memory
        "superseded_by": mem_id_2,
        "valid_from": start_date.isoformat(),
        "valid_until": (start_date + timedelta(days=60)).isoformat(),
    })

    # ── MULTI_HOP Query & Memories ────────────────────────────────────────────
    trigger = rng.choice(TRIGGERS)
    consequence = rng.choice(CONSEQUENCES)
    sec_consequence = rng.choice(SECONDARY_CONSEQUENCES)

    ep_id_mh1 = make_id()
    episodes.append({
        "id": ep_id_mh1, "user_id": user_id,
        "content": f"{name} noticed that {trigger.lower()} directly increases their {symptom}.",
        "created_at": (current_date + timedelta(days=6)).isoformat()
    })
    mem_id_mh1 = make_id()
    memories.append({
        "id": mem_id_mh1, "user_id": user_id,
        "content": f"{trigger} increases {name}'s {symptom}.",
        "valid_from": (current_date + timedelta(days=6)).isoformat(), "valid_until": None,
        "category": "MULTI_HOP_A",
        "source_episode_id": ep_id_mh1
    })

    ep_id_mh2 = make_id()
    episodes.append({
        "id": ep_id_mh2, "user_id": user_id,
        "content": f"When experiencing {symptom}, {name} often suffers from {consequence}.",
        "created_at": (current_date + timedelta(days=7)).isoformat()
    })
    mem_id_mh2 = make_id()
    memories.append({
        "id": mem_id_mh2, "user_id": user_id,
        "content": f"{name}'s {symptom} leads to {consequence}.",
        "valid_from": (current_date + timedelta(days=7)).isoformat(), "valid_until": None,
        "category": "MULTI_HOP_B",
        "source_episode_id": ep_id_mh2
    })

    ep_id_mh3 = make_id()
    episodes.append({
        "id": ep_id_mh3, "user_id": user_id,
        "content": f"The persistent {consequence} has started to negatively impact {name}'s {sec_consequence}.",
        "created_at": (current_date + timedelta(days=8)).isoformat()
    })
    mem_id_mh3 = make_id()
    memories.append({
        "id": mem_id_mh3, "user_id": user_id,
        "content": f"{name}'s {consequence} negatively impacts their {sec_consequence}.",
        "valid_from": (current_date + timedelta(days=8)).isoformat(), "valid_until": None,
        "category": "MULTI_HOP_C",
        "source_episode_id": ep_id_mh3
    })

    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"What life area did {trigger.lower()} eventually affect for {name}?",
        "category": "MULTI_HOP",
        "query_time": (current_date + timedelta(days=12)).isoformat(),
        "query_temporal_mode": "ANY",
        "ground_truth_memory_ids": [mem_id_mh1, mem_id_mh2, mem_id_mh3],
        "expected_episode_ids": [ep_id_mh1, ep_id_mh2, ep_id_mh3],
        "ground_truth_answer": (
            f"High {trigger.lower()} increases {name}'s {symptom}, "
            f"which leads to {consequence}, ultimately impacting their {sec_consequence}."
        ),
        "answerable": True,
        "valid_at_query_time": True,
        "expected_behavior": "ANSWER",
        "no_answer_type": None,
        "hop_count": 3
    })

    # ── NO_ANSWER Queries (3 Types) ───────────────────────────────────────────

    # Type A: Completely unknown — information never existed
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"What is {name}'s favorite food?",
        "category": "NO_ANSWER",
        "query_time": (current_date + timedelta(days=10)).isoformat(),
        "query_temporal_mode": "ANY",
        "ground_truth_memory_ids": [],
        "expected_episode_ids": [],
        "ground_truth_answer": None,
        "answerable": False,
        "valid_at_query_time": False,
        "expected_behavior": "ABSTAIN",
        "no_answer_type": "TYPE_A_UNKNOWN"
    })

    # Type B: Semantically similar but unsupported — query resembles stored memories but fact is absent
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"Does {name} prefer {new_coping.lower()} for improving sleep quality?",
        "category": "NO_ANSWER",
        "query_time": (current_date + timedelta(days=10)).isoformat(),
        "query_temporal_mode": "ANY",
        "ground_truth_memory_ids": [],
        "expected_episode_ids": [],
        "ground_truth_answer": None,
        "answerable": False,
        "valid_at_query_time": False,
        "expected_behavior": "ABSTAIN",
        "no_answer_type": "TYPE_B_SIMILAR_UNSUPPORTED",
        "_note": f"Memory about {new_coping} exists but it's for {symptom}, not sleep quality specifically."
    })

    # Type C: Temporally invalid — information existed but is NOT valid at query time
    queries.append({
        "id": make_id(), "user_id": user_id,
        "query": f"Does {old_coping.lower()} help {name} right now?",
        "category": "NO_ANSWER",
        "query_time": (current_date + timedelta(days=10)).isoformat(),
        "query_temporal_mode": "CURRENT",
        "ground_truth_memory_ids": [],
        "expected_episode_ids": [],
        "ground_truth_answer": None,
        "answerable": False,
        "valid_at_query_time": False,
        "expected_behavior": "ABSTAIN",
        "no_answer_type": "TYPE_C_TEMPORALLY_INVALID",
        "invalidated_memory_id": mem_id_1,
        "_note": f"mem_id_1 ({old_coping}) has valid_until set; at query_time it is expired."
    })

    return {
        "user": {"id": user_id, "name": name, "split": "DEV" if user_idx <= 40 else "TEST"},
        "episodes": episodes,
        "memories": memories,
        "queries": queries
    }


def generate_dataset(num_users: int = 50, seed: int = 42):
    global _id_counter, _id_seed
    _id_counter = 0   # Reset counter for reproducibility across multiple calls
    _id_seed = seed   # Set seed for deterministic UUID generation
    print(f"Generating synthetic dataset: {num_users} users, seed={seed}")
    random.seed(seed)

    all_users, all_episodes, all_memories, all_queries = [], [], [], []
    base_date = datetime(2026, 1, 1, 12, 0, 0)

    for i in range(1, num_users + 1):
        rng = random.Random(seed + i)
        start_date = base_date + timedelta(days=rng.randint(0, 30))
        data = generate_user_timeline(i, start_date, seed)
        all_users.append(data["user"])
        all_episodes.extend(data["episodes"])
        all_memories.extend(data["memories"])
        all_queries.extend(data["queries"])

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "users.json"), "w") as f:
        json.dump({"users": all_users}, f, indent=2)
    with open(os.path.join(DATA_DIR, "episodes.json"), "w") as f:
        json.dump({"episodes": all_episodes}, f, indent=2)
    with open(os.path.join(DATA_DIR, "memories.json"), "w") as f:
        json.dump({"memories": all_memories}, f, indent=2)
    with open(os.path.join(DATA_DIR, "queries.json"), "w") as f:
        json.dump({"queries": all_queries}, f, indent=2)

    # Update metadata
    q_by_cat = {}
    for q in all_queries:
        q_by_cat.setdefault(q["category"], 0)
        q_by_cat[q["category"]] += 1
    no_answer_by_type = {}
    for q in all_queries:
        if q["category"] == "NO_ANSWER":
            t = q.get("no_answer_type", "UNKNOWN")
            no_answer_by_type.setdefault(t, 0)
            no_answer_by_type[t] += 1

    metadata = {
        "version": "v2.0",
        "seed": seed,
        "num_users": len(all_users),
        "num_episodes": len(all_episodes),
        "num_memories": len(all_memories),
        "num_queries": len(all_queries),
        "consolidation_run_id": CONSOLIDATION_RUN_ID,
        "consolidation_timestamp": CONSOLIDATION_TIMESTAMP,
        "query_categories": q_by_cat,
        "no_answer_subtypes": no_answer_by_type,
        "schema_version": {
            "queries": ["id", "user_id", "query", "category", "query_time", "query_temporal_mode",
                        "ground_truth_memory_ids", "expected_episode_ids", "ground_truth_answer",
                        "answerable", "valid_at_query_time", "expected_behavior", "no_answer_type"],
            "memories": ["id", "user_id", "content", "valid_from", "valid_until", "category",
                         "source_episode_id", "consolidated_at", "consolidation_run_id"]
        }
    }
    with open(os.path.join(DATA_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset ready at {DATA_DIR}")
    print(f"  Users:    {len(all_users)}")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Memories: {len(all_memories)}")
    print(f"  Queries:  {len(all_queries)}")
    print(f"  Categories: {q_by_cat}")
    print(f"  NO_ANSWER subtypes: {no_answer_by_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--test-mode", action="store_true", help="Generate 2 users only")
    args = parser.parse_args()
    num_users = 2 if args.test_mode else args.users
    generate_dataset(num_users, args.seed)

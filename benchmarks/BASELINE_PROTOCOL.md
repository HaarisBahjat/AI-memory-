# Baseline Protocol — Longitudinal AI Memory System Benchmark

**Version:** 1.0  
**Status:** Frozen for Phase 9.4 evaluation  
**Date:** 2026-08-30

---

## Preamble

This document precisely defines every system configuration evaluated in the benchmark.
Each baseline must differ from every other in exactly one measurable dimension.

No baseline result may be compared against another unless both use **the same ground-truth type**.

---

## Ground-Truth Type Definitions

| ID | Ground-Truth Set | Object Type | Retrieved By |
|---|---|---|---|
| **GT-M** | `ground_truth_memory_ids` | Semantic memory UUIDs | Vector search over `semantic_memories` table |
| **GT-E** | `expected_episode_ids` | Episode UUIDs | Recency window over `episodes` table |

> **Critical rule:** Baselines A and B are evaluated using GT-E (episode IDs).  
> Baselines C, D, E, and the Proposed System are evaluated using GT-M (memory IDs).  
> These two result sets must **never be numerically compared** directly.

---

## Baseline A — No Memory Context

**system_id:** `no_memory`  
**Ground Truth:** GT-E (evaluated on answer quality, not retrieval)

### Pipeline
```
Query → LLM (no memory context) → Answer
```

### What it measures
The floor performance of an LLM without any memory access. Evaluates pure language model capability on longitudinal questions.

### Valid comparisons
- Answer quality metrics only (strict accuracy, hallucination rate, abstention rate)
- **Cannot** be fairly compared against retrieval F1@5 using memory IDs

### Notes
Baseline A scores 0.167 on retrieval metrics only because abstention scoring gives it credit for NO_ANSWER queries (it retrieves nothing, which is correct for 1-in-6 queries). This is an artifact of the evaluation design, not system capability.

---

## Baseline B — Recency Window (Raw Episode Retrieval)

**system_id:** `recency_window`  
**Ground Truth:** GT-E (evaluated against `expected_episode_ids`)

### Pipeline
```
Query → Fetch last N episodes for user (by timestamp) → Return episode IDs
```

### Configuration
```yaml
features:
  episodic_memory: true
  semantic_memory: false
  temporal_layer: false
  knowledge_graph: false

retrieval:
  episode_count: 5   # Last 5 episodes by descending timestamp
```

### What it measures
Whether simple recency-based episode lookup retrieves relevant context.

### Valid comparisons
- Against GT-E (episode IDs) only
- **Must not** be compared against GT-M baselines using the same F1@5 column

### Notes
The reason Baseline B scores 0.000 on GT-M metrics is that it retrieves episode UUIDs, not semantic memory UUIDs. The two object types do not intersect. This is an **evaluation design issue**, not a system failure. Baseline B must be evaluated on its own ground-truth type (GT-E).

---

## Baseline C — Vector RAG (Semantic Memory Only)

**system_id:** `vector_only`  
**Ground Truth:** GT-M (evaluated against `ground_truth_memory_ids`)

### Pipeline
```
Query
  → Embed(query)
  → cosine_similarity(query_emb, semantic_memories)
  → Top-K results (no temporal filtering)
```

### Configuration
```yaml
features:
  semantic_memory: true
  temporal_layer: false
  knowledge_graph: false

retrieval:
  top_k: 5
  similarity_threshold: 0.65
  decay_lambda: 0.0
```

### What it measures
Standard vector retrieval without temporal awareness.

### The critical reviewer question
> "Is temporal filtering actually necessary, or does vector search alone perform comparably?"

Baseline C is the primary ablation for the temporal layer. The Proposed System must demonstrate a statistically significant improvement over Baseline C to support temporal filtering claims.

---

## Baseline D — Vector RAG + Temporal Decay (Pre-Consolidation Memories)

**system_id:** `recency_decay_preconsolidation`  
**Ground Truth:** GT-M  
**Memory source:** All semantic memories regardless of `consolidation_run_id`

### Pipeline
```
Query
  → Embed(query)
  → Top-K cosine similarity retrieval
  → Temporal validity filter (valid_from ≤ query_time ≤ valid_until)
  → Time-decay reranking:
       score_final = similarity_score × exp(−λ × age_in_days)
       where λ = 0.005 (configurable)
```

### Configuration
```yaml
features:
  semantic_memory: true
  temporal_layer: true
  knowledge_graph: false
  consolidation: false   # Retrieves from ALL memories, not only consolidated ones

retrieval:
  top_k: 5
  similarity_threshold: 0.65
  decay_lambda: 0.005
```

### What it measures
The benefit of temporal filtering and decay reranking WITHOUT requiring the semantic consolidation pipeline to have run.

### The critical reviewer question
> "Does the consolidation pipeline provide additional benefit beyond temporal decay?"

Baseline D is the primary ablation for the consolidation layer. If Baseline E significantly outperforms Baseline D, it supports the value of semantic consolidation.

---

## Baseline E — Consolidated Semantic Memory (Post-Consolidation Only)

**system_id:** `consolidated_semantic`  
**Ground Truth:** GT-M  
**Memory source:** Only memories where `consolidation_run_id = CONSOLIDATION_RUN_ID` AND `consolidated_at IS NOT NULL`

### Pipeline
```
Query
  → Embed(query)
  → Top-K cosine similarity retrieval
      FILTER: consolidated_at IS NOT NULL AND consolidation_run_id = ?
  → Temporal validity filter
  → Time-decay reranking (same λ as Baseline D)
```

### Configuration
```yaml
features:
  semantic_memory: true
  temporal_layer: true
  knowledge_graph: false
  consolidation: true     # Restricts retrieval to post-consolidation memories ONLY

retrieval:
  top_k: 5
  similarity_threshold: 0.65
  decay_lambda: 0.005

consolidation:
  run_id: "consolidation_run_2026_phase2"
  filter_field: "consolidation_run_id"
```

### What it measures
The retrieval quality when only memories produced by the semantic consolidation pipeline are available. This reflects the real production scenario where the system has run consolidation.

### Baseline D vs Baseline E — Key distinction
- **Baseline D:** Retrieves from ANY semantic memory (consolidated or raw)
- **Baseline E:** Retrieves only from memories produced by a specific consolidation run

Since all memories in this dataset have `consolidated_at` set, Baselines D and E will produce similar results with the current dataset. A future benchmark with a mix of consolidated and raw memories will show the distinction more clearly. This is documented as a known limitation.

---

## Proposed System — Full Layered Memory Architecture

**system_id:** `full_system`  
**Ground Truth:** GT-M

### Pipeline
```
Query
  → Embed(query)
  → Top-K cosine similarity retrieval from consolidated memories
  → Temporal validity filter (valid_from ≤ query_time ≤ valid_until)
  → Time-decay reranking
  → Graph expansion (1-hop category-based traversal)
  → Final re-rank and select top-K
```

### Configuration
```yaml
features:
  semantic_memory: true
  temporal_layer: true
  knowledge_graph: true
  consolidation: true

retrieval:
  top_k: 5
  similarity_threshold: 0.65
  decay_lambda: 0.005

consolidation:
  run_id: "consolidation_run_2026_phase2"

graph:
  max_depth: 1             # 1-hop category-based expansion
  seed_top_k: 3
```

### What it measures
The complete proposed architecture including the graph expansion layer. Must outperform Baseline D/E to justify the graph layer — especially on MULTI_HOP queries.

---

## Ablation Configurations

### `no_temporal` — Full System minus temporal layer
Identical to the Proposed System but `temporal_layer: false`. Tests the isolated contribution of temporal validity filtering and decay reranking.

### `no_graph` — Full System minus graph expansion
Identical to the Proposed System but `knowledge_graph: false`. Tests the isolated contribution of graph expansion.

### `no_consolidation` — Full System minus consolidated memories
Uses episodic recency fallback (no semantic memory). Evaluated against **GT-E**, not GT-M. Cannot be compared numerically against Baselines C–E.

---

## Comparison Validity Matrix

| Comparison | Valid? | Ground Truth | Metric |
|---|---|---|---|
| Proposed vs Baseline C | ✅ | GT-M | F1@5, MRR |
| Proposed vs Baseline D | ✅ | GT-M | F1@5, MRR |
| Proposed vs Baseline E | ✅ | GT-M | F1@5, MRR |
| Proposed vs `no_temporal` | ✅ | GT-M | F1@5, MRR |
| Proposed vs `no_graph` | ✅ | GT-M | F1@5, MRR |
| Baseline C vs Baseline D | ✅ | GT-M | F1@5, MRR |
| Proposed vs Baseline A | ⚠️ | Different GT types | Answer quality only |
| Proposed vs Baseline B | ⚠️ | Different GT types | Cannot compare retrieval F1 |
| Proposed vs `no_consolidation` | ⚠️ | Different GT types | Storage and latency only |
| Baseline B vs any GT-M system | ❌ | Incompatible | Never compare numerically |

---

## Decay Function Documentation

$$\text{score\_final} = \text{similarity\_score} \times e^{-\lambda \cdot \Delta t}$$

Where:
- $\lambda = 0.005$ (selected empirically; approximately halves weight after 139 days)
- $\Delta t$ = age of memory in days at query time, computed as `query_time - created_at`
- Temporal validity is checked separately: memories with `valid_until < query_time` are filtered out **before** decay scoring

---

## Reproducibility

All configurations are stored as YAML files in `benchmarks/configs/`.
The consolidation run ID `consolidation_run_2026_phase2` must match the value in `generate_dataset.py`.
Re-running `generate_dataset.py --seed 42` will reproduce identical dataset files.

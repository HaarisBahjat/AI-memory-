# Longitudinal AI Memory System — Benchmark Suite

**Version:** 2.0 (Phase 9.4)  
**Status:** Paper-ready (Research Validity Audit Complete)

---

## Environment

| Variable | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.11.x |
| PostgreSQL | 16.x |
| pgvector extension | 0.7.x |
| Embedding model | `text-embedding-3-small` (1536 dim) |
| LLM judge model | `gpt-4o-mini` |
| Random seed | `42` (default) |

Required environment variables (set in `.env`):
```
OPENAI_API_KEY=...
DATABASE_URL=postgresql+asyncpg://...
```

---

## Installation

```bash
pip install -r benchmarks/requirements-benchmark.txt
```

---

## Benchmark Execution Order (Full Pipeline)

Run all commands from the **repository root** (not the benchmarks directory).

### 1. Generate Dataset

```bash
python benchmarks/scripts/generate_dataset.py --seed 42 --users 50
```

Outputs:
- `benchmarks/dataset/synthetic/queries.json` — 400 queries (8 categories × 50 users)
- `benchmarks/dataset/synthetic/episodes.json` — 1,100 raw episodes
- `benchmarks/dataset/synthetic/memories.json` — 250 consolidated memories
- `benchmarks/dataset/synthetic/dataset_metadata.json` — full provenance metadata

### 2. Seed the Database

```bash
python benchmarks/scripts/rag_benchmark.py --seed
```

This:
- Clears previous benchmark data from PostgreSQL
- Inserts all users, episodes, and semantic memories
- Computes and caches embeddings (via OpenAI API, cached to `benchmarks/cache/`)

### 3. Run Retrieval Benchmarks

```bash
python benchmarks/scripts/rag_benchmark.py
```

Runs all 9 configurations:
- `baseline_a` — No memory
- `baseline_b` — Recency window (GT-E evaluation)
- `baseline_c_vector` — Vector RAG only
- `baseline_d_recency_decay` — Vector + Temporal Decay
- `baseline_e_consolidated` — Consolidated Semantic Memory only
- `proposed_full` — Full System (Vector + Temporal + Graph)
- `no_temporal` — Ablation: remove temporal layer
- `no_graph` — Ablation: remove graph layer
- `no_consolidation` — Ablation: episode fallback (GT-E)

Or run a single config:
```bash
python benchmarks/scripts/rag_benchmark.py --config proposed_full
```

### 4. Run Multi-Hop Dedicated Benchmark

```bash
python benchmarks/scripts/multihop_benchmark.py
```

Evaluates only MULTI_HOP queries and computes:
- Both-Facts-Retrieved Rate
- Graph benefit verdict (honest — reports "no improvement" if applicable)

### 5. Run Consolidation Benchmark

```bash
python benchmarks/scripts/consolidation_benchmark.py
```

Computes:
- Storage: raw episode text bytes vs consolidated memory text bytes
- Information preservation rate (via source_episode_id mapping)
- Experiment A: Episode retrieval on GT-E ground truth
- Experiment B: References to semantic retrieval results from step 3

### 6. Run Statistical Analysis

```bash
python benchmarks/scripts/statistical_analysis.py
```

Requires step 3 to have run first. Computes:
- Wilcoxon Signed-Rank Test (real, not hardcoded)
- Rank-biserial correlation effect size
- Bootstrap 95% confidence intervals
- Benjamini-Hochberg FDR correction

Outputs:
- `benchmarks/results/statistical_analysis.json`
- `benchmarks/results/statistical_analysis.csv`

### 7. Run Scalability Benchmark

```bash
python benchmarks/scripts/scalability_benchmark.py
```

> ⚠️ This inserts up to 10,000 synthetic memories and runs 100 timed queries per scale.
> Takes approximately 10-15 minutes. Cleans up test data after completion.

Tests retrieval latency at scales: 100, 1k, 5k, 10k  
Extrapolates to 50k and 100k using log-linear fit (clearly labeled as projections).

Outputs:
- `benchmarks/results/scalability_analysis.json`
- `benchmarks/results/latency_vs_memory_size.csv`

### 8. Run Answer Quality Benchmark (LLM calls — has cost)

```bash
python benchmarks/scripts/answer_benchmark.py
```

> ⚠️ This calls `gpt-4o-mini` for answer generation and LLM-judge scoring.
> Cost: approximately $0.02-0.05 USD for the default 50-query sample.

### 9. LLM Judge Validation (Human annotation required)

```bash
# Step 1: Generate annotation template
python benchmarks/scripts/human_judge_validation.py --generate-template --n 30

# Step 2: Fill in human_score field in the generated JSON file
# benchmarks/results/human_annotation_template.json

# Step 3: Compute Cohen's Kappa
python benchmarks/scripts/human_judge_validation.py --human-scores benchmarks/results/human_annotation_template.json
```

### 10. Multi-Seed Robustness Check

```bash
python benchmarks/scripts/run_multi_seed.py
```

> ⚠️ Runs the full pipeline 5 times (seeds 42, 123, 456, 789, 2026). Takes ~30-45 minutes.
> Only runs deterministic retrieval (no LLM calls).

---

## Ground-Truth Types — Critical Note

| Baseline | Ground-Truth Type | Object Type |
|---|---|---|
| Baseline A | N/A (answer quality only) | — |
| **Baseline B** | **GT-E** (`expected_episode_ids`) | Episode UUIDs |
| Baseline C, D, E | GT-M (`ground_truth_memory_ids`) | Semantic Memory UUIDs |
| Full System, Ablations | GT-M | Semantic Memory UUIDs |

**Baseline B results CANNOT be numerically compared against Baselines C–E.**  
See `benchmarks/BASELINE_PROTOCOL.md` for the full comparison validity matrix.

---

## Results Directory Structure

```
benchmarks/results/
  run_<timestamp>/
    retrieval_<config>.json           ← Aggregate + per-category breakdown
    retrieval_<config>_per_query.json ← Per-query F1@5, category, diagnostics
  statistical_analysis.json           ← Real Wilcoxon tests + FDR
  statistical_analysis.csv            ← Per-query paired data
  scalability_analysis.json           ← Real latency measurements
  latency_vs_memory_size.csv          ← Scale vs P50/P95/P99
  consolidation_analysis.json         ← Fair consolidation experiment
  multihop_analysis.json              ← Dedicated multi-hop evaluation
  multi_seed_summary.json             ← Variance across seeds
  human_annotation_template.json      ← For LLM judge validation
  llm_judge_validation.json           ← Cohen's Kappa results
```

---

## Cost Analysis

| Operation | Per Run | Per 100 Queries |
|---|---|---|
| Embedding (text-embedding-3-small, 1536d) | $0.000130/1k tokens | ~$0.002 |
| RAG generation (gpt-4o-mini) | $0.00015/1k input + $0.0006/1k output | ~$0.012 |
| LLM judge (gpt-4o-mini) | same as above | ~$0.005 |
| Consolidation (one-time) | $0.002-0.008 | N/A |

**Infrastructure (local hardware, no dollar cost):**
- CPU: [see benchmarks/protocol/environment.md]
- RAM: 16 GB
- Storage: ~600 MB for 10k vector embeddings
- PostgreSQL single-node, no cloud costs

Infrastructure cost is NOT expressed in dollar terms for local deployments.

---

## Reproducibility Checklist

- [ ] Set random seed: `--seed 42`
- [ ] Run `generate_dataset.py` before any benchmark
- [ ] Run `rag_benchmark.py --seed` before `rag_benchmark.py`
- [ ] Embedding cache is in `benchmarks/cache/` — share this cache for exact reproduction
- [ ] Model versions: `text-embedding-3-small` (2024-09 checkpoint), `gpt-4o-mini-2024-07-18`
- [ ] All per-query results stored in timestamped `run_*` directories
- [ ] `statistical_analysis.py` reads the **latest** run directory automatically

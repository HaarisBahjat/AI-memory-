# Limitations — Longitudinal AI Memory System

This document enumerates the known limitations of the system and the evaluation methodology for inclusion in the research paper.

---

## 1. Synthetic Dataset Bias

**Limitation:** All 300 evaluation queries were generated from procedural templates using a fixed vocabulary of symptoms, coping mechanisms, and triggers. The dataset structure is deterministic (seed=42) and may not capture the full distribution of real-world user queries.

**Impact:** Retrieval metrics may be optimistically biased because the synthetic memories were generated to semantically match the query templates. Real-world queries involve more paraphrasing, ambiguity, and domain shift.

**Mitigation:** A human-authored validation subset (20–30 queries) is recommended for sanity-checking. The benchmark architecture is designed to support real query injection via `inject_human_queries.py`.

---

## 2. Knowledge Graph Implementation

**Limitation:** The "graph expansion" in this benchmark is a simplified 1-hop category-based traversal, not a full Temporal Knowledge Graph (TKG) as described in the system design. The production `TemporalGraphEngine` (in `app/services/graph_service.py`) is the intended full implementation.

**Impact:** Multi-hop results may understate the potential benefit of a full TKG traversal with typed edges, relationship weights, and temporal validity on graph edges.

**Mitigation:** The graph expansion is explicitly documented as an approximation. Paper claims about graph benefit should be scoped to the implemented functionality.

---

## 3. Single Hardware Environment

**Limitation:** All latency measurements were performed on a single local machine with no database load from concurrent users. Production environments with multiple simultaneous users will exhibit different (likely higher) latencies.

**Impact:** P50/P95/P99 latency claims are optimistic for production deployment.

**Mitigation:** Clearly state the hardware environment (see `benchmarks/protocol/environment.md`) and add a disclaimer that latencies are for single-user sequential evaluation.

---

## 4. LLM Judge Validation

**Limitation:** The LLM-as-a-judge (gpt-4o-mini) evaluation was run on a 5% sample (15 queries per 300). Inter-annotator agreement (Cohen's Kappa) between the LLM judge and human annotators was not measured due to the cost of human annotation.

**Impact:** Claims about "Strict Accuracy" and "Hallucination Rate" are based on LLM-judged assessments, which may not perfectly align with human judgements.

**Mitigation:** The `human_judge_validation.py` script can be used to measure Cohen's Kappa on a small human-labelled subset when budget permits.

---

## 5. Memory Consolidation Oracle

**Limitation:** Semantic consolidation in the benchmark is performed by a scripted template (not by running the actual consolidation pipeline). The 250 semantic memories were manually generated to represent the consolidation output.

**Impact:** The real consolidation pipeline (nightly LLM-based fact extraction) may produce different memory texts and coverage, affecting retrieval quality.

**Mitigation:** Future work should run the live consolidation pipeline on a cohort of real users and re-benchmark.

---

## 6. Temporal Scope

**Limitation:** The synthetic dataset covers a ~5-month window (January–May 2026). Memory decay parameters (lambda=0.005) are tuned for this timeframe. For much longer or shorter memory windows, re-tuning may be necessary.

**Impact:** Time-decay scores may not generalise to longitudinal datasets spanning years.

---

## 7. Category Balance in Aggregate Metrics

**Limitation:** The 6 query categories contribute equally (50 queries each) to the aggregate F1@5. This means the graph layer's contribution on MULTI_HOP queries is diluted by the 5x larger set of non-multi-hop queries in the aggregate.

**Impact:** Aggregate F1@5 is not the right metric to evaluate the graph layer specifically. Category-stratified metrics are required.

**Mitigation:** Category-stratified results should be reported alongside aggregate metrics in the paper.

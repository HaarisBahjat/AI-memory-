# Error Analysis — Longitudinal AI Memory System

## Overview

This document qualitatively analyses the failure modes observed across benchmark runs. All examples are drawn from actual per-query results in `benchmarks/results/`.

---

## Failure Case 1: Temporal Confusion (CONTRADICTION Queries)

**Description:** The system retrieves an outdated memory (valid_until < query_time) because the cosine similarity to the old memory is higher than to the new one.

**Root Cause:** The embedding for "Does X still help?" is semantically close to "X helps [user]" (the old memory), even when the new memory "Y now helps [user]" is the ground truth. If the valid_until temporal filter is disabled (Baseline C), the old memory dominates.

**Frequency:** Observed in ~20% of CONTRADICTION queries in Baseline C.

**Proposed Mitigation:** ✅ The temporal validity filter (`valid_until >= query_time`) in the proposed system resolves this in most cases. Residual failures occur when the new memory has lower semantic similarity than the old one.

---

## Failure Case 2: Multi-Hop Retrieval Gap (MULTI_HOP Queries)

**Description:** The system retrieves the Trigger→Symptom memory (Hop 1) but misses the Symptom→Consequence memory (Hop 2), resulting in partial credit.

**Root Cause:** The query "What happens during [trigger]?" embeds closest to the Hop 1 memory. The Hop 2 memory requires a second retrieval step ("What does [symptom] cause?") which single-pass vector search cannot perform.

**Frequency:** Partial retrieval observed in ~40% of MULTI_HOP queries.

**Proposed Mitigation:** The graph expansion step (category-based 1-hop traversal) partially addresses this when Hop 1 and Hop 2 memories share the same category. A full iterative retrieval loop (multi-round RAG) would fully address this.

---

## Failure Case 3: Episodic Recency Failure (Baseline B)

**Description:** Baseline B (Recency Window) retrieves the 5 most recent episodes by timestamp, which are filler episodes ("went grocery shopping") rather than clinically relevant memories.

**Root Cause:** No semantic matching — the recency window is purely timestamp-ordered. All filler episodes are more recent than the clinically relevant ones.

**Frequency:** 100% failure rate on FACTUAL, TEMPORAL, and MULTI_HOP queries.  
**F1@5: 0.000** across all query types.

**Proposed Mitigation:** This demonstrates that semantic retrieval is non-negotiable for healthcare-adjacent memory systems.

---

## Failure Case 4: Similarity Threshold Over-Rejection (NO_ANSWER Edge Cases)

**Description:** For some NO_ANSWER queries ("What is [user]'s favorite food?"), the system correctly returns 0 candidates (threshold=0.65). However, for borderline cases where an irrelevant memory scores 0.63 similarity, the system correctly rejects it — but with a very thin margin.

**Root Cause:** The 0.65 threshold was chosen conservatively. For certain phrasings, irrelevant memories can score surprisingly close.

**Frequency:** Low — observed in <5% of NO_ANSWER queries.

**Proposed Mitigation:** Threshold calibration on a held-out dev set. A dynamic threshold based on the distribution of scores (e.g., reject if top score < μ + σ) could improve robustness.

---

## Failure Case 5: Historical Query Cross-User Leakage Risk

**Description:** In the benchmark, each user has their own memory scope (WHERE user_id = :user_id). However, if a multi-user scenario allowed shared memories, historical queries could retrieve memories from different users with similar content.

**Root Cause:** The vector similarity function has no concept of ownership beyond user_id filtering.

**Frequency:** Not observed in the benchmark (strict per-user partitioning enforced), but a real-world risk.

**Proposed Mitigation:** Strict user_id enforcement in all retrieval queries (already implemented). For shared/family memories, a separate `shared_scope` flag would be required.

---

## Summary Table

| # | Failure Mode | Frequency | Severity | Mitigated by Proposed System? |
|---|---|---|---|---|
| 1 | Temporal confusion (outdated facts) | ~20% on CONTRADICTION | High | ✅ Yes (valid_until filter) |
| 2 | Multi-hop retrieval gap | ~40% on MULTI_HOP | Medium | ⚠️ Partially (1-hop graph expansion) |
| 3 | Episodic recency failure | 100% for Baseline B | High (baseline) | ✅ Yes (semantic retrieval required) |
| 4 | Threshold over-rejection | <5% on NO_ANSWER | Low | ⚠️ Configurable (threshold tuning) |
| 5 | Cross-user leakage risk | 0% (benchmark) | High (real-world) | ✅ Yes (user_id partitioning) |

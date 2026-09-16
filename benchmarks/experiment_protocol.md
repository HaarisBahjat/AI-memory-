# Experiment Protocol — AI Wellness Longitudinal Memory System
**Version:** v1.0  
**Date:** 2026-08-29  
**Status:** Pre-registered (frozen before any results are observed)

> **Pre-registration statement:**  
> This document defines the research questions, hypotheses, primary metrics, baselines, dataset structure, statistical tests, and multiple-comparison correction *before* any experimental runs are executed. Changes made after observing results must be documented as exploratory analyses and clearly distinguished from the confirmatory results below.

---

## 1. Primary Research Question

> **Does a layered memory architecture combining episodic memory, semantic consolidation, temporal validity, knowledge-graph reasoning, and hybrid retrieval improve longitudinal personalized AI performance compared with simpler memory and retrieval baselines?**

---

## 2. Research Questions

### RQ1 — Retrieval Quality
Does the proposed memory architecture improve retrieval quality compared with simpler memory systems?

**Primary metric:** F1@5  
**Secondary metrics:** Precision@5, Recall@5, Hit Rate@5, MRR, NDCG@5

---

### RQ2 — Temporal Reasoning and Contradiction Resolution
Does explicit temporal memory improve the retrieval of facts that change over time?

**Primary metric:** Temporal Accuracy  
**Secondary metrics:** Historical Accuracy, Current-State Accuracy, Contradiction Resolution Accuracy, Temporal Recall@5

---

### RQ3 — Memory Consolidation
Does semantic consolidation reduce memory redundancy while preserving important information and retrieval quality?

**Primary metric:** Memory Reduction Ratio  
**Secondary metrics:** Information Retention, Duplicate Detection F1, Retrieval Quality Before vs. After Consolidation

---

### RQ4 — Multi-Hop Reasoning
Does graph-based retrieval improve questions requiring relationships between multiple memories?

**Primary metric:** Multi-Hop Retrieval Accuracy  
**Secondary metrics:** Multi-Hop Hit Rate, Graph Path Coverage, Answer Accuracy

---

### RQ5 — Efficiency and Scalability
What accuracy, latency, cost, and scalability trade-offs are introduced by the full memory architecture?

**Primary metrics:** P95 Retrieval Latency, Cost per Query  
**Secondary metrics:** P50, P99 latency, Token usage, Storage usage, Accuracy vs. Memory Size scaling curve

---

## 3. Evaluation Hypotheses

*These hypotheses are fixed before running any experiments. Testing additional hypotheses after observing results is permissible but must be labelled as exploratory.*

| ID | Hypothesis |
|----|-----------|
| H1 | The full memory architecture achieves higher retrieval quality (F1@5) than vector-only retrieval. |
| H2 | Explicit temporal reasoning improves Temporal Accuracy on temporal and contradiction queries compared with non-temporal baselines. |
| H3 | Semantic consolidation reduces the number of stored memories (Memory Reduction Ratio > 0.30) without significantly reducing F1@5 (degradation < 5%). |
| H4 | Graph-based retrieval improves Multi-Hop Retrieval Accuracy compared with vector retrieval alone. |
| H5 | The full architecture incurs higher P95 latency than vector-only retrieval, but provides measurable F1@5 improvements that justify the additional cost. |

---

## 4. Systems Under Evaluation

| ID | System | Long-Term Memory | Consolidation | Temporal Logic | Graph | Vector |
|----|--------|:---:|:---:|:---:|:---:|:---:|
| A  | No Long-Term Memory | ❌ | ❌ | ❌ | ❌ | ❌ |
| B  | Recency Window | Limited | ❌ | ❌ | ❌ | ❌ |
| C  | Vector RAG | ✅ | ❌ | ❌ | ❌ | ✅ |
| D  | Recency-Decay Vector RAG | ✅ | ❌ | Basic | ❌ | ✅ |
| E  | Semantic Memory | ✅ | ✅ | ❌ | ❌ | ✅ |
| P  | **Full Proposed System** | ✅ | ✅ | Explicit | ✅ | Hybrid |

---

## 5. Dataset Specification

### 5.1 Synthetic Dataset (Dataset A)

| Parameter | Value |
|-----------|-------|
| Number of users | 50 |
| Episodes per user | ≥ 20 |
| Total episodes | ≥ 1,000 |
| Words per episode | 150–300 |
| Dataset version | v1.0 |
| Random seed | 42 |

**Required content per user history:**
- ≥ 2 temporal fact changes (e.g. coping mechanism changes over time)
- ≥ 1 direct contradiction (outdated fact invalidated by newer fact)
- ≥ 1 multi-hop relationship chain (A → B → C requiring graph traversal)
- ≥ 2 no-answer queries (information not present in memory)

### 5.2 Human-Validated Subset (Dataset B)

| Parameter | Value |
|-----------|-------|
| Number of dialogues | ≥ 20 |
| Purpose | Sanity check only — not primary evaluation |
| Reviewer | At least one independent human reviewer |

Human validators check:
- Facts are realistic and natural
- Temporal changes resemble real conversational drift
- Contradictions match real-world update patterns
- Ground truth labels are correct
- Query difficulty is appropriate

### 5.3 Query Set

| Category | Count | Description |
|----------|-------|-------------|
| Factual | 50 | Simple direct fact retrieval |
| Temporal (CURRENT) | 50 | Requires retrieving current-state truth |
| Historical | 40 | Requires retrieving past truth at a specified time |
| Contradiction | 40 | Outdated fact must be rejected |
| Multi-Hop | 40 | Requires graph path traversal |
| No-Answer | 30 | No correct answer exists in memory |
| **Total** | **250** | |

---

## 6. Ground Truth Specification

### 6.1 Memory Temporal Fields

Every memory must have explicit validity bounds:
```json
{
  "memory_id": "memory_001",
  "content": "Walking helps reduce anxiety.",
  "valid_from": "2026-01-10",
  "valid_until": "2026-05-01"
}
```

### 6.2 Query Schema

Every query must specify:
- `query_type`: FACTUAL | TEMPORAL | HISTORICAL | CONTRADICTION | MULTI_HOP | NO_ANSWER
- `query_time`: ISO date string
- `query_temporal_mode`: CURRENT | HISTORICAL | ANY
- `ground_truth_memory_ids`: list of memory IDs that are correct at query_time
- `invalid_memory_ids`: list of memory IDs that are outdated and must NOT be returned
- `ground_truth_answer`: string or null (for NO_ANSWER)
- `expected_behavior`: ANSWER | ABSTAIN

---

## 7. Primary Metrics

| Metric | Formula | Query Types |
|--------|---------|-------------|
| F1@K | 2·P·R / (P+R) | All |
| Precision@K | Relevant Retrieved / K | All |
| Recall@K | Relevant Retrieved / Total Relevant | All |
| Hit Rate@K | 1 if any correct in top-K else 0 | All |
| MRR | Mean(1 / rank of first correct result) | All |
| NDCG@K | DCG@K / IDCG@K | All |
| Temporal Accuracy | Correct temporal answers / Total temporal queries | TEMPORAL, HISTORICAL |
| Contradiction Resolution Accuracy | Correctly rejected outdated / Total contradiction queries | CONTRADICTION |
| Multi-Hop Accuracy | Correct paths retrieved / Total multi-hop queries | MULTI_HOP |
| No-Answer Accuracy | Correct abstentions / Total no-answer queries | NO_ANSWER |
| Memory Reduction Ratio | (Raw − Final) / Raw | Consolidation |

---

## 8. Statistical Analysis Protocol

### 8.1 Tests

For all primary pairwise comparisons (Proposed System vs. each Baseline):
- **Test:** Wilcoxon signed-rank test (non-parametric, paired)
- **Why:** Retrieval metrics are not normally distributed and are query-paired

For consolidation before/after:
- **Test:** Paired Wilcoxon signed-rank on per-user retrieval quality

### 8.2 Multiple-Comparison Correction

**Method:** Benjamini–Hochberg False Discovery Rate (FDR)  
**Applied to:** All p-values from all primary RQ1–RQ4 pairwise comparisons  
**α level:** 0.05 (corrected)

### 8.3 Reporting Requirements

For every primary comparison, report:
- Raw p-value
- BH-corrected p-value
- Cohen's d effect size
- 95% confidence interval on the difference in primary metric

**Do NOT report only `p < 0.05`.**

---

## 9. LLM-as-Judge Validation Protocol

1. Randomly sample 100 generated answers (stratified by query type).
2. Two independent human raters score each answer on the 0–4 rubric.
3. Calculate inter-rater reliability (Cohen's Kappa). If κ < 0.60, adjudicate disagreements.
4. Run the LLM judge on the same 100 answers using the identical rubric and prompt.
5. Measure human–LLM agreement (Weighted Kappa + Pearson correlation).
6. **Gate:** If weighted κ (human–LLM) < 0.60, do NOT use the LLM judge as primary evaluator. Use human ratings instead and note this limitation.
7. If κ ≥ 0.60, use the LLM judge for the full 250-query benchmark.

**Required to report in paper:**
- Judge model name and version
- Judge prompt (exact text, verbatim)
- Temperature used (0)
- Human validation sample size
- Human–LLM Weighted Kappa and Pearson r

---

## 10. Reproducibility Requirements

Every experiment run must save the following to `benchmarks/results/<run_id>/`:
```
run_metadata.json:
  - experiment_id (UUID)
  - timestamp (ISO 8601)
  - git_commit_hash
  - dataset_version
  - protocol_version
  - config_snapshot (full copy of config.yaml)
  - random_seed
  - model_versions (embedding, chat, judge)
  - prompt_versions (retrieval prompt, judge prompt)

results/:
  - retrieval_metrics.json
  - temporal_metrics.json
  - consolidation_metrics.json
  - multi_hop_metrics.json
  - no_answer_metrics.json
  - latency_metrics.json
  - cost_metrics.json
  - ablation_results.json
  - scalability_results.json
  - statistical_analysis.json
  - error_analysis.json
```

---

## 11. Limitations to Acknowledge

1. **Dataset Bias:** Synthetic data may not fully represent natural human conversation.
2. **Small Human-Validated Subset:** 20–30 dialogues are a sanity check, not proof of broad generalization.
3. **Single-Domain Scope:** Evaluation focuses on longitudinal wellness context; results may not generalize to legal, financial, or enterprise memory systems.
4. **LLM Dependence:** Fact extraction and answer generation quality depends on the LLM.
5. **Graph Extraction Errors:** Incorrect entities or relationships propagate into retrieval.
6. **Infrastructure Complexity:** Full system requires more deployment components than vector-only retrieval.
7. **Cost Estimates:** API costs are estimates based on public rate cards and may differ from actual billing.

# Research Protocol and Hypotheses

## 1. Research Questions
*   **RQ1 (Retrieval Quality):** Does the proposed Longitudinal Memory Architecture maintain or improve general retrieval quality compared to baseline vector retrieval methods?
*   **RQ2 (Temporal & Contradiction):** How does explicit temporal validity filtering and time-decay reranking affect accuracy on contradictory or time-sensitive queries?
*   **RQ3 (Consolidation Efficiency):** Can memory consolidation significantly reduce the raw storage volume (Memory Reduction Ratio) without causing a statistically significant degradation in F1@5 retrieval quality?
*   **RQ4 (Multi-Hop Reasoning):** Does the inclusion of a knowledge graph layer improve retrieval recall on multi-hop logical queries compared to purely semantic vector search?
*   **RQ5 (Cost & Scalability):** Can the layered memory system maintain sub-100ms latency and minimal API overhead at scales of up to 100,000 episodic memories?

## 2. Hypotheses
*   **H1:** The Proposed System will demonstrate a statistically significant improvement in F1@5 over Baseline C (Vector RAG) on a blended dataset (p < 0.05).
*   **H2:** The Proposed System will exhibit a lower Hallucination Rate and higher Strict Accuracy on Contradiction queries compared to Baselines C and E.
*   **H3:** Semantic consolidation will achieve a Memory Reduction Ratio of >50% while F1@5 degradation remains within a 5% margin (p > 0.05 for non-inferiority).
*   **H4:** Multi-Hop Recall will be strictly greater for the Proposed System than for Baseline D (Recency-Decay Vector RAG).

## 3. Baselines
*   **Baseline A (No Persistent Memory):** RAG with zero retrieved context (only immediate conversation context).
*   **Baseline B (Recency Window):** RAG with the last 5 episodic turns (no semantic search).
*   **Baseline C (Vector RAG):** Standard vector search over episodic memory (no time decay or temporal bounds).
*   **Baseline D (Recency-Decay Vector RAG):** Vector search over episodic memory with exponential time-decay applied.
*   **Baseline E (Consolidated Semantic Vector Retrieval):** Vector search over consolidated semantic memories, but without temporal validity bounds or knowledge graph traversal.
*   **Proposed System (Full):** Layered short-term, episodic, and semantic memory with temporal validity, time decay, and knowledge graph traversal.

## 4. Primary Metrics
*   **Retrieval Metrics:** F1@5, Precision@5, Recall@5, MRR, Hit Rate@5
*   **Generation Metrics:** Strict Accuracy, Hallucination Rate, Abstention Accuracy, False Answer Rate.
*   **Efficiency Metrics:** Memory Reduction Ratio, P50/P95/P99 Latency (ms), Tokens/Cost per 100 queries.

## 5. Statistical Tests & Significance Threshold
*   **Significance Threshold (Alpha):** 0.05
*   **Multiple Comparisons:** Benjamini-Hochberg False Discovery Rate (FDR) correction will be applied across related hypothesis tests.
*   **Primary Test:** Wilcoxon Signed-Rank Test (paired, non-parametric) for query-level F1@5 and Accuracy comparisons.

## 6. Pre-Registration Freeze
These protocols and hypotheses are frozen prior to final benchmark dataset generation and execution.

# Experimental Environment

## Hardware
*   **CPU:** Local 8-Core Processor (x86_64)
*   **RAM:** 16 GB DDR4
*   **Storage:** NVMe SSD
*   **Operating System:** Windows 10/11 (or equivalent mock environment)

## Software & Infrastructure
*   **Python Version:** 3.11.x
*   **Database:** PostgreSQL 15+
*   **Vector Extension:** pgvector 0.5.0
*   **Cache/Message Broker:** Redis (local instance)
*   **Primary Database Driver:** SQLAlchemy (asyncpg)

## AI Models
*   **Embedding Model:** `text-embedding-3-small` (Dimensions: 1536)
*   **Chat Generation Model:** `gpt-4o-mini`
*   **LLM Judge Model:** `gpt-4o-mini` (or standard evaluation fallback)
*   **Model Temperature:** 0.0 (Deterministic evaluation for generation)

## Database Configuration
*   **Vector Index Type:** HNSW (Hierarchical Navigable Small World)
*   **HNSW Parameters:** `m=16, ef_construction=64`
*   **Similarity Metric:** Cosine Similarity

## Execution Configuration
*   **Batch Size:** Sequential (Batch=1) for accurate query latency measurements.
*   **Number of Benchmark Runs:** 1 (Deterministic validation loop).
*   **Warm-up Runs:** 5 (To populate PostgreSQL caching buffers before timing).
*   **Random Seed:** 42 (For any subset selection algorithms).

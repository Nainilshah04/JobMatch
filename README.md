# JobMatch: Production-Style AI Job-Matching Backend

> **Note on Data**: The job postings dataset in `data/jobs.csv` is **synthetic data** generated via `scripts/generate_jobs.py` (~850 realistic software engineering, AI/ML, data, and product roles) modeled after real-world startups and tech platforms (including YC-backed startups like Elip).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Client / Resume Payload         |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         FastAPI Search Layer          |
                      +---------------------------------------+
                               /                     \
                              v                       v
               +---------------------------+  +--------------------------+
               | Vector Retrieval (Cosine) |  | Keyword Retrieval (BM25/ |
               | pgvector (HNSW index)     |  | tsvector GIN index)      |
               +---------------------------+  +--------------------------+
                              \                       /
                               v                     v
                      +---------------------------------------+
                      |  Hybrid Fusion (RRF: k=60)            |
                      +---------------------------------------+
                                          | (Top 20)
                                          v
                      +---------------------------------------+
                      | Reranking (Cross-Encoder / LLM Rank)  |
                      +---------------------------------------+
                                          | (Top 5)
                                          v
                      +---------------------------------------+
                      | LLM Explanation (Structured Pydantic) |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Ranked Jobs + Skill Match/Gap Insights|
                      +---------------------------------------+
```

---

## Design Decisions

### Phase 1: Storage & Embedding
1. **pgvector with HNSW Index**:
   - *Decision*: Used HNSW (`m=16, ef_construction=64`) on the 384-dimensional `embedding` column instead of IVFFlat.
   - *Tradeoff*: HNSW uses slightly more memory but offers superior query latency, avoids IVFFlat clustering warm-up/reindexing overhead as data scales, and maintains high recall.

2. **Generated & Stored Weighted `tsvector`**:
   - *Decision*: Defined `tsv` as a `GENERATED ALWAYS AS (...) STORED` column with weights: Title (`A`), Skills (`B`), Company (`C`), Description (`D`).
   - *Tradeoff*: Increases disk storage marginally, but completely removes runtime `to_tsvector` CPU overhead during keyword searches and prioritizes role title & core skills over body text.

3. **Embedding Representation & Context Truncation**:
   - *Decision*: Constructed embedding input as: `Job Title: {title}. Company: {company}. Location: {location}. Key Skills: {skills}. Summary: {first_400_chars}`.
   - *Tradeoff*: `all-MiniLM-L6-v2` has a 256/512 token limit. Embedding entire multi-page descriptions dilutes key skill tokens; a structured representation keeps vectors semantically dense and discriminative.

4. **Normalized Cosine Embeddings**:
   - *Decision*: Embeddings are L2-normalized during inference (`normalize_embeddings=True`).
   - *Tradeoff*: Allows cosine distance (`<=>`) to map cleanly to inner product and guarantees bounded similarity metrics between 0.0 and 1.0.

### Phase 2: Retrieval & Hybrid Fusion
5. **Reciprocal Rank Fusion (RRF, $k=60$) vs. Linear Combination**:
   - *Decision*: Blended vector and keyword search using RRF with $k=60$ rather than linear weighted score fusion ($\alpha \cdot S_{vec} + (1-\alpha) \cdot S_{kw}$).
   - *Tradeoff*: Cosine similarity $[0, 1]$ and `ts_rank` $[0, >10]$ have fundamentally non-comparable distributions. Linear weighting requires continuous calibration as data grows. RRF operates purely on ordinal ranks, making it scale-invariant, robust to outliers, and parameter-light.

6. **Full-Text Parsing with `websearch_to_tsquery`**:
   - *Decision*: Handled user input using `websearch_to_tsquery` falling back to `plainto_tsquery`.
   - *Tradeoff*: Unlike raw `to_tsquery`, `websearch_to_tsquery` gracefully parses quotes, unclosed brackets, and logical operators without raising SQL runtime exceptions on user typos.

### Phase 3: Two-Stage Retrieve-Then-Rerank
7. **Two-Stage Architecture (Bi-Encoder + Cross-Encoder)**:
   - *Decision*: Retrieve the top 20 candidates via hybrid search and rerank them using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - *Tradeoff*: Bi-encoders allow sub-millisecond approximate nearest neighbors ($O(\log N)$ via HNSW) but compute vector representations independently without cross-token attention. A Cross-Encoder computes full token-to-token cross-attention for maximum ranking precision ($O(N)$), which is computationally feasible only on a filtered candidate pool (top 20).

8. **Cross-Encoder Logit Sigmoid Calibration**:
   - *Decision*: Applied sigmoid $\sigma(s) = \frac{1}{1 + e^{-s}}$ to convert raw cross-encoder logits into calibrated probabilities $[0, 1]$.
   - *Tradeoff*: Produces intuitive, normalized match confidence scores while preserving exact ranking order.

9. **Cross-Encoder vs. LLM Reranking Toggle**:
   - *Decision*: Provided a runtime flag (`cross_encoder` vs `llm`).
   - *Tradeoff*: Cross-encoder executes locally with ~30-50ms latency and zero API cost; LLM reranking provides deep contextual reasoning but incurs higher network latency and per-token API cost.

### Phase 4: LLM Structured Output & Skill Gap Analysis
10. **Pydantic Schema Validation with Automated Feedback Retries**:
    - *Decision*: Bound LLM outputs to the `JobExplanation` schema (`match_score`, `matched_skills`, `missing_skills`, `reason`) with up to 2 retries on `ValidationError`, passing the parser error directly back to the model.
    - *Tradeoff*: Retrying on invalid output prevents corrupted JSON crashes at the expense of extra round-trip latency for malformed responses.

11. **Deterministic Heuristic Fallback**:
    - *Decision*: Implemented keyword-token skill matching fallback when API keys are unconfigured or provider endpoints time out.
    - *Tradeoff*: Guarantees 100% system uptime and offline testability without blocking backend services during upstream LLM outages.

### Phase 5: API Layer & Bespoke SaaS Frontend
12. **Unified Search & Resume Interface (`POST /search`)**:
    - *Decision*: Merged short query and multiline unstructured resume text into a single endpoint with automatic truncation to fit embedding context limits.
    - *Tradeoff*: Simplifies client integration and avoids maintaining divergent search paths for candidate bios vs. title lookups.

13. **Modern Pure-Vanilla SaaS Frontend**:
    - *Decision*: Built a bespoke, high-performance UI in vanilla HTML5/CSS3/ES6 served directly by FastAPI without React or build tooling.
    - *Tradeoff*: Zero bundle compilation overhead and sub-millisecond page load, with a clean Linear/Vercel-inspired dark theme and live telemetry.

---

## Quickstart

### 1. Prerequisites
- Python 3.11+
- PostgreSQL database with `pgvector` enabled (e.g. Neon serverless Postgres)

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env with your DATABASE_URL (e.g., from Neon)
```

### 3. Generate Data and Populate Database
```bash
# 1. Generate realistic synthetic job records (~850 records)
python scripts/generate_jobs.py

# 2. Ingest into PostgreSQL and generate vector embeddings
python scripts/load_jobs.py --embed
```

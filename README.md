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

## Design Decisions (Phase 1)

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

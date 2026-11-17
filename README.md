# JobMatch: Production-Style AI Job-Matching & Hybrid Retrieval Backend

[![Render Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=flat-square&logo=render&logoColor=white)](https://jobmatch-qcqc.onrender.com)
[![CI / Test Suite](https://img.shields.io/badge/pytest-16%20passed-success?style=flat-square)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW%20Cosine-336791?style=flat-square)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

> 🚀 **Live Production Demo**: [https://jobmatch-qcqc.onrender.com](https://jobmatch-qcqc.onrender.com)  
> Interactive Unstop-style web portal with semantic/hybrid search, PDF resume drag-and-drop parsing, persistent wishlist, and AI candidate-job fit evaluation.
>
> 📊 **Note on Data**: The dataset in `data/jobs.csv` is **synthetic data** generated via `scripts/generate_jobs.py` (~850 realistic software engineering, AI/ML, data platform, DevOps, and product management roles), modeled after real-world tech startups and recruitment platforms.

---

## Architecture Overview

```
                          [ Candidate Query / Resume PDF / Text ]
                                             |
                                             v
                          +-------------------------------------+
                          |         FastAPI Search Layer        |
                          |            (POST /search)           |
                          +-------------------------------------+
                                     /               \
                                    /                 \
                                   v                   v
             +---------------------------+   +--------------------------+
             | Vector Retrieval (Cosine) |   | Keyword Retrieval (FTS)  |
             | all-MiniLM-L6-v2 (384d)   |   | PostgreSQL tsvector      |
             | pgvector (HNSW Index)     |   | Weighted GIN Index       |
             +---------------------------+   +--------------------------+
                                   \                   /
                                    \                 /
                                     v               v
                          +-------------------------------------+
                          |   Reciprocal Rank Fusion (k=60)     |
                          |   Unites Lexical + Semantic Ranks   |
                          +-------------------------------------+
                                             |  (Top 20 Candidates)
                                             v
                          +-------------------------------------+
                          |       Two-Stage Reranker            |
                          | ms-marco-MiniLM-L-6-v2 / LLM Score  |
                          |   (Sigmoid Probability Calibration) |
                          +-------------------------------------+
                                             |  (Top 5 - 10 Roles)
                                             v
                          +-------------------------------------+
                          |       LLM Fit & Skill-Gap Engine    |
                          | Structured Pydantic JSON Output:    |
                          | - Match Score (0-100)               |
                          | - Matched Skills & Missing Skills   |
                          | - Constructive Recruiter Rationale  |
                          +-------------------------------------+
                                             |
                                             v
                          [ Unstop-Style Clean Career Portal UI ]
```

---

## Why Hybrid Search Beats Vector-Only Search

In search and recommendation systems, a central architectural question is:
> *"Why not just embed the query and use pure cosine vector search across all jobs?"*

### The Tradeoff Analysis:

1. **The Out-of-Vocabulary & Acronym Problem**:
   - Modern bi-encoder embeddings (`all-MiniLM-L6-v2`) project words into high-dimensional semantic spaces. While they understand general synonymy (e.g. mapping *"software engineer"* to *"developer"*), they often suffer from **lexical drift** on exact technical acronyms, library names, or version identifiers (`FastAPI`, `dbt`, `vLLM`, `CUDA`, `gRPC`, `pgvector`).
   - Pure vector search might return a generic Python data analysis job for a *"dbt analytics engineer"* query simply because both discuss data pipelines in Python.
   - **PostgreSQL Full-Text Search (tsvector)** provides exact lexical precision for specific required tools.

2. **Why Reciprocal Rank Fusion (RRF $k=60$) over Linear Score Combination**:
   - Cosine similarity scores are bounded $[0, 1]$, whereas full-text `ts_rank` scores are unbounded $[0, >10]$.
   - Direct linear combination ($\alpha \cdot S_{vec} + (1-\alpha) \cdot S_{kw}$) requires constant calibration; changes in query length or corpus size skew linear weights.
   - **RRF** operates strictly on ordinal rank positions:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{keyword}\}} \frac{1}{60 + \text{rank}_m(d)}$$
     This guarantees scale-invariance, outlier dampening, and high recall without tuning arbitrary weight coefficients.

---

## Design Decisions & Tradeoffs

### Phase 1: Database Storage & Embeddings
1. **pgvector with HNSW Index**:
   - *Decision*: Used HNSW (`m=16, ef_construction=64`) on the 384-dimensional `embedding` column instead of IVFFlat.
   - *Tradeoff*: HNSW uses slightly more RAM but offers $O(\log N)$ query latency and avoids IVFFlat clustering warm-up/reindexing overhead as data scales.
2. **Generated & Stored Weighted `tsvector`**:
   - *Decision*: Defined `tsv` as a `GENERATED ALWAYS AS (...) STORED` column with weights: Title (`A`), Skills (`B`), Company (`C`), Description (`D`).
   - *Tradeoff*: Increases disk footprint marginally, but eliminates runtime `to_tsvector` CPU overhead during search and prioritizes job titles over long descriptions.
3. **Embedding Representation & Context Truncation**:
   - *Decision*: Constructed embedding input as: `Job Title: {title}. Company: {company}. Location: {location}. Key Skills: {skills}. Summary: {first_400_chars}`.
   - *Tradeoff*: `all-MiniLM-L6-v2` has a 256/512 token limit. Embedding entire multi-page descriptions dilutes key skill tokens; a structured representation keeps vectors semantically dense and discriminative.

### Phase 2 & 3: Retrieval & Two-Stage Reranking
4. **Two-Stage Retrieve-Then-Rerank Architecture**:
   - *Decision*: Retrieve the top 20 candidates via hybrid search and rerank them using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - *Tradeoff*: Bi-encoders allow sub-millisecond approximate nearest neighbors ($O(\log N)$ via HNSW) but compute vector representations independently without cross-token attention. A Cross-Encoder computes full token-to-token cross-attention for maximum ranking precision ($O(N)$), which is computationally feasible only on a filtered candidate pool (top 20).
5. **Cross-Encoder Logit Sigmoid Calibration**:
   - *Decision*: Applied sigmoid $\sigma(s) = \frac{1}{1 + e^{-s}}$ to convert raw cross-encoder logits into calibrated probabilities $[0, 1]$.
   - *Tradeoff*: Produces intuitive, normalized match confidence scores while preserving exact ranking order.

### Phase 4 & 5: LLM Structured Output & Resilience
6. **Pydantic Schema Validation with Automated Feedback Retries**:
   - *Decision*: Bound LLM outputs to the `JobExplanation` schema (`match_score`, `matched_skills`, `missing_skills`, `reason`) with up to 2 retries on `ValidationError`, passing the parser error directly back to the model.
   - *Tradeoff*: Retrying on invalid output prevents corrupted JSON crashes at the expense of extra round-trip latency for malformed responses.
7. **Resilient Local Fallback Engine**:
   - *Decision*: Added an in-memory vector & keyword fallback engine over `data/jobs.csv` when PostgreSQL is offline or unconfigured.
   - *Tradeoff*: Guarantees 100% system availability for local demos and offline testing without blocking backend endpoints.
8. **PDF Resume Text Extraction**:
   - *Decision*: Implemented `POST /upload-resume` using lightweight `pypdf` with drag-and-drop UI integration.
   - *Tradeoff*: Zero external OCR service dependencies; fast text extraction from real candidate PDFs.

---

## Quickstart & Local Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Nainilshah04/JobMatch.git
cd JobMatch

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: For cloud PostgreSQL, create a free database at [neon.tech](https://neon.tech) and set `DATABASE_URL`. If left default, the app automatically runs in resilient local demo mode over the 850 synthetic jobs).*

### 3. Run Ingestion (Optional for Cloud Postgres)
```bash
# Generate synthetic dataset
python scripts/generate_jobs.py

# Ingest and precompute embeddings
python scripts/load_jobs.py --embed
```

### 4. Launch Application
```bash
python -m uvicorn app.main:app --reload --port 8000
```
- **Web Application**: Open [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running with Docker & Docker Compose

### Option A: Docker Compose (App + PostgreSQL with pgvector)
```bash
docker-compose up --build
```
Access the application at `http://localhost:8000`.

### Option B: Standalone Docker Image
```bash
# Build the production image (pre-caches ML model weights)
docker build -t jobmatch .

# Run the container
docker run -p 8000:8000 jobmatch
```

---

## Live Cloud Deployment & Render Architecture

JobMatch is deployed live in production on Render:

> 🌐 **Live URL**: [https://jobmatch-qcqc.onrender.com](https://jobmatch-qcqc.onrender.com)  
> 📖 **Interactive OpenAPI Specs**: [https://jobmatch-qcqc.onrender.com/docs](https://jobmatch-qcqc.onrender.com/docs)  
> 🩺 **Healthcheck Endpoint**: [https://jobmatch-qcqc.onrender.com/health](https://jobmatch-qcqc.onrender.com/health)

### 512MB RAM Optimization Strategies for Free-Tier Cloud Containers
Deploying bi-encoder & cross-encoder transformer architectures within Render's strict **512MB RAM ceiling** required critical engineering optimizations:
1. **CPU-Only PyTorch Wheel**: Standard PyTorch bundles CUDA binaries exceeding 1.5GB disk and ~350MB base RAM. We install the lightweight CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`), cutting base image size from 2.2GB to under 700MB.
2. **Pre-Encoded Offline Embeddings (`data/job_embeddings.npy`)**: Rather than encoding all 850 jobs on container startup (which creates a massive peak memory spike), embeddings are precomputed offline and loaded directly via memory-mapped NumPy in <2MB RAM.
3. **Single-Threaded Execution**: PyTorch by default allocates worker threads across all container CPUs, rapidly exhausting free-tier memory. We enforce `OMP_NUM_THREADS=1` and `MALLOC_ARENA_MAX=2` in `Dockerfile` and `render.yaml`.
4. **Resilient Dual-Engine Backend**: If cloud PostgreSQL (Neon) is not configured, the service seamlessly falls back to fast in-memory cosine & BM25-style keyword search over the local dataset without failing container healthchecks.

### Deploying Your Own Instance on Render
1. Fork or clone this repository: `https://github.com/Nainilshah04/JobMatch`.
2. In [Render Dashboard](https://dashboard.render.com), click **New + > Blueprint** and select your repository (it automatically picks up `render.yaml`).
3. (Optional) Set `DATABASE_URL` (e.g. from [neon.tech](https://neon.tech)) and `LLM_API_KEY` in Environment settings.
4. Render builds the Docker container and exposes the service automatically on port `10000`.

---

## System Limitations & Future Extensions

1. **Context Length Truncation**:
   - `all-MiniLM-L6-v2` truncates after 256/512 tokens. While structured summarization mitigates this, transitioning to late-interaction models (e.g., **ColBERTv2**) would preserve multi-vector token granularities.
2. **Continuous Real-Time Ingestion**:
   - Currently, ingestion runs via batch script. A production iteration would use event-driven change-data-capture (CDC via Debezium / Kafka) to embed new job postings in real time as companies publish them.
3. **Candidate Personalization Profile**:
   - Extending beyond static queries to maintain candidate preference embeddings (salary expectations, career trajectory, remote preferences).
4. **Heuristic Evaluation Benchmark Labels**:
   - The 25 evaluation queries in `eval/queries.json` currently use heuristic/draft relevance labels derived via skill & keyword matching across `data/jobs.csv`. While effective for relative pipeline comparisons, production ranking evaluation requires human recruiter-annotated judgments.


## Evaluation Benchmark & Retrieval Metrics

### Information Retrieval Benchmark Results (25 Queries)

> Evaluated over `25` diverse queries using Recall@10 and MRR (Mean Reciprocal Rank). Note: relevance labels are heuristic/draft.

| Retrieval Pipeline | Recall@10 | MRR | Avg Latency (ms) | Pipeline Description |
| :--- | :---: | :---: | :---: | :--- |
| **Vector-only** | `5.0%` | `0.1591` | `382.8ms` | Dense semantic retrieval using all-MiniLM-L6-v2 embeddings (cosine similarity) |
| **Keyword-only** | `5.5%` | `0.2213` | `240.7ms` | Lexical search using PostgreSQL tsvector / token overlap |
| **Hybrid (RRF k=60)** | `7.0%` | `0.1790` | `247.1ms` | Reciprocal Rank Fusion uniting vector and keyword rank positions (k=60) |
| **Hybrid + Rerank** | `6.5%` | `0.1453` | `5303.1ms` | Two-stage pipeline: top 20 hybrid candidates reranked by cross-encoder |

#### Programmatic Benchmark Insights:
- **Best Recall@10**: Hybrid (RRF k=60) (7.0%)
- **Best MRR**: Keyword-only (0.2213)
- **Reranker Impact vs Plain Hybrid**: In this benchmark, cross-encoder reranking decreased MRR (0.1790 -> 0.1453) and decreased Recall@10 (7.0% -> 6.5%).

### Retrieval Analysis & Observations
- **Hybrid Retrieval Impact**: In this benchmark, Hybrid (RRF k=60) marginally improved Recall@10 over single-modality retrievers (7.0% vs 5.0% for vector-only and 5.5% for keyword-only), indicating that combining semantic vector similarity with lexical keyword matching helps capture jobs that either modality alone misses.
- **Cross-Encoder Reranker Performance**: In this benchmark, cross-encoder reranking did not improve MRR (0.1453 vs 0.1790) and marginally decreased Recall@10 (6.5% vs 7.0%) compared to plain hybrid. A plausible reason is domain divergence: `ms-marco-MiniLM-L-6-v2` is pretrained on general web-search question answering (MS MARCO passages), which differs substantially from recruiter search where specific technical skill tokens, seniority indicators, and title alignments dominate candidate relevance. Generalist QA cross-encoders may downweight exact skill acronyms in favor of conversational narrative text.
- **Benchmark Scale & Statistical Caveats**: This evaluation was conducted on 25 synthetic queries with draft/heuristic labels. Small metric variances (e.g. ±1-2%) are not statistically significant and reflect prototype evaluation rather than large-scale production ranking.


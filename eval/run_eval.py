"""Evaluation harness for JobMatch retrieval pipelines.

Computes Recall@10, Precision@10, nDCG@10, and Mean Reciprocal Rank (MRR) across four retrieval pipelines:
1. Vector-only (Cosine similarity via all-MiniLM-L6-v2)
2. Keyword-only (PostgreSQL full-text search ts_rank or local token overlap)
3. Hybrid (Reciprocal Rank Fusion, k=60)
4. Hybrid + Rerank (Hybrid top 20 + Cross-Encoder reranking)

Outputs a Markdown benchmark results table and updates README.md.
"""

import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import JobSearchResult
from app.rerank import rerank_candidates
from app.retrieval import (
    get_active_backend,
    search_hybrid,
    search_keyword,
    search_vector,
    _load_local_jobs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    retrieved_ids: List[int],
    ground_truth_ids: Set[int],
    top_k: int = 10,
) -> Dict[str, float]:
    """Compute Recall@K, Precision@K, nDCG@K, and Reciprocal Rank (RR) for a single query.

    Args:
        retrieved_ids: Ranked list of retrieved job IDs.
        ground_truth_ids: Set of relevant/ground-truth job IDs.
        top_k: Cutoff rank for evaluation (default 10).

    Returns:
        Dict with keys: 'recall', 'precision', 'ndcg', 'mrr'

    Note on Recall@K when |ground_truth| > K:
        When a query has more relevant documents than the cutoff K (e.g. 15 relevant docs for K=10),
        Recall@K is mathematically capped at K / |ground_truth| (e.g. 10/15 = 66.7%).
        Even a hypothetical perfect retrieval pipeline that returns 10 relevant documents in top 10
        cannot achieve 100% recall under this cutoff without increasing K.
    """
    if not ground_truth_ids:
        return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0, "mrr": 0.0}

    retrieved_k = retrieved_ids[:top_k]
    matched_set = set(retrieved_k) & ground_truth_ids

    # 1. Recall@K: fraction of total ground-truth items found within top_k cutoff
    recall_at_k = len(matched_set) / len(ground_truth_ids)

    # 2. Precision@K: fraction of top_k retrieved items that are relevant
    precision_at_k = len(matched_set) / top_k if top_k > 0 else 0.0

    # 3. Reciprocal Rank (RR): reciprocal rank of first relevant item
    reciprocal_rank = 0.0
    for rank, job_id in enumerate(retrieved_k, start=1):
        if job_id in ground_truth_ids:
            reciprocal_rank = 1.0 / rank
            break

    # 4. nDCG@K: Normalized Discounted Cumulative Gain with binary relevance
    # DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    dcg = 0.0
    for rank, job_id in enumerate(retrieved_k, start=1):
        if job_id in ground_truth_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG@K = sum_{i=1}^{min(K, |ground_truth|)} 1 / log2(i + 1)
    ideal_count = min(top_k, len(ground_truth_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    return {
        "recall": recall_at_k,
        "precision": precision_at_k,
        "ndcg": ndcg,
        "mrr": reciprocal_rank,
    }


def warmup_pipelines() -> None:
    """Execute 2 throwaway queries per pipeline to eliminate cold-start latency bias."""
    logger.info("Executing model and cache warm-up (2 throwaway queries per pipeline)...")
    warmup_queries = ["Python Software Engineer", "Data Scientist Machine Learning"]
    for wq in warmup_queries:
        _ = search_vector(wq, top_k=2)
        _ = search_keyword(wq, top_k=2)
        pool = search_hybrid(wq, top_k=5)
        _ = rerank_candidates(wq, pool, reranker_type="cross_encoder")[:2]
    logger.info("Warm-up complete. Starting timed benchmark evaluation...")


def run_evaluation(queries_path: str, top_k: int = 10) -> Tuple[Dict[str, Dict[str, float]], int, str]:
    """Run full evaluation suite across all 4 retrieval modes."""
    with open(queries_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    total_queries = len(queries)
    backend = get_active_backend()
    logger.info(f"Loaded {total_queries} benchmark queries from {queries_path}. Active backend: {backend}")

    # Warm up models and caches before timing
    warmup_pipelines()

    modes = ["Vector-only", "Keyword-only", "Hybrid (RRF k=60)", "Hybrid + Rerank"]
    results: Dict[str, Dict[str, float]] = {
        m: {"recall": 0.0, "precision": 0.0, "ndcg": 0.0, "mrr": 0.0, "latency_ms": 0.0}
        for m in modes
    }

    for q in queries:
        query_text = q["query"]
        relevant_ids = set(q["relevant_job_ids"])

        # 1. Vector-only
        t0 = time.time()
        vec_jobs = search_vector(query_text, top_k=top_k)
        t_vec = (time.time() - t0) * 1000.0
        m_vec = compute_metrics([j.id for j in vec_jobs], relevant_ids, top_k=top_k)
        for k in ["recall", "precision", "ndcg", "mrr"]:
            results["Vector-only"][k] += m_vec[k]
        results["Vector-only"]["latency_ms"] += t_vec

        # 2. Keyword-only
        t0 = time.time()
        kw_jobs = search_keyword(query_text, top_k=top_k)
        t_kw = (time.time() - t0) * 1000.0
        m_kw = compute_metrics([j.id for j in kw_jobs], relevant_ids, top_k=top_k)
        for k in ["recall", "precision", "ndcg", "mrr"]:
            results["Keyword-only"][k] += m_kw[k]
        results["Keyword-only"]["latency_ms"] += t_kw

        # 3. Hybrid (RRF)
        t0 = time.time()
        hyb_jobs = search_hybrid(query_text, top_k=top_k)
        t_hyb = (time.time() - t0) * 1000.0
        m_hyb = compute_metrics([j.id for j in hyb_jobs], relevant_ids, top_k=top_k)
        for k in ["recall", "precision", "ndcg", "mrr"]:
            results["Hybrid (RRF k=60)"][k] += m_hyb[k]
        results["Hybrid (RRF k=60)"]["latency_ms"] += t_hyb

        # 4. Hybrid + Rerank
        t0 = time.time()
        candidate_pool = search_hybrid(query_text, top_k=20)
        reranked_jobs = rerank_candidates(query_text, candidate_pool, reranker_type="cross_encoder")[:top_k]
        t_rerank = (time.time() - t0) * 1000.0
        m_rerank = compute_metrics([j.id for j in reranked_jobs], relevant_ids, top_k=top_k)
        for k in ["recall", "precision", "ndcg", "mrr"]:
            results["Hybrid + Rerank"][k] += m_rerank[k]
        results["Hybrid + Rerank"]["latency_ms"] += t_rerank

    # Average metrics
    for m in modes:
        results[m]["recall"] = round(results[m]["recall"] / total_queries, 4)
        results[m]["precision"] = round(results[m]["precision"] / total_queries, 4)
        results[m]["ndcg"] = round(results[m]["ndcg"] / total_queries, 4)
        results[m]["mrr"] = round(results[m]["mrr"] / total_queries, 4)
        results[m]["latency_ms"] = round(results[m]["latency_ms"] / total_queries, 2)

    return results, total_queries, backend


def format_markdown_table(
    results: Dict[str, Dict[str, float]],
    num_queries: int,
    total_jobs: int,
    backend: str,
) -> str:
    """Format evaluation benchmark results as a Markdown table with programmatically derived insights."""
    backend_display = (
        f"`{backend}` (PostgreSQL with pgvector & tsvector)"
        if backend == "postgres+pgvector"
        else f"`{backend}` (In-memory cosine & token-overlap fallback)"
    )

    table = [
        f"### Information Retrieval Benchmark Results ({num_queries} Queries over {total_jobs} Jobs)",
        "",
        f"> **Evaluation Environment**: Active Backend: {backend_display}. Evaluated over `{num_queries}` diverse queries across `{total_jobs}` jobs. Note: relevance labels are heuristic/draft.",
        "",
        "| Retrieval Pipeline | Recall@10 | Precision@10 | nDCG@10 | MRR | Avg Latency (ms) | Pipeline Description |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    notes = {
        "Vector-only": "Dense semantic retrieval using all-MiniLM-L6-v2 embeddings (cosine similarity)",
        "Keyword-only": "Lexical search using PostgreSQL tsvector / token overlap",
        "Hybrid (RRF k=60)": "Reciprocal Rank Fusion uniting vector and keyword rank positions (k=60)",
        "Hybrid + Rerank": "Two-stage pipeline: top 20 hybrid candidates reranked by cross-encoder",
    }

    for mode, metrics in results.items():
        recall_pct = f"{metrics['recall'] * 100:.1f}%"
        prec_pct = f"{metrics['precision'] * 100:.1f}%"
        ndcg_val = f"{metrics['ndcg']:.4f}"
        mrr_val = f"{metrics['mrr']:.4f}"
        latency = f"{metrics['latency_ms']:.1f}ms"
        note = notes.get(mode, "")
        table.append(f"| **{mode}** | `{recall_pct}` | `{prec_pct}` | `{ndcg_val}` | `{mrr_val}` | `{latency}` | {note} |")

    table.append("")
    table.append("#### Programmatic Benchmark Insights:")

    best_recall = max(results.items(), key=lambda x: x[1].get("recall", 0.0))
    best_prec = max(results.items(), key=lambda x: x[1].get("precision", 0.0))
    best_ndcg = max(results.items(), key=lambda x: x[1].get("ndcg", 0.0))
    best_mrr = max(results.items(), key=lambda x: x[1].get("mrr", 0.0))

    table.append(f"- **Best Recall@10**: {best_recall[0]} ({best_recall[1]['recall'] * 100:.1f}%)")
    table.append(f"- **Best Precision@10**: {best_prec[0]} ({best_prec[1]['precision'] * 100:.1f}%)")
    table.append(f"- **Best nDCG@10**: {best_ndcg[0]} ({best_ndcg[1]['ndcg']:.4f})")
    table.append(f"- **Best MRR**: {best_mrr[0]} ({best_mrr[1]['mrr']:.4f})")

    if "Hybrid + Rerank" in results and "Hybrid (RRF k=60)" in results:
        hyb = results["Hybrid (RRF k=60)"]
        rr = results["Hybrid + Rerank"]
        mrr_diff = rr.get("mrr", 0.0) - hyb.get("mrr", 0.0)
        rec_diff = rr.get("recall", 0.0) - hyb.get("recall", 0.0)

        mrr_status = "improved" if mrr_diff > 0.001 else ("decreased" if mrr_diff < -0.001 else "was unchanged")
        rec_status = "improved" if rec_diff > 0.001 else ("decreased" if rec_diff < -0.001 else "was unchanged")
        table.append(
            f"- **Reranker Impact vs Plain Hybrid**: In this benchmark, cross-encoder reranking {mrr_status} MRR "
            f"({hyb.get('mrr', 0.0):.4f} -> {rr.get('mrr', 0.0):.4f}) and {rec_status} Recall@10 "
            f"({hyb.get('recall', 0.0)*100:.1f}% -> {rr.get('recall', 0.0)*100:.1f}%)."
        )

    table.append("")
    table.append("### Retrieval Analysis & Observations")
    table.append(
        f"- **Hybrid Retrieval Impact**: In this benchmark, Hybrid (RRF k=60) achieved {results['Hybrid (RRF k=60)']['recall']*100:.1f}% Recall@10 "
        f"compared to {results['Vector-only']['recall']*100:.1f}% for Vector-only and {results['Keyword-only']['recall']*100:.1f}% for Keyword-only. "
        "Reciprocal Rank Fusion successfully aggregates candidates across both modalities, mitigating the individual blindspots of pure vector or lexical search."
    )
    table.append(
        f"- **Cross-Encoder Reranker Performance**: In this benchmark, cross-encoder reranking yielded MRR {results['Hybrid + Rerank']['mrr']:.4f} "
        f"versus plain Hybrid MRR {results['Hybrid (RRF k=60)']['mrr']:.4f}, and Recall@10 {results['Hybrid + Rerank']['recall']*100:.1f}% "
        f"versus {results['Hybrid (RRF k=60)']['recall']*100:.1f}%. A plausible reason is domain divergence: `ms-marco-MiniLM-L-6-v2` is pretrained on "
        "general web-search question answering passages, which differs from structured recruiter queries that require strict alignment on specific technical skills, titles, and seniority levels."
    )
    table.append(
        f"- **Benchmark Scale & Statistical Caveats**: This evaluation was executed over {num_queries} queries with draft heuristic relevance labels on backend `{backend}`. "
        "Small metric variances (e.g. ±1-2%) are not statistically definitive and serve as an architectural reference rather than production validation."
    )
    table.append("")
    return "\n".join(table)


def update_readme_benchmark(markdown_table: str, readme_path: str) -> None:
    """Insert or update the benchmark results in README.md."""
    if not os.path.exists(readme_path):
        return

    with open(readme_path, mode="r", encoding="utf-8") as f:
        content = f.read()

    section_header = "## Evaluation Benchmark & Retrieval Metrics"
    new_section = f"{section_header}\n\n{markdown_table}\n"

    if section_header in content:
        # Replace existing section up to next ## header
        parts = content.split(section_header)
        after_header = parts[1]
        next_section_idx = after_header.find("\n## ")
        if next_section_idx != -1:
            updated_content = parts[0] + new_section + after_header[next_section_idx:]
        else:
            updated_content = parts[0] + new_section
    else:
        updated_content = content + "\n\n" + new_section

    with open(readme_path, mode="w", encoding="utf-8") as f:
        f.write(updated_content)

    logger.info(f"Updated benchmark results in {readme_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    queries_file = os.path.join(base_dir, "eval", "queries.json")
    readme_file = os.path.join(base_dir, "README.md")

    # Load dataset size dynamically from active dataset
    jobs_cache, _ = _load_local_jobs()
    total_jobs = len(jobs_cache)

    print("\nStarting JobMatch Retrieval Evaluation Benchmark...")
    eval_results, total_queries, backend = run_evaluation(queries_file, top_k=10)

    md_table = format_markdown_table(
        results=eval_results,
        num_queries=total_queries,
        total_jobs=total_jobs,
        backend=backend,
    )
    print("\n" + md_table)

    update_readme_benchmark(md_table, readme_file)
    print("Evaluation benchmark completed successfully.")

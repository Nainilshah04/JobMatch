"""CLI test script to verify Phase 2 retrieval modes.

Runs a sample query across Vector-only, Keyword-only, and Hybrid (RRF) search
and prints top ranked results for comparison.
"""

import argparse
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import search_vector, search_keyword, search_hybrid


def display_results(title: str, results) -> None:
    """Format and print search results cleanly."""
    print(f"\n{'='*70}")
    print(f"  {title.upper()} (Found: {len(results)})")
    print(f"{'='*70}")
    if not results:
        print("  No matching jobs found.")
        return

    for rank, job in enumerate(results, start=1):
        rem_tag = "[Remote]" if job.remote else f"[{job.location}]"
        print(
            f"  {rank:2d}. [ID {job.id:3d}] {job.title:<32} | {job.company:<16} | "
            f"Score: {job.score:.5f} | {rem_tag}"
        )
        print(f"      Skills: {job.skills}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test JobMatch retrieval modes.")
    parser.add_argument(
        "--query",
        type=str,
        default="Python Backend Engineer with FastAPI and Docker",
        help="Search query text or candidate resume snippet",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to display")
    parser.add_argument("--location", type=str, default=None, help="Optional location filter")
    parser.add_argument("--remote", action="store_true", help="Filter for remote jobs only")
    args = parser.parse_args()

    remote_filter = True if args.remote else None

    print(f"\nRunning retrieval test with query: '{args.query}'")
    if args.location:
        print(f"Location filter: {args.location}")
    if remote_filter:
        print("Remote filter: True")

    try:
        # 1. Vector-only Search
        vec_results = search_vector(
            query_text=args.query,
            top_k=args.top_k,
            location=args.location,
            remote=remote_filter,
        )
        display_results("1. Vector-Only Search (Cosine Similarity)", vec_results)

        # 2. Keyword-only Search
        kw_results = search_keyword(
            query_text=args.query,
            top_k=args.top_k,
            location=args.location,
            remote=remote_filter,
        )
        display_results("2. Keyword-Only Search (Postgres FTS ts_rank)", kw_results)

        # 3. Hybrid Search
        hybrid_results = search_hybrid(
            query_text=args.query,
            top_k=args.top_k,
            location=args.location,
            remote=remote_filter,
        )
        display_results("3. Hybrid Search (Reciprocal Rank Fusion k=60)", hybrid_results)

    except Exception as e:
        print(f"\n[Error executing retrieval]: {e}")
        print("Ensure PostgreSQL is running and DATABASE_URL is correctly set in .env.")
        sys.exit(1)


if __name__ == "__main__":
    main()

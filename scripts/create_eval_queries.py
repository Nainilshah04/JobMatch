"""Generate initial draft benchmark queries with ground-truth relevant job IDs."""

import csv
import json
import os

QUERIES_SPEC = [
    ("Senior Python Backend Engineer FastAPI", ["Python", "FastAPI", "Docker", "PostgreSQL"]),
    ("Go Distributed Systems Microservices gRPC", ["Go", "gRPC", "Kubernetes", "Kafka"]),
    ("Django Backend Developer Celery AWS", ["Python", "Django", "Celery", "AWS"]),
    ("Java Spring Boot Microservices Elasticsearch", ["Java", "Spring Boot", "Kafka", "Elasticsearch"]),
    ("Node.js TypeScript API Developer", ["Node.js", "TypeScript", "Express", "MongoDB"]),
    ("LLM Application Engineer LangChain RAG", ["Python", "LLMs", "LangChain", "OpenAI API"]),
    ("PyTorch Computer Vision Deep Learning", ["Python", "PyTorch", "Transformers", "CUDA"]),
    ("NLP Research Engineer Embeddings Semantic Search", ["Python", "NLP", "Embeddings", "Semantic Search"]),
    ("MLOps Engineer Kubeflow Triton AWS SageMaker", ["Python", "MLflow", "Kubeflow", "Kubernetes"]),
    ("Scikit-Learn Machine Learning Data Scientist", ["Python", "Scikit-Learn", "XGBoost", "Pandas"]),
    ("Snowflake dbt Data Platform Engineer", ["SQL", "dbt", "Snowflake", "BigQuery"]),
    ("Apache Spark Big Data Engineer Airflow", ["Python", "SQL", "Apache Spark", "Airflow"]),
    ("Streaming Data Engineer Kafka Apache Flink", ["Python", "Kafka", "Apache Flink", "AWS Redshift"]),
    ("Databricks Delta Lake ETL Pipeline Developer", ["Python", "SQL", "Databricks", "Delta Lake"]),
    ("ClickHouse Presto Data Warehousing Engineer", ["SQL", "Python", "ClickHouse", "Presto"]),
    ("Kubernetes Terraform Cloud Infrastructure SRE", ["Kubernetes", "Docker", "Terraform", "AWS"]),
    ("Prometheus Grafana Observability Engineer Linux", ["Prometheus", "Grafana", "Linux", "Kubernetes"]),
    ("GCP DevOps ArgoCD Helm Platform Engineer", ["GCP", "Kubernetes", "Helm", "ArgoCD"]),
    ("AWS Security Cloud Architect Compliance", ["AWS", "Terraform", "Bash", "Security"]),
    ("Technical Product Manager Search & Recommender", ["Product Management", "Search Recommenders", "SQL"]),
    ("AI Product Manager LLM Roadmapping", ["AI Product Development", "Product Strategy", "Agile"]),
    ("Growth Product Manager AB Testing User Research", ["Product Strategy", "A/B Testing", "Data Analysis"]),
    ("Full Stack React TypeScript Python Developer", ["Python", "FastAPI", "React", "TypeScript"]),
    ("Next.js Frontend UI Tailwind CSS Developer", ["TypeScript", "React", "Next.js", "Tailwind CSS"]),
    ("GraphQL REST API Frontend Engineer", ["React", "Redux", "TypeScript", "GraphQL"]),
]


def generate_queries():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs_csv = os.path.join(base_dir, "data", "jobs.csv")
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    with open(jobs_csv, mode="r", encoding="utf-8") as f:
        jobs = list(csv.DictReader(f))

    eval_data = {
        "_meta": {
            "status": "DRAFT_EVAL_BENCHMARK",
            "note": "25 evaluation benchmark queries with relevance labels derived via skill & role heuristics across data/jobs.csv. Intended for verification & retrieval evaluation.",
            "total_queries": len(QUERIES_SPEC),
            "benchmark_dataset_size": len(jobs),
        },
        "queries": [],
    }

    for q_id, (q_text, skills) in enumerate(QUERIES_SPEC, start=1):
        matching_job_ids = []
        for j in jobs:
            score = 0
            title_lower = j["title"].lower()
            skills_lower = j["skills"].lower()
            desc_lower = j["description"].lower()

            # Weight skill matches
            for s in skills:
                if s.lower() in skills_lower:
                    score += 3
                if s.lower() in desc_lower:
                    score += 1

            # Weight title keyword matches
            for word in q_text.lower().split():
                if len(word) > 3 and word in title_lower:
                    score += 4

            if score >= 7:
                matching_job_ids.append(int(j["id"]))

        # Cap relevant set to top 5-8 genuine matches
        eval_data["queries"].append({
            "id": q_id,
            "query": q_text,
            "target_skills": skills,
            "relevant_job_ids": matching_job_ids[:8],
        })

    out_file = os.path.join(eval_dir, "queries.json")
    with open(out_file, mode="w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Generated {len(eval_data['queries'])} benchmark queries in {out_file}")


if __name__ == "__main__":
    generate_queries()

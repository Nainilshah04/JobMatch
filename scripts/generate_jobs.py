"""Synthetic Job Postings Generator for JobMatch.

Generates ~850 highly realistic job postings across Data, Backend, Machine Learning,
DevOps, and Product roles. Used when an external dataset is not provided.
The resulting records are saved to data/jobs.csv.
"""

import csv
import os
import random
from typing import List, Dict, Any

# Seed for reproducible synthetic generation
random.seed(42)

ROLES = {
    "Backend": [
        "Backend Software Engineer",
        "Senior Backend Engineer",
        "Staff Python Engineer",
        "Go Backend Developer",
        "Distributed Systems Engineer",
        "API Platform Engineer",
        "Cloud Backend Engineer",
        "Junior Backend Developer",
    ],
    "Machine Learning & AI": [
        "Machine Learning Engineer",
        "Senior AI/ML Engineer",
        "LLM Application Engineer",
        "RAG & Search Engineer",
        "NLP Research Engineer",
        "Computer Vision Engineer",
        "MLOps Engineer",
        "AI Platform Engineer",
    ],
    "Data": [
        "Data Engineer",
        "Senior Data Platform Engineer",
        "Analytics Engineer",
        "Big Data Engineer",
        "Data Scientist",
        "Quantitative Data Analyst",
        "Business Intelligence Engineer",
    ],
    "DevOps & Infrastructure": [
        "DevOps Engineer",
        "Site Reliability Engineer (SRE)",
        "Cloud Infrastructure Engineer",
        "Platform & Kubernetes Engineer",
        "Security & Infrastructure Engineer",
    ],
    "Product & Engineering Management": [
        "Technical Product Manager",
        "AI Product Manager",
        "Associate Product Manager",
        "Product Manager - Search & Discovery",
        "Engineering Manager - Data Platforms",
    ],
    "Full Stack": [
        "Full Stack Engineer",
        "Senior Full Stack Developer",
        "Product Engineer (Full Stack)",
        "Frontend & UI Engineer",
    ],
}

COMPANIES = [
    ("Elip AI", "AI career agent and intelligent candidate matching platform (YC W25)"),
    ("VectorFlow", "Enterprise vector search and neural retrieval infrastructure"),
    ("NeuralGraph", "Knowledge graph retrieval augmented generation platform"),
    ("DataPulse", "Real-time streaming analytics and event-driven data systems"),
    ("CloudScale", "Automated multi-cloud deployment and Kubernetes management"),
    ("AgenticSystems", "Autonomous multi-agent workflows for enterprise productivity"),
    ("FinMetrics", "Algorithmic risk evaluation and quantitative finance intelligence"),
    ("HealthVector", "Clinical NLP and medical document intelligence solutions"),
    ("RetailMind", "E-commerce recommendations and real-time personalized search"),
    ("CyberShield", "Automated threat detection and cloud security compliance"),
    ("OmniStack", "Modern developer tools and API orchestration framework"),
    ("KubeOps", "Next-generation observability, tracing, and cluster resilience"),
    ("SignalMesh", "Distributed messaging and low-latency microservice fabric"),
    ("TalentEngine", "Next-generation recruitment tech and skill graph verification"),
    ("CognitiveCore", "Fine-tuned foundation models for enterprise document workflows"),
]

LOCATIONS = [
    ("Bengaluru, India", False),
    ("Gurugram, India", False),
    ("Hyderabad, India", False),
    ("San Francisco, CA", False),
    ("New York, NY", False),
    ("Seattle, WA", False),
    ("Austin, TX", False),
    ("London, UK", False),
    ("Berlin, Germany", False),
    ("Remote, India", True),
    ("Remote, US", True),
    ("Remote, Global", True),
]

ROLE_SKILLS = {
    "Backend": [
        ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis", "REST APIs"],
        ["Go", "gRPC", "Kubernetes", "Kafka", "PostgreSQL", "Microservices"],
        ["Python", "Django", "Celery", "PostgreSQL", "AWS", "CI/CD"],
        ["Java", "Spring Boot", "Kafka", "PostgreSQL", "Elasticsearch", "Docker"],
        ["Node.js", "TypeScript", "Express", "MongoDB", "PostgreSQL", "Docker"],
        ["Python", "AsyncIO", "PostgreSQL", "pgvector", "Redis", "Docker"],
    ],
    "Machine Learning & AI": [
        ["Python", "PyTorch", "Sentence-Transformers", "pgvector", "RAG", "FastAPI"],
        ["Python", "LLMs", "LangChain", "OpenAI API", "Vector Databases", "Prompt Engineering"],
        ["Python", "Transformers", "HuggingFace", "Fine-Tuning", "vLLM", "CUDA"],
        ["Python", "MLflow", "Kubeflow", "Docker", "Kubernetes", "AWS SageMaker", "Triton"],
        ["Python", "Scikit-Learn", "XGBoost", "Pandas", "Feature Stores", "SQL"],
        ["Python", "NLP", "Spacy", "Embeddings", "Semantic Search", "FastAPI"],
    ],
    "Data": [
        ["Python", "SQL", "Apache Spark", "Airflow", "Snowflake", "dbt"],
        ["SQL", "dbt", "Snowflake", "BigQuery", "Data Modeling", "Tableau"],
        ["Python", "Kafka", "Apache Flink", "PostgreSQL", "AWS Redshift", "Docker"],
        ["Python", "SQL", "Pandas", "Databricks", "Delta Lake", "ETL Pipelines"],
        ["SQL", "Python", "Data Warehousing", "ClickHouse", "Presto", "Metabase"],
    ],
    "DevOps & Infrastructure": [
        ["Kubernetes", "Docker", "Terraform", "AWS", "CI/CD", "GitHub Actions"],
        ["Prometheus", "Grafana", "Linux", "Kubernetes", "Python", "Terraform"],
        ["GCP", "Kubernetes", "Helm", "ArgoCD", "Infrastructure as Code", "Docker"],
        ["AWS", "Terraform", "Docker", "PostgreSQL Admin", "Bash", "Security"],
    ],
    "Product & Engineering Management": [
        ["Product Strategy", "Technical Specifications", "Agile", "Roadmapping", "A/B Testing", "Data Analysis"],
        ["AI Product Development", "User Research", "Metrics & KPIs", "System Design Understanding", "Stakeholder Management"],
        ["Product Management", "Search Recommenders", "SQL", "User Journey Mapping", "Cross-functional Leadership"],
    ],
    "Full Stack": [
        ["TypeScript", "React", "Next.js", "Node.js", "PostgreSQL", "Tailwind CSS"],
        ["Python", "FastAPI", "React", "TypeScript", "Docker", "PostgreSQL"],
        ["React", "Redux", "TypeScript", "GraphQL", "REST APIs", "CSS Modules"],
    ],
}

SALARY_RANGES = [
    "₹15,00,000 - ₹25,00,000",
    "₹22,00,000 - ₹35,00,000",
    "₹30,00,000 - ₹50,00,000",
    "₹45,00,000 - ₹75,00,000",
    "$110,000 - $140,000",
    "$130,000 - $170,000",
    "$160,000 - $210,000",
    "$190,000 - $250,000",
]

RESPONSIBILITY_TEMPLATES = [
    "Design, build, and maintain robust production APIs and microservices serving high-throughput workloads.",
    "Collaborate with cross-functional product, AI, and infrastructure teams to deliver high-impact user experiences.",
    "Optimize data storage, indexing, and retrieval layers for low latency and high availability.",
    "Lead architecture reviews, implement clean automated testing, and uphold engineering excellence across the stack.",
    "Deploy containerized applications to cloud environments using modern CI/CD automation and observability.",
]

REQUIREMENTS_TEMPLATES = [
    "Strong hands-on software development experience with proven track record of shipping production features.",
    "Solid understanding of relational databases (PostgreSQL preferred), schema design, and query optimization.",
    "Familiarity with distributed systems, RESTful principles, asynchronous processing, and cloud architectures.",
    "Demonstrated problem-solving abilities, self-driven ownership, and strong technical communication.",
]


def generate_single_job(job_id: int) -> Dict[str, Any]:
    """Create a single coherent, realistic synthetic job posting."""
    category = random.choice(list(ROLES.keys()))
    title = random.choice(ROLES[category])
    company, company_desc = random.choice(COMPANIES)
    location_name, is_remote_loc = random.choice(LOCATIONS)
    remote = is_remote_loc or (random.random() < 0.35)
    
    skill_set = random.choice(ROLE_SKILLS[category]).copy()
    # Randomly introduce 1-2 complementary skills
    extra_pool = ["Docker", "Git", "Linux", "FastAPI", "SQL", "Redis", "CI/CD", "AWS"]
    if random.random() < 0.5:
        add_skill = random.choice(extra_pool)
        if add_skill not in skill_set:
            skill_set.append(add_skill)
    
    skills_str = ", ".join(skill_set)
    salary = random.choice(SALARY_RANGES) if random.random() > 0.1 else None

    # Construct rich, realistic description
    primary_skills_text = ", ".join(skill_set[:3])
    description = (
        f"About {company}:\n"
        f"{company} is building {company_desc}. We are expanding our engineering team to solve challenging scalability, "
        f"reliability, and product problems.\n\n"
        f"Role Overview:\n"
        f"As a {title} at {company}, you will work on core backend and platform capabilities using {primary_skills_text}. "
        f"You will have significant autonomy to shape system design, scale services, and ship features directly impacting users.\n\n"
        f"Key Responsibilities:\n"
        f"- {random.choice(RESPONSIBILITY_TEMPLATES)}\n"
        f"- {random.choice(RESPONSIBILITY_TEMPLATES)}\n"
        f"- Work closely with technical leads to build maintainable, well-tested code.\n\n"
        f"Requirements:\n"
        f"- Hands-on proficiency with {skills_str}.\n"
        f"- {random.choice(REQUIREMENTS_TEMPLATES)}\n"
        f"- {random.choice(REQUIREMENTS_TEMPLATES)}\n\n"
        f"What We Offer:\n"
        f"Competitive compensation, flexible work environment ({'Remote' if remote else location_name}), "
        f"and opportunities to work with state-of-the-art AI and distributed systems."
    )

    return {
        "id": job_id,
        "title": title,
        "company": company,
        "location": location_name,
        "description": description,
        "skills": skills_str,
        "salary": salary or "",
        "remote": str(remote).lower(),
    }


def generate_jobs_dataset(output_path: str, count: int = 850) -> None:
    """Generate dataset and save to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fieldnames = ["id", "title", "company", "location", "description", "skills", "salary", "remote"]
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, count + 1):
            record = generate_single_job(i)
            writer.writerow(record)
            
    print(f"Generated {count} synthetic jobs successfully at: {output_path}")


if __name__ == "__main__":
    target_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jobs.csv")
    generate_jobs_dataset(target_csv, count=850)

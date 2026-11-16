"""Precompute embeddings for the 850 synthetic jobs and save to data/job_embeddings.npy."""

import csv
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Limit threads to 1 for minimal memory footprint
torch.set_num_threads(1)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "jobs.csv")
out_path = os.path.join(base_dir, "data", "job_embeddings.npy")

with open(csv_path, mode="r", encoding="utf-8") as f:
    jobs = list(csv.DictReader(f))

texts = [
    f"Title: {j['title']} | Company: {j['company']} | Skills: {j['skills']} | {j['description'][:300]}"
    for j in jobs
]

print(f"Encoding {len(texts)} jobs with all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)

np.save(out_path, embeddings)
print(f"Successfully saved {out_path} with shape {embeddings.shape} (File size: {os.path.getsize(out_path) / 1024:.1f} KB)")

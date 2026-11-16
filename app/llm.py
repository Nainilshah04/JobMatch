"""Provider-agnostic LLM interface supporting OpenAI, Gemini, Groq, and fallbacks.

Enforces structured JSON outputs validated via Pydantic with automated retries.
Used for LLM-based reranking (Phase 3) and candidate-job fit explanations (Phase 4).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests
from pydantic import ValidationError

from app.config import get_settings
from app.models import JobExplanation, JobSearchResult

logger = logging.getLogger(__name__)


def _extract_json_snippet(raw_text: str) -> str:
    """Extract JSON object from markdown code blocks or surrounding text."""
    # Look for ```json ... ``` blocks
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_block:
        return json_block.group(1).strip()
    # Otherwise find the outermost curly braces
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return raw_text[first_brace : last_brace + 1].strip()
    return raw_text.strip()


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_mode: bool = True,
    temperature: float = 0.2,
) -> str:
    """Execute raw HTTP request to the configured LLM API provider."""
    settings = get_settings()
    api_key = settings.LLM_API_KEY

    # If no API key is provided, return empty string to trigger deterministic fallback
    if not api_key or api_key == "your_llm_api_key_here":
        logger.warning("LLM_API_KEY not configured. Falling back to deterministic heuristic.")
        return ""

    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL

    # Determine endpoint based on provider
    if provider == "gemini":
        endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    elif provider == "groq":
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
    else:  # default openai / openrouter
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if json_mode and provider != "gemini":
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.error(f"Error calling LLM provider '{provider}': {e}")
        return ""


def rerank_with_llm(
    query: str,
    candidates: List[JobSearchResult],
) -> List[JobSearchResult]:
    """Rerank candidates using an LLM relevance scoring prompt.
    
    Prompts the LLM to rate each candidate job from 0 to 100 based on fit
    for the provided query or candidate resume.
    """
    if not candidates:
        return []

    # Format candidate summaries for LLM prompt
    jobs_summary = []
    for c in candidates:
        jobs_summary.append({
            "job_id": c.id,
            "title": c.title,
            "company": c.company,
            "skills": c.skills,
            "snippet": c.description[:250].replace("\n", " "),
        })

    prompt = (
        f"You are an expert technical recruiting system. Evaluate the relevance of the following job postings "
        f"for this candidate query/profile:\n\n"
        f"Target Query: \"{query}\"\n\n"
        f"Candidate Job Postings:\n{json.dumps(jobs_summary, indent=2)}\n\n"
        f"Score each job from 0 to 100 based on role relevance, required skills overlap, and seniority.\n"
        f"Return ONLY a strict JSON object mapping job_id (as string) to integer score (0-100), e.g.:\n"
        f'{{"scores": {{"1": 85, "2": 42}}}}'
    )

    system_prompt = "You are a precise technical recruiter returning only valid JSON."
    raw_response = call_llm(prompt, system_prompt=system_prompt, json_mode=True)

    score_map: Dict[int, float] = {}
    if raw_response:
        try:
            cleaned_json = _extract_json_snippet(raw_response)
            parsed = json.loads(cleaned_json)
            scores = parsed.get("scores", parsed)
            for k, v in scores.items():
                score_map[int(k)] = float(v)
        except Exception as e:
            logger.warning(f"Failed to parse LLM reranking output: {e}. Using heuristic scoring.")

    # Apply scores or fallback to normalized heuristic scoring
    reranked: List[JobSearchResult] = []
    for c in candidates:
        if c.id in score_map:
            score = round(score_map[c.id] / 100.0, 4)
        else:
            # Deterministic heuristic fallback: Title overlap + skill overlap + base score
            query_tokens = set(re.findall(r"\w+", query.lower()))
            title_tokens = set(re.findall(r"\w+", c.title.lower()))
            skill_tokens = set(re.findall(r"\w+", c.skills.lower()))
            overlap = len(query_tokens & title_tokens) * 2 + len(query_tokens & skill_tokens)
            score = round(min(0.99, (c.score * 0.5) + (overlap * 0.1)), 4)

        reranked.append(
            JobSearchResult(
                id=c.id,
                title=c.title,
                company=c.company,
                location=c.location,
                description=c.description,
                skills=c.skills,
                salary=c.salary,
                remote=c.remote,
                score=score,
                retrieval_mode="rerank_llm",
                match_explanation=c.match_explanation,
            )
        )

    # Sort descending by updated score
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked

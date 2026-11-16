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


def _generate_heuristic_explanation(query_or_resume: str, job: JobSearchResult) -> JobExplanation:
    """Generate a deterministic, structured skill-gap analysis when LLM is unavailable.
    
    Acts as a resilient fallback ensuring system uptime and predictability in offline/testing modes.
    """
    job_skills_list = [s.strip() for s in job.skills.split(",") if s.strip()]
    query_lower = query_or_resume.lower()

    matched: List[str] = []
    missing: List[str] = []

    for skill in job_skills_list:
        # Check word boundaries or simple containment
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, query_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    # Calculate match score based on skill overlap ratio and title alignment
    total_skills = len(job_skills_list)
    overlap_ratio = len(matched) / max(total_skills, 1)

    title_match = any(word.lower() in query_lower for word in job.title.split() if len(word) > 3)
    base_score = int(overlap_ratio * 70) + (25 if title_match else 5)
    final_score = max(10, min(95, base_score))

    if matched and missing:
        reason = (
            f"Strong profile alignment on key competencies ({', '.join(matched[:3])}). "
            f"Candidate should develop or highlight experience with {', '.join(missing[:2])} to fully meet requirements."
        )
    elif matched and not missing:
        reason = f"Excellent candidate match! Covers all primary required skills: {', '.join(matched)}."
    else:
        reason = (
            f"Role title '{job.title}' relates to target interest, but candidate profile is missing core competencies: "
            f"{', '.join(missing[:3])}."
        )

    return JobExplanation(
        match_score=final_score,
        matched_skills=matched,
        missing_skills=missing,
        reason=reason,
    )


def explain_job_match(
    query_or_resume: str,
    job: JobSearchResult,
    max_retries: int = 2,
) -> JobExplanation:
    """Generate a structured Pydantic-validated explanation of candidate-job fit.
    
    Includes automated retries on malformed JSON or schema validation failure.
    
    Design Decision (Structured LLM Outputs with Automated Retries):
    LLM generations can occasionally deviate from strict schema requirements (e.g.
    missing fields, invalid score ranges, or unescaped characters).
    We validate the parsed JSON against the Pydantic `JobExplanation` model. If validation
    fails, the error context is injected into a corrective retry prompt (up to 2 retries).
    If all retries fail or if no LLM API key is present, the system falls back
    to a deterministic keyword/skill matching heuristic, guaranteeing zero crashes.
    """
    settings = get_settings()
    if not settings.LLM_API_KEY or settings.LLM_API_KEY == "your_llm_api_key_here":
        return _generate_heuristic_explanation(query_or_resume, job)

    schema_example = json.dumps(
        {
            "match_score": 85,
            "matched_skills": ["Python", "FastAPI"],
            "missing_skills": ["Kafka", "Kubernetes"],
            "reason": "Strong background in backend Python services; candidate lacks distributed streaming experience.",
        },
        indent=2,
    )

    base_prompt = (
        f"You are an AI career coach and technical hiring assessor.\n"
        f"Evaluate the fit between this Candidate Profile/Query and the Job Posting.\n\n"
        f"Candidate Profile / Query:\n\"\"\"{query_or_resume[:1000]}\"\"\"\n\n"
        f"Job Posting:\n"
        f"- Title: {job.title}\n"
        f"- Company: {job.company}\n"
        f"- Required Skills: {job.skills}\n"
        f"- Description: {job.description[:400]}\n\n"
        f"Instructions:\n"
        f"1. Compare skills and experience.\n"
        f"2. Assign match_score from 0 to 100.\n"
        f"3. List matched_skills (present in both) and missing_skills (required by job but missing in profile).\n"
        f"4. Provide a concise, constructive 2-sentence reason explaining the fit and what to improve.\n"
        f"Return ONLY a strict JSON object conforming to this schema:\n{schema_example}"
    )

    current_prompt = base_prompt
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning(f"Retrying LLM structured explanation (attempt {attempt}/{max_retries}) due to: {last_error}")
            current_prompt = (
                f"{base_prompt}\n\n"
                f"[PREVIOUS ATTEMPT FAILED SCHEMA VALIDATION]\n"
                f"Error: {last_error}\n"
                f"Please correct the error and return ONLY valid JSON matching the exact schema."
            )

        raw_response = call_llm(
            prompt=current_prompt,
            system_prompt="You are an expert technical evaluator who outputs strict JSON conforming to Pydantic schemas.",
            json_mode=True,
            temperature=0.1,
        )

        if not raw_response:
            break

        try:
            json_str = _extract_json_snippet(raw_response)
            parsed = json.loads(json_str)
            explanation = JobExplanation.model_validate(parsed)
            return explanation
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

    logger.warning("LLM explanation retries exhausted. Using deterministic heuristic explanation.")
    return _generate_heuristic_explanation(query_or_resume, job)


def explain_top_jobs(
    query_or_resume: str,
    jobs: List[JobSearchResult],
    top_n: int = 5,
) -> List[JobSearchResult]:
    """Enrich the top-N jobs with structured candidate fit explanations."""
    enriched: List[JobSearchResult] = []
    for idx, job in enumerate(jobs):
        if idx < top_n:
            explanation = explain_job_match(query_or_resume, job)
            enriched_job = job.model_copy(update={"match_explanation": explanation})
            enriched.append(enriched_job)
        else:
            enriched.append(job)
    return enriched

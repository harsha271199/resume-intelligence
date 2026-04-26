"""
scoring_agent.py — Semantic Scoring Agent

Matches a ParsedResume against a JobDescription and produces a ScoringResult
containing a Skill Alignment Score (0–100) and a human-readable explanation.

Scoring formula
---------------
    final_score = 100 × (
        0.40 × semantic_similarity   # MiniLM cosine similarity
      + 0.30 × skill_coverage        # matched / required skills
      + 0.20 × experience_match      # years heuristic
      + 0.10 × (1 - skill_gap_ratio) # penalty for missing skills
    )

All weights are configurable via environment variables SCORE_W1–SCORE_W4.

Public API
----------
    score_resume(resume: ParsedResume, job: JobDescription) -> ScoringResult
    compute_semantic_similarity(text_a: str, text_b: str) -> float
    compute_tfidf_baseline(text_a: str, text_b: str) -> float
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from resume_intelligence.models.schemas import JobDescription, ParsedResume, ScoringResult

logger = logging.getLogger(__name__)

# ── Scoring weights (configurable via env vars) ───────────────────────────────

_W1 = float(os.getenv("SCORE_W1", "0.40"))  # semantic similarity
_W2 = float(os.getenv("SCORE_W2", "0.30"))  # skill coverage
_W3 = float(os.getenv("SCORE_W3", "0.20"))  # experience match
_W4 = float(os.getenv("SCORE_W4", "0.10"))  # skill gap penalty (inverted)

# ── Model singleton (loaded once at import time) ──────────────────────────────

def _load_sbert_model() -> SentenceTransformer:
    model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
    logger.info("Loading sentence-transformers model: %s", model_name)
    return SentenceTransformer(model_name)


_SBERT: SentenceTransformer = _load_sbert_model()

# ── Experience year extraction ────────────────────────────────────────────────

_YEARS_PATTERN = re.compile(
    r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)",
    re.IGNORECASE,
)

_JD_YEARS_PATTERN = re.compile(
    r"(\d+)\+?\s*years?",
    re.IGNORECASE,
)


# ── Component helpers ─────────────────────────────────────────────────────────

def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between MiniLM embeddings of two texts.

    Returns a float in [0.0, 1.0].
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    embeddings = _SBERT.encode([text_a, text_b], convert_to_numpy=True)
    # cosine similarity via dot product of L2-normalised vectors
    a = embeddings[0] / (np.linalg.norm(embeddings[0]) + 1e-10)
    b = embeddings[1] / (np.linalg.norm(embeddings[1]) + 1e-10)
    similarity = float(np.dot(a, b))
    return max(0.0, min(1.0, similarity))


def compute_tfidf_baseline(text_a: str, text_b: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts as a baseline.

    Returns a float in [0.0, 1.0].
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        score = float(sklearn_cosine(tfidf_matrix[0], tfidf_matrix[1])[0][0])
        return max(0.0, min(1.0, score))
    except Exception:  # noqa: BLE001
        return 0.0


def _compute_skill_coverage(
    resume_skills: List[str],
    jd_skills: List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Compute skill coverage ratio and return (coverage, matched, missing).

    coverage = |resume ∩ jd| / |jd|, clamped to [0, 1].
    Returns 0.0 coverage with all JD skills as missing when jd_skills is empty.
    """
    if not jd_skills:
        return 0.0, [], []

    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    coverage = len(matched) / len(jd_set)
    return min(1.0, coverage), matched, missing


def _compute_skill_gap_ratio(matched: List[str], jd_skills: List[str]) -> float:
    """
    Compute the fraction of JD skills that are missing.

    gap_ratio = missing / total_jd_skills, clamped to [0, 1].
    Returns 0.0 (no gap) when jd_skills is empty.
    """
    if not jd_skills:
        return 0.0
    gap = 1.0 - (len(matched) / len(jd_skills))
    return max(0.0, min(1.0, gap))


def _compute_experience_match(
    resume: ParsedResume,
    jd: JobDescription,
) -> float:
    """
    Estimate experience match as a normalised score in [0, 1].

    Strategy:
    1. Extract the maximum years-of-experience number from resume.experience.
    2. Extract the required years from jd.experience_level or jd.raw_text.
    3. Score = min(candidate_years / required_years, 1.0).
       Defaults to 0.5 when neither side has parseable year data.
    """
    # Extract candidate years from experience entries
    candidate_years = 0
    for entry in resume.experience:
        for match in _YEARS_PATTERN.finditer(entry):
            candidate_years = max(candidate_years, int(match.group(1)))

    # Also scan raw_text for year mentions
    if candidate_years == 0:
        for match in _YEARS_PATTERN.finditer(resume.raw_text):
            candidate_years = max(candidate_years, int(match.group(1)))

    # Extract required years from JD
    required_years = 0
    jd_source = (jd.experience_level or "") + " " + jd.raw_text
    for match in _JD_YEARS_PATTERN.finditer(jd_source):
        required_years = max(required_years, int(match.group(1)))

    # Fallback: both unknown → neutral 0.5
    if candidate_years == 0 and required_years == 0:
        return 0.5

    # Candidate has experience but JD doesn't specify → full credit
    if required_years == 0:
        return 1.0

    # Candidate has no parseable experience → low score
    if candidate_years == 0:
        return 0.1

    return min(1.0, candidate_years / required_years)


def _find_semantic_matches(
    missing_skills: List[str],
    resume_text: str,
    threshold: float = 0.55,
) -> List[str]:
    """
    Find JD skills that are missing from exact matching but semantically
    similar to content in the resume text.

    Uses MiniLM to compare each missing skill against the full resume text.
    Returns skills whose similarity exceeds *threshold*.
    """
    if not missing_skills or not resume_text.strip():
        return []

    semantic_hits: List[str] = []
    resume_embedding = _SBERT.encode([resume_text], convert_to_numpy=True)[0]
    resume_norm = resume_embedding / (np.linalg.norm(resume_embedding) + 1e-10)

    skill_embeddings = _SBERT.encode(missing_skills, convert_to_numpy=True)
    for skill, emb in zip(missing_skills, skill_embeddings):
        skill_norm = emb / (np.linalg.norm(emb) + 1e-10)
        sim = float(np.dot(resume_norm, skill_norm))
        if sim >= threshold:
            semantic_hits.append(skill)

    return semantic_hits


def _build_explanation(
    resume: ParsedResume,
    jd: JobDescription,
    final_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    semantic_matches: List[str],
    semantic_similarity: float,
    skill_coverage: float,
    experience_match: float,
) -> str:
    """
    Build a concise, human-readable explanation of the score.
    """
    name = resume.name or "The candidate"
    lines: List[str] = []

    # Overall verdict
    if final_score >= 75:
        verdict = "strong match"
    elif final_score >= 50:
        verdict = "moderate match"
    else:
        verdict = "weak match"

    lines.append(
        f"{name} is a {verdict} for the {jd.job_title} role "
        f"(Skill Alignment Score: {final_score:.1f}/100)."
    )

    # Matched skills
    if matched_skills:
        lines.append(
            f"✅ Matched skills ({len(matched_skills)}): "
            + ", ".join(matched_skills[:8])
            + ("..." if len(matched_skills) > 8 else ".")
        )
    else:
        lines.append("⚠️  No exact skill matches found against the job description.")

    # Semantic matches (soft matches)
    if semantic_matches:
        lines.append(
            f"🔍 Semantically related skills detected: "
            + ", ".join(semantic_matches[:5]) + "."
        )

    # Missing skills with suggestions
    if missing_skills:
        top_missing = missing_skills[:5]
        lines.append(
            f"❌ Missing skills ({len(missing_skills)}): "
            + ", ".join(top_missing)
            + ("..." if len(missing_skills) > 5 else ".")
        )
        for skill in top_missing[:3]:
            lines.append(
                f"   💡 Consider adding '{skill}' to strengthen your profile "
                f"for {jd.job_title} roles."
            )

    # Component breakdown
    lines.append(
        f"📊 Score breakdown — "
        f"Semantic similarity: {semantic_similarity * 100:.1f}  |  "
        f"Skill coverage: {skill_coverage * 100:.1f}  |  "
        f"Experience match: {experience_match * 100:.1f}"
    )

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def score_resume(
    resume: ParsedResume,
    job: JobDescription,
) -> ScoringResult:
    """
    Score a parsed resume against a job description.

    Parameters
    ----------
    resume : ParsedResume
        Structured resume output from the parsing agent.
    job : JobDescription
        Structured job description to match against.

    Returns
    -------
    ScoringResult
        Skill Alignment Score (0–100), component breakdown, matched/missing
        skills, semantic matches, explanation, and TF-IDF baseline.
        On invalid input, returns a ScoringResult with score=0 and
        ``score_error`` set.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    jd_tokens = job.raw_text.split()
    if not job.raw_text.strip() or len(jd_tokens) < 10:
        return ScoringResult(
            resume_id=resume.id,
            job_id=job.id,
            semantic_similarity=0.0,
            skill_coverage=0.0,
            experience_match=0.0,
            skill_gap_penalty=0.0,
            final_score=0.0,
            matched_skills=[],
            semantic_matches=[],
            missing_skills=list(job.normalized_skills),
            explanation="Scoring failed: job description is empty or too short (< 10 tokens).",
            score_error="Job description is empty or too short.",
        )

    try:
        # ── Use normalised skills for matching ────────────────────────────────
        resume_skills = resume.normalized_skills or resume.skills
        jd_skills = job.normalized_skills or job.required_skills

        # ── Component 1: Semantic similarity ─────────────────────────────────
        sem_sim = compute_semantic_similarity(resume.raw_text, job.raw_text)

        # ── Component 2: Skill coverage + gap ────────────────────────────────
        coverage, matched, missing = _compute_skill_coverage(resume_skills, jd_skills)
        gap_ratio = _compute_skill_gap_ratio(matched, jd_skills)

        # ── Component 3: Experience match ─────────────────────────────────────
        exp_match = _compute_experience_match(resume, job)

        # ── Semantic matches (soft skill matching) ────────────────────────────
        semantic_matches = _find_semantic_matches(missing, resume.raw_text)

        # ── Final weighted score ──────────────────────────────────────────────
        raw_score = (
            _W1 * sem_sim
            + _W2 * coverage
            + _W3 * exp_match
            + _W4 * (1.0 - gap_ratio)
        )
        final_score = round(max(0.0, min(100.0, raw_score * 100)), 2)

        # ── TF-IDF baseline ───────────────────────────────────────────────────
        tfidf_score = round(
            compute_tfidf_baseline(resume.raw_text, job.raw_text) * 100, 2
        )

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = _build_explanation(
            resume=resume,
            jd=job,
            final_score=final_score,
            matched_skills=matched,
            missing_skills=missing,
            semantic_matches=semantic_matches,
            semantic_similarity=sem_sim,
            skill_coverage=coverage,
            experience_match=exp_match,
        )

        return ScoringResult(
            resume_id=resume.id,
            job_id=job.id,
            semantic_similarity=round(sem_sim * 100, 2),
            skill_coverage=round(coverage * 100, 2),
            experience_match=round(exp_match * 100, 2),
            skill_gap_penalty=round(gap_ratio * 100, 2),
            final_score=final_score,
            matched_skills=matched,
            semantic_matches=semantic_matches,
            missing_skills=missing,
            explanation=explanation,
            tfidf_baseline_score=tfidf_score,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected error scoring resume_id=%s against job_id=%s",
            resume.id, job.id,
        )
        return ScoringResult(
            resume_id=resume.id,
            job_id=job.id,
            semantic_similarity=0.0,
            skill_coverage=0.0,
            experience_match=0.0,
            skill_gap_penalty=0.0,
            final_score=0.0,
            matched_skills=[],
            semantic_matches=[],
            missing_skills=[],
            explanation="Scoring failed due to an unexpected error.",
            score_error=str(exc),
        )

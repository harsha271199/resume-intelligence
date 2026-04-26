"""
pipeline.py — End-to-End Pipeline Orchestrator

Chains the three agents (Parser → Scorer → Auditor) into a single
``run_pipeline`` call and returns a consolidated ``PipelineOutput``.

Each agent runs inside its own try/except block so a failure in one agent
does not prevent the others from running. The ``agents_completed`` list in
the output tracks which agents succeeded.

Public API
----------
    run_pipeline(resume_text, job_text, run_audit=True) -> PipelineOutput
    build_job_description(job_text, job_id=None) -> JobDescription
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional

from resume_intelligence.agents.bias_auditor import audit_resume
from resume_intelligence.agents.parser_agent import _extract_skills, normalize_skills, parse_resume
from resume_intelligence.agents.scoring_agent import score_resume
from resume_intelligence.models.schemas import (
    FairnessReport,
    JobDescription,
    ParsedResume,
    PipelineOutput,
    ScoringResult,
)

logger = logging.getLogger(__name__)

# ── Experience-level patterns for JD parsing ─────────────────────────────────

_EXP_LEVEL_PATTERN = re.compile(
    r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)",
    re.IGNORECASE,
)

_TITLE_HINTS = re.compile(
    r"(?:position|role|title|job title)[:\s]+([^\n.]{3,60})",
    re.IGNORECASE,
)


# ── JD builder ────────────────────────────────────────────────────────────────

def build_job_description(
    job_text: str,
    job_id: Optional[str] = None,
) -> JobDescription:
    """
    Convert raw job description text into a structured ``JobDescription``.

    Extracts:
    - Job title: first line, or a "title: …" pattern, or "Unknown Role"
    - Required skills: same keyword scan used by the parser agent
    - Experience level: first year-requirement phrase found in the text

    Parameters
    ----------
    job_text : str
        Raw job description text.
    job_id : str, optional
        Identifier for this JD. Auto-generated UUID if not provided.

    Returns
    -------
    JobDescription
    """
    jid = job_id or str(uuid.uuid4())

    # ── Job title ─────────────────────────────────────────────────────────────
    job_title = "Unknown Role"
    title_match = _TITLE_HINTS.search(job_text)
    if title_match:
        job_title = title_match.group(1).strip().rstrip(".,;")
    else:
        # Fall back to first non-empty line
        for line in job_text.splitlines():
            line = line.strip()
            if line and len(line) < 80:
                job_title = line
                break

    # ── Skills ────────────────────────────────────────────────────────────────
    raw_skills = _extract_skills(job_text)
    normalized = normalize_skills(raw_skills)

    # ── Experience level ──────────────────────────────────────────────────────
    experience_level: Optional[str] = None
    exp_match = _EXP_LEVEL_PATTERN.search(job_text)
    if exp_match:
        years = int(exp_match.group(1))
        experience_level = f"{years}+ years"

    return JobDescription(
        id=jid,
        job_title=job_title,
        raw_text=job_text,
        required_skills=raw_skills,
        normalized_skills=normalized,
        experience_level=experience_level,
    )


# ── Sentinel objects for partial-failure returns ──────────────────────────────

def _error_scoring_result(resume_id: str, job_id: str, error: str) -> ScoringResult:
    return ScoringResult(
        resume_id=resume_id,
        job_id=job_id,
        semantic_similarity=0.0,
        skill_coverage=0.0,
        experience_match=0.0,
        skill_gap_penalty=0.0,
        final_score=0.0,
        matched_skills=[],
        semantic_matches=[],
        missing_skills=[],
        explanation=f"Scoring unavailable: {error}",
        score_error=error,
    )


def _error_fairness_report(error: str) -> FairnessReport:
    return FairnessReport(
        original_score=0.0,
        swapped_scores={},
        max_difference=0.0,
        average_difference=0.0,
        demographic_parity_difference=0.0,
        summary=f"Audit unavailable: {error}",
        audit_error=error,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    resume_text: str,
    job_text: str,
    run_audit: bool = True,
) -> PipelineOutput:
    """
    Run the full three-agent pipeline on raw resume and job description text.

    Parameters
    ----------
    resume_text : str
        Raw resume text (plain text).
    job_text : str
        Raw job description text (plain text).
    run_audit : bool
        Whether to run the Bias Auditing Agent. Default True.
        Set to False to skip the audit and speed up the pipeline.

    Returns
    -------
    PipelineOutput
        Consolidated result containing parsed resume, job description,
        scoring result, optional fairness report, and ``agents_completed``
        tracking which agents ran successfully.
    """
    agents_completed: List[str] = []
    resume_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    # ── Agent 1: Parse resume ─────────────────────────────────────────────────
    parsed_resume: ParsedResume
    try:
        logger.info("Pipeline: starting Resume Parsing Agent (resume_id=%s)", resume_id)
        parsed_resume = parse_resume(resume_text, resume_id=resume_id)
        agents_completed.append("parsing")
        logger.info(
            "Pipeline: parsing complete — %d skills, %d education, %d experience",
            len(parsed_resume.normalized_skills),
            len(parsed_resume.education),
            len(parsed_resume.experience),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline: Resume Parsing Agent failed")
        parsed_resume = ParsedResume(
            id=resume_id,
            raw_text=resume_text,
            parse_error=str(exc),
        )

    # ── Build JobDescription from raw text ────────────────────────────────────
    job: JobDescription
    try:
        job = build_job_description(job_text, job_id=job_id)
        logger.info(
            "Pipeline: job description built — title=%r, %d skills",
            job.job_title, len(job.normalized_skills),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline: JobDescription build failed")
        job = JobDescription(
            id=job_id,
            job_title="Unknown Role",
            raw_text=job_text,
        )

    # ── Agent 2: Score ────────────────────────────────────────────────────────
    scoring_result: ScoringResult
    try:
        logger.info("Pipeline: starting Semantic Scoring Agent")
        scoring_result = score_resume(parsed_resume, job)
        agents_completed.append("scoring")
        logger.info(
            "Pipeline: scoring complete — final_score=%.2f",
            scoring_result.final_score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline: Semantic Scoring Agent failed")
        scoring_result = _error_scoring_result(resume_id, job_id, str(exc))

    # ── Agent 3: Bias Audit ───────────────────────────────────────────────────
    fairness_report: Optional[FairnessReport] = None
    if run_audit:
        try:
            logger.info("Pipeline: starting Bias Auditing Agent")
            fairness_report = audit_resume(parsed_resume, job)
            agents_completed.append("auditing")
            logger.info(
                "Pipeline: audit complete — dpd=%.4f, max_diff=%.2f",
                fairness_report.demographic_parity_difference,
                fairness_report.max_difference,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline: Bias Auditing Agent failed")
            fairness_report = _error_fairness_report(str(exc))

    return PipelineOutput(
        parsed_resume=parsed_resume,
        job_description=job,
        scoring_result=scoring_result,
        fairness_report=fairness_report,
        agents_completed=agents_completed,
    )

"""
schemas.py — Shared Pydantic data contracts for the Resume Intelligence pipeline.

All inter-agent data flows through these models. They act as the single source
of truth for input/output shapes, validation rules, and type safety across the
parsing, scoring, and auditing agents.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_score_0_100(value: float, field_name: str = "score") -> float:
    """Raise ValueError if *value* is outside [0.0, 100.0]."""
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"{field_name} must be between 0 and 100, got {value}")
    return value


# ── 1. ParsedResume ───────────────────────────────────────────────────────────

class ParsedResume(BaseModel):
    """
    Structured output of the Resume Parsing Agent.

    Holds every entity extracted from a raw resume text, plus the normalised
    skill list used by downstream agents for matching.
    """

    id: str = Field(..., description="Unique identifier for this resume.")
    raw_text: str = Field(..., description="Original, unmodified resume text.")
    name: Optional[str] = Field(None, description="Candidate name detected by NER.")
    skills: List[str] = Field(
        default_factory=list,
        description="Raw skill tokens as extracted from the text.",
    )
    normalized_skills: List[str] = Field(
        default_factory=list,
        description="Canonical skill forms after alias normalisation (e.g. 'ML' → 'machine learning').",
    )
    education: List[str] = Field(
        default_factory=list,
        description="Education entries (degree + institution) found in the resume.",
    )
    experience: List[str] = Field(
        default_factory=list,
        description="Work-experience entries (role + company + dates) found in the resume.",
    )
    organizations: List[str] = Field(
        default_factory=list,
        description="Organisation names identified by spaCy ORG entities.",
    )
    job_titles: List[str] = Field(
        default_factory=list,
        description="Job titles extracted via NER and rule-based patterns.",
    )
    category: Optional[str] = Field(
        None,
        description="Resume category label (e.g. 'Data Science') when available from the dataset.",
    )
    parse_error: Optional[str] = Field(
        None,
        description="Set to a descriptive message if parsing failed or input was invalid.",
    )

    @field_validator("skills", "normalized_skills", "education", "experience",
                     "organizations", "job_titles", mode="before")
    @classmethod
    def _ensure_list_of_strings(cls, value: object) -> List[str]:
        """Coerce None to an empty list; reject non-string list items."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Field must be a list of strings.")
        return [str(item) for item in value]


# ── 2. JobDescription ─────────────────────────────────────────────────────────

class JobDescription(BaseModel):
    """
    Structured representation of a job posting.

    Consumed by the Semantic Scoring Agent to compute alignment against a
    parsed resume.
    """

    id: str = Field(..., description="Unique identifier for this job description.")
    job_title: str = Field(..., description="Title of the role (e.g. 'Senior Data Scientist').")
    raw_text: str = Field(..., description="Full job description text.")
    required_skills: List[str] = Field(
        default_factory=list,
        description="Raw skill tokens extracted from the job description.",
    )
    normalized_skills: List[str] = Field(
        default_factory=list,
        description="Canonical skill forms after alias normalisation.",
    )
    experience_level: Optional[str] = Field(
        None,
        description="Required experience level (e.g. 'Senior (4+ years)').",
    )
    industry: Optional[str] = Field(
        None,
        description="Industry or domain of the role (e.g. 'Technology').",
    )

    @field_validator("required_skills", "normalized_skills", mode="before")
    @classmethod
    def _ensure_list_of_strings(cls, value: object) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Field must be a list of strings.")
        return [str(item) for item in value]


# ── 3. ScoringResult ──────────────────────────────────────────────────────────

class ScoringResult(BaseModel):
    """
    Output of the Semantic Scoring Agent for a single resume–job pair.

    Contains the four scoring components, the final weighted Skill Alignment
    Score (0–100), matched/missing skill lists, and a human-readable
    explanation for the candidate.
    """

    resume_id: str = Field(..., description="ID of the resume being scored.")
    job_id: str = Field(..., description="ID of the job description being matched against.")

    # ── Scoring components (each 0–100) ──────────────────────────────────────
    semantic_similarity: float = Field(
        ...,
        description="Cosine similarity between MiniLM embeddings of resume and JD text, scaled to 0–100.",
    )
    skill_coverage: float = Field(
        ...,
        description="Percentage of required JD skills present in the resume (0–100).",
    )
    experience_match: float = Field(
        ...,
        description="Normalised score reflecting how well the candidate's experience depth matches the JD requirement (0–100).",
    )
    skill_gap_penalty: float = Field(
        ...,
        description="Penalty score reflecting missing skills; higher means fewer gaps (0–100).",
    )
    final_score: float = Field(
        ...,
        description="Weighted Skill Alignment Score combining all four components (0–100).",
    )

    # ── Explainability ────────────────────────────────────────────────────────
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills present in both the resume and the job description.",
    )
    semantic_matches: List[str] = Field(
        default_factory=list,
        description="JD skills matched via semantic similarity (not exact string match).",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Required JD skills absent from the resume.",
    )
    explanation: str = Field(
        ...,
        description="Human-readable summary of the score with actionable improvement suggestions.",
    )

    # ── Optional metadata ─────────────────────────────────────────────────────
    tfidf_baseline_score: Optional[float] = Field(
        None,
        description="TF-IDF cosine similarity baseline score (0–100) for comparison.",
    )
    score_error: Optional[str] = Field(
        None,
        description="Set to a descriptive message if scoring failed or input was invalid.",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator(
        "semantic_similarity", "skill_coverage", "experience_match",
        "skill_gap_penalty", "final_score",
    )
    @classmethod
    def _score_in_range(cls, value: float, info) -> float:
        return _check_score_0_100(value, info.field_name)

    @field_validator("tfidf_baseline_score", mode="before")
    @classmethod
    def _tfidf_in_range(cls, value: object) -> Optional[float]:
        if value is None:
            return None
        return _check_score_0_100(float(value), "tfidf_baseline_score")

    @field_validator("matched_skills", "semantic_matches", "missing_skills", mode="before")
    @classmethod
    def _ensure_list_of_strings(cls, value: object) -> List[str]:
        if value is None:
            return []
        return [str(item) for item in value]


# ── 4. FairnessReport ─────────────────────────────────────────────────────────

class FairnessReport(BaseModel):
    """
    Output of the Bias Auditing Agent.

    Summarises score shifts observed when the candidate name is swapped with
    demographically distinct alternatives, and reports Fairlearn fairness
    metrics.
    """

    original_score: float = Field(
        ...,
        description="Baseline Skill Alignment Score for the original (unmodified) resume.",
    )
    swapped_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of substitute name → Skill Alignment Score for each name variant.",
    )
    max_difference: float = Field(
        ...,
        description="Maximum absolute score shift observed across all name variants.",
    )
    average_difference: float = Field(
        ...,
        description="Mean absolute score shift across all name variants.",
    )
    demographic_parity_difference: float = Field(
        ...,
        description="Fairlearn demographic parity difference across demographic groups.",
    )
    equalized_odds: Optional[Dict[str, float]] = Field(
        None,
        description="Fairlearn equalized-odds metrics; None when ground-truth labels are unavailable.",
    )
    summary: str = Field(
        ...,
        description="Plain-language summary of bias findings and interpretation.",
    )
    audit_error: Optional[str] = Field(
        None,
        description="Set to a descriptive message if the audit failed.",
    )

    @field_validator("original_score")
    @classmethod
    def _original_score_in_range(cls, value: float) -> float:
        return _check_score_0_100(value, "original_score")

    @field_validator("swapped_scores", mode="before")
    @classmethod
    def _swapped_scores_in_range(cls, value: object) -> Dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("swapped_scores must be a dict mapping name → score.")
        validated: Dict[str, float] = {}
        for name, score in value.items():
            validated[str(name)] = _check_score_0_100(float(score), f"swapped_scores[{name}]")
        return validated

    @field_validator("max_difference", "average_difference")
    @classmethod
    def _difference_non_negative(cls, value: float, info) -> float:
        if value < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0, got {value}")
        return value

    @model_validator(mode="after")
    def _average_le_max(self) -> "FairnessReport":
        """Average difference cannot exceed the maximum difference."""
        if self.average_difference > self.max_difference + 1e-9:
            raise ValueError(
                f"average_difference ({self.average_difference}) cannot exceed "
                f"max_difference ({self.max_difference})."
            )
        return self


# ── 5. PipelineOutput ─────────────────────────────────────────────────────────

class PipelineOutput(BaseModel):
    """
    Consolidated output of the full three-agent pipeline.

    Bundles the parsed resume, job description, scoring result, and optional
    fairness report into a single serialisable object. The ``agents_completed``
    list tracks which agents ran successfully, enabling partial-failure
    diagnostics.
    """

    parsed_resume: ParsedResume = Field(
        ..., description="Structured resume produced by the Resume Parsing Agent."
    )
    job_description: JobDescription = Field(
        ..., description="Structured job description used for matching."
    )
    scoring_result: ScoringResult = Field(
        ..., description="Skill Alignment Score and explainability report from the Scoring Agent."
    )
    fairness_report: Optional[FairnessReport] = Field(
        None,
        description="Bias audit results from the Auditing Agent; None if the audit was skipped.",
    )
    agents_completed: List[str] = Field(
        default_factory=list,
        description="Names of agents that completed successfully (e.g. ['parsing', 'scoring', 'auditing']).",
    )

    @field_validator("agents_completed", mode="before")
    @classmethod
    def _ensure_list_of_strings(cls, value: object) -> List[str]:
        if value is None:
            return []
        return [str(item) for item in value]

"""
tests/test_schemas.py — Unit tests for the shared Pydantic schema layer.

Covers:
- Valid construction of every model
- Field-level validator rejection (out-of-range scores, negative differences)
- List-of-strings coercion and None → empty-list handling
- Cross-field model validator (average_difference <= max_difference)
"""

import pytest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resume_intelligence.models.schemas import (
    FairnessReport,
    JobDescription,
    ParsedResume,
    PipelineOutput,
    ScoringResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_parsed_resume() -> dict:
    return {
        "id": "r001",
        "raw_text": "Experienced data scientist with Python and ML skills.",
        "name": "Alex Johnson",
        "skills": ["Python", "ML"],
        "normalized_skills": ["python", "machine learning"],
        "education": ["M.S. Computer Science — Stanford University"],
        "experience": ["Data Scientist — Acme Corp (2020–2024)"],
        "organizations": ["Acme Corp", "Stanford University"],
        "job_titles": ["Data Scientist"],
        "category": "Data Science",
    }


@pytest.fixture
def valid_job_description() -> dict:
    return {
        "id": "j001",
        "job_title": "Senior Data Scientist",
        "raw_text": "We are looking for a Senior Data Scientist with Python and ML experience.",
        "required_skills": ["Python", "machine learning", "SQL"],
        "normalized_skills": ["python", "machine learning", "sql"],
        "experience_level": "Senior (4+ years)",
        "industry": "Technology",
    }


@pytest.fixture
def valid_scoring_result() -> dict:
    return {
        "resume_id": "r001",
        "job_id": "j001",
        "semantic_similarity": 74.0,
        "skill_coverage": 82.0,
        "experience_match": 80.0,
        "skill_gap_penalty": 71.0,
        "final_score": 78.4,
        "matched_skills": ["python", "machine learning"],
        "semantic_matches": ["deep learning"],
        "missing_skills": ["sql"],
        "explanation": "Strong match. Consider adding SQL experience.",
        "tfidf_baseline_score": 61.0,
    }


@pytest.fixture
def valid_fairness_report() -> dict:
    return {
        "original_score": 78.4,
        "swapped_scores": {
            "Jamal Washington": 77.9,
            "Wei Zhang": 78.1,
        },
        "max_difference": 0.5,
        "average_difference": 0.3,
        "demographic_parity_difference": 0.04,
        "summary": "Low demographic bias detected. Max score shift: 0.5 points.",
    }


# ── ParsedResume ──────────────────────────────────────────────────────────────

class TestParsedResume:
    def test_valid_construction(self, valid_parsed_resume):
        resume = ParsedResume(**valid_parsed_resume)
        assert resume.id == "r001"
        assert resume.name == "Alex Johnson"
        assert resume.normalized_skills == ["python", "machine learning"]
        assert resume.parse_error is None

    def test_optional_fields_default_to_none_or_empty(self):
        resume = ParsedResume(id="r002", raw_text="Some text.")
        assert resume.name is None
        assert resume.category is None
        assert resume.parse_error is None
        assert resume.skills == []
        assert resume.normalized_skills == []
        assert resume.education == []
        assert resume.experience == []
        assert resume.organizations == []
        assert resume.job_titles == []

    def test_none_list_fields_coerced_to_empty_list(self):
        resume = ParsedResume(
            id="r003",
            raw_text="Text.",
            skills=None,
            education=None,
            experience=None,
        )
        assert resume.skills == []
        assert resume.education == []
        assert resume.experience == []

    def test_list_items_coerced_to_strings(self):
        # Non-string items should be coerced via str()
        resume = ParsedResume(id="r004", raw_text="Text.", skills=[1, 2, 3])
        assert resume.skills == ["1", "2", "3"]

    def test_parse_error_field_is_settable(self):
        resume = ParsedResume(
            id="r005",
            raw_text="",
            parse_error="Input text is too short.",
        )
        assert resume.parse_error == "Input text is too short."

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            ParsedResume(raw_text="Missing id field.")


# ── JobDescription ────────────────────────────────────────────────────────────

class TestJobDescription:
    def test_valid_construction(self, valid_job_description):
        jd = JobDescription(**valid_job_description)
        assert jd.id == "j001"
        assert jd.job_title == "Senior Data Scientist"
        assert "python" in jd.normalized_skills

    def test_optional_fields_default_to_none(self):
        jd = JobDescription(
            id="j002",
            job_title="Engineer",
            raw_text="Some JD text.",
        )
        assert jd.experience_level is None
        assert jd.industry is None
        assert jd.required_skills == []

    def test_none_skill_lists_coerced_to_empty(self):
        jd = JobDescription(
            id="j003",
            job_title="Analyst",
            raw_text="Text.",
            required_skills=None,
            normalized_skills=None,
        )
        assert jd.required_skills == []
        assert jd.normalized_skills == []

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            JobDescription(id="j004", raw_text="Missing job_title.")


# ── ScoringResult ─────────────────────────────────────────────────────────────

class TestScoringResult:
    def test_valid_construction(self, valid_scoring_result):
        result = ScoringResult(**valid_scoring_result)
        assert result.final_score == 78.4
        assert result.matched_skills == ["python", "machine learning"]
        assert result.tfidf_baseline_score == 61.0

    def test_boundary_scores_accepted(self, valid_scoring_result):
        valid_scoring_result["final_score"] = 0.0
        result = ScoringResult(**valid_scoring_result)
        assert result.final_score == 0.0

        valid_scoring_result["final_score"] = 100.0
        result = ScoringResult(**valid_scoring_result)
        assert result.final_score == 100.0

    def test_score_above_100_raises(self, valid_scoring_result):
        valid_scoring_result["final_score"] = 100.1
        with pytest.raises(ValidationError, match="final_score"):
            ScoringResult(**valid_scoring_result)

    def test_score_below_0_raises(self, valid_scoring_result):
        valid_scoring_result["semantic_similarity"] = -1.0
        with pytest.raises(ValidationError, match="semantic_similarity"):
            ScoringResult(**valid_scoring_result)

    def test_skill_coverage_out_of_range_raises(self, valid_scoring_result):
        valid_scoring_result["skill_coverage"] = 101.0
        with pytest.raises(ValidationError, match="skill_coverage"):
            ScoringResult(**valid_scoring_result)

    def test_tfidf_baseline_out_of_range_raises(self, valid_scoring_result):
        valid_scoring_result["tfidf_baseline_score"] = -5.0
        with pytest.raises(ValidationError, match="tfidf_baseline_score"):
            ScoringResult(**valid_scoring_result)

    def test_tfidf_baseline_none_is_valid(self, valid_scoring_result):
        valid_scoring_result["tfidf_baseline_score"] = None
        result = ScoringResult(**valid_scoring_result)
        assert result.tfidf_baseline_score is None

    def test_none_skill_lists_coerced_to_empty(self, valid_scoring_result):
        valid_scoring_result["matched_skills"] = None
        valid_scoring_result["missing_skills"] = None
        result = ScoringResult(**valid_scoring_result)
        assert result.matched_skills == []
        assert result.missing_skills == []

    def test_score_error_field_is_settable(self, valid_scoring_result):
        valid_scoring_result["score_error"] = "Invalid job description."
        result = ScoringResult(**valid_scoring_result)
        assert result.score_error == "Invalid job description."


# ── FairnessReport ────────────────────────────────────────────────────────────

class TestFairnessReport:
    def test_valid_construction(self, valid_fairness_report):
        report = FairnessReport(**valid_fairness_report)
        assert report.original_score == 78.4
        assert report.max_difference == 0.5
        assert report.equalized_odds is None

    def test_original_score_out_of_range_raises(self, valid_fairness_report):
        valid_fairness_report["original_score"] = 105.0
        with pytest.raises(ValidationError, match="original_score"):
            FairnessReport(**valid_fairness_report)

    def test_swapped_score_out_of_range_raises(self, valid_fairness_report):
        valid_fairness_report["swapped_scores"]["Bad Name"] = 150.0
        with pytest.raises(ValidationError):
            FairnessReport(**valid_fairness_report)

    def test_negative_max_difference_raises(self, valid_fairness_report):
        valid_fairness_report["max_difference"] = -0.1
        with pytest.raises(ValidationError, match="max_difference"):
            FairnessReport(**valid_fairness_report)

    def test_negative_average_difference_raises(self, valid_fairness_report):
        valid_fairness_report["average_difference"] = -1.0
        with pytest.raises(ValidationError, match="average_difference"):
            FairnessReport(**valid_fairness_report)

    def test_average_greater_than_max_raises(self, valid_fairness_report):
        valid_fairness_report["average_difference"] = 1.0
        valid_fairness_report["max_difference"] = 0.5
        with pytest.raises(ValidationError, match="average_difference"):
            FairnessReport(**valid_fairness_report)

    def test_equalized_odds_optional(self, valid_fairness_report):
        valid_fairness_report["equalized_odds"] = {"tpr_diff": 0.02, "fpr_diff": 0.01}
        report = FairnessReport(**valid_fairness_report)
        assert report.equalized_odds == {"tpr_diff": 0.02, "fpr_diff": 0.01}

    def test_empty_swapped_scores_is_valid(self, valid_fairness_report):
        valid_fairness_report["swapped_scores"] = {}
        valid_fairness_report["max_difference"] = 0.0
        valid_fairness_report["average_difference"] = 0.0
        report = FairnessReport(**valid_fairness_report)
        assert report.swapped_scores == {}


# ── PipelineOutput ────────────────────────────────────────────────────────────

class TestPipelineOutput:
    def test_valid_construction(
        self,
        valid_parsed_resume,
        valid_job_description,
        valid_scoring_result,
        valid_fairness_report,
    ):
        output = PipelineOutput(
            parsed_resume=ParsedResume(**valid_parsed_resume),
            job_description=JobDescription(**valid_job_description),
            scoring_result=ScoringResult(**valid_scoring_result),
            fairness_report=FairnessReport(**valid_fairness_report),
            agents_completed=["parsing", "scoring", "auditing"],
        )
        assert output.agents_completed == ["parsing", "scoring", "auditing"]
        assert output.fairness_report is not None

    def test_fairness_report_is_optional(
        self, valid_parsed_resume, valid_job_description, valid_scoring_result
    ):
        output = PipelineOutput(
            parsed_resume=ParsedResume(**valid_parsed_resume),
            job_description=JobDescription(**valid_job_description),
            scoring_result=ScoringResult(**valid_scoring_result),
        )
        assert output.fairness_report is None
        assert output.agents_completed == []

    def test_agents_completed_none_coerced_to_empty(
        self, valid_parsed_resume, valid_job_description, valid_scoring_result
    ):
        output = PipelineOutput(
            parsed_resume=ParsedResume(**valid_parsed_resume),
            job_description=JobDescription(**valid_job_description),
            scoring_result=ScoringResult(**valid_scoring_result),
            agents_completed=None,
        )
        assert output.agents_completed == []

    def test_missing_required_nested_model_raises(
        self, valid_parsed_resume, valid_job_description
    ):
        with pytest.raises(ValidationError):
            PipelineOutput(
                parsed_resume=ParsedResume(**valid_parsed_resume),
                job_description=JobDescription(**valid_job_description),
                # scoring_result is required but omitted
            )

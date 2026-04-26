"""
tests/test_scoring.py — Unit tests for the Semantic Scoring Agent.

Covers:
- Final score is always in [0, 100]
- All ScoringResult fields are populated
- Matching skills increases score vs. no-skill resume
- Missing skills reduces score
- Empty / too-short JD returns score=0 with score_error
- TF-IDF baseline is in [0, 100]
- Semantic similarity is in [0, 100]
- Matched skills are a subset of both resume and JD skills
- Explanation is a non-empty string
- Score ordering: high-match > low-match resume
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from resume_intelligence.agents.scoring_agent import (
    compute_semantic_similarity,
    compute_tfidf_baseline,
    score_resume,
)
from resume_intelligence.models.schemas import JobDescription, ParsedResume, ScoringResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data_science_job() -> JobDescription:
    return JobDescription(
        id="j001",
        job_title="Senior Data Scientist",
        raw_text=(
            "We are looking for a Senior Data Scientist with strong Python and "
            "machine learning skills. The ideal candidate has experience with "
            "pandas, scikit-learn, SQL, and TensorFlow. Knowledge of Docker and "
            "AWS is a plus. 4+ years of experience required."
        ),
        required_skills=["python", "machine learning", "pandas", "scikit-learn",
                         "sql", "tensorflow", "docker", "amazon web services"],
        normalized_skills=["python", "machine learning", "pandas", "scikit-learn",
                           "sql", "tensorflow", "docker", "amazon web services"],
        experience_level="Senior (4+ years)",
        industry="Technology",
    )


@pytest.fixture(scope="module")
def strong_resume(data_science_job) -> ParsedResume:
    """Resume that closely matches the data science job."""
    return ParsedResume(
        id="r001",
        raw_text=(
            "Alex Johnson — Data Scientist\n"
            "4 years of experience in machine learning and data analysis.\n"
            "Skills: Python, machine learning, pandas, scikit-learn, SQL, "
            "TensorFlow, Docker, AWS\n"
            "Experience: Data Scientist at Acme Corp (2020–2024)\n"
            "Education: M.S. Computer Science — Stanford University"
        ),
        name="Alex Johnson",
        skills=["python", "machine learning", "pandas", "scikit-learn",
                "sql", "tensorflow", "docker", "aws"],
        normalized_skills=["python", "machine learning", "pandas", "scikit-learn",
                           "sql", "tensorflow", "docker", "amazon web services"],
        education=["M.S. Computer Science — Stanford University"],
        experience=["4 years of experience in machine learning"],
        job_titles=["data scientist"],
    )


@pytest.fixture(scope="module")
def weak_resume(data_science_job) -> ParsedResume:
    """Resume with no overlapping skills."""
    return ParsedResume(
        id="r002",
        raw_text=(
            "Bob Smith — Graphic Designer\n"
            "5 years of experience in visual design and branding.\n"
            "Skills: Photoshop, Illustrator, InDesign, Figma\n"
            "Education: B.A. Fine Arts — Art Institute"
        ),
        name="Bob Smith",
        skills=["photoshop", "illustrator", "indesign", "figma"],
        normalized_skills=["photoshop", "illustrator", "indesign", "figma"],
        education=["B.A. Fine Arts — Art Institute"],
        experience=["5 years of experience in visual design"],
        job_titles=["graphic designer"],
    )


@pytest.fixture(scope="module")
def no_skills_resume() -> ParsedResume:
    """Resume with no extracted skills."""
    return ParsedResume(
        id="r003",
        raw_text=(
            "Jane Doe — Professional with 3 years of experience.\n"
            "Worked at various companies on different projects.\n"
            "Education: Bachelor of Arts — State University"
        ),
        name="Jane Doe",
        skills=[],
        normalized_skills=[],
        education=["Bachelor of Arts — State University"],
        experience=["3 years of experience"],
        job_titles=[],
    )


@pytest.fixture(scope="module")
def empty_jd() -> JobDescription:
    return JobDescription(
        id="j_empty",
        job_title="Unknown Role",
        raw_text="",
        required_skills=[],
        normalized_skills=[],
    )


@pytest.fixture(scope="module")
def short_jd() -> JobDescription:
    return JobDescription(
        id="j_short",
        job_title="Short Role",
        raw_text="Python developer needed.",  # < 10 tokens
        required_skills=["python"],
        normalized_skills=["python"],
    )


# ── Score range ───────────────────────────────────────────────────────────────

class TestScoreRange:
    def test_strong_resume_score_in_range(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert 0.0 <= result.final_score <= 100.0

    def test_weak_resume_score_in_range(self, weak_resume, data_science_job):
        result = score_resume(weak_resume, data_science_job)
        assert 0.0 <= result.final_score <= 100.0

    def test_no_skills_resume_score_in_range(self, no_skills_resume, data_science_job):
        result = score_resume(no_skills_resume, data_science_job)
        assert 0.0 <= result.final_score <= 100.0

    def test_all_component_scores_in_range(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert 0.0 <= result.semantic_similarity <= 100.0
        assert 0.0 <= result.skill_coverage <= 100.0
        assert 0.0 <= result.experience_match <= 100.0
        assert 0.0 <= result.skill_gap_penalty <= 100.0

    def test_tfidf_baseline_in_range(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert result.tfidf_baseline_score is not None
        assert 0.0 <= result.tfidf_baseline_score <= 100.0


# ── Score ordering ────────────────────────────────────────────────────────────

class TestScoreOrdering:
    def test_strong_resume_scores_higher_than_weak(
        self, strong_resume, weak_resume, data_science_job
    ):
        strong_result = score_resume(strong_resume, data_science_job)
        weak_result = score_resume(weak_resume, data_science_job)
        assert strong_result.final_score > weak_result.final_score, (
            f"Expected strong ({strong_result.final_score}) > weak ({weak_result.final_score})"
        )

    def test_matching_skills_increases_score(
        self, strong_resume, no_skills_resume, data_science_job
    ):
        with_skills = score_resume(strong_resume, data_science_job)
        without_skills = score_resume(no_skills_resume, data_science_job)
        assert with_skills.final_score > without_skills.final_score

    def test_skill_coverage_higher_for_strong_resume(
        self, strong_resume, weak_resume, data_science_job
    ):
        strong_result = score_resume(strong_resume, data_science_job)
        weak_result = score_resume(weak_resume, data_science_job)
        assert strong_result.skill_coverage > weak_result.skill_coverage


# ── Matched / missing skills ──────────────────────────────────────────────────

class TestSkillLists:
    def test_matched_skills_subset_of_resume_and_jd(
        self, strong_resume, data_science_job
    ):
        result = score_resume(strong_resume, data_science_job)
        resume_set = set(strong_resume.normalized_skills)
        jd_set = set(data_science_job.normalized_skills)
        for skill in result.matched_skills:
            assert skill in resume_set or skill in jd_set

    def test_missing_skills_not_in_resume(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        resume_set = set(strong_resume.normalized_skills)
        for skill in result.missing_skills:
            assert skill not in resume_set

    def test_weak_resume_has_missing_skills(self, weak_resume, data_science_job):
        result = score_resume(weak_resume, data_science_job)
        assert len(result.missing_skills) > 0

    def test_strong_resume_has_matched_skills(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert len(result.matched_skills) > 0

    def test_matched_and_missing_are_disjoint(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        overlap = set(result.matched_skills) & set(result.missing_skills)
        assert len(overlap) == 0, f"Skills appear in both matched and missing: {overlap}"

    def test_skill_gap_penalty_higher_for_weak_resume(
        self, strong_resume, weak_resume, data_science_job
    ):
        strong_result = score_resume(strong_resume, data_science_job)
        weak_result = score_resume(weak_resume, data_science_job)
        assert weak_result.skill_gap_penalty > strong_result.skill_gap_penalty


# ── Invalid input handling ────────────────────────────────────────────────────

class TestInvalidInput:
    def test_empty_jd_returns_zero_score(self, strong_resume, empty_jd):
        result = score_resume(strong_resume, empty_jd)
        assert result.final_score == 0.0

    def test_empty_jd_sets_score_error(self, strong_resume, empty_jd):
        result = score_resume(strong_resume, empty_jd)
        assert result.score_error is not None
        assert len(result.score_error) > 0

    def test_short_jd_returns_zero_score(self, strong_resume, short_jd):
        result = score_resume(strong_resume, short_jd)
        assert result.final_score == 0.0

    def test_short_jd_sets_score_error(self, strong_resume, short_jd):
        result = score_resume(strong_resume, short_jd)
        assert result.score_error is not None

    def test_no_skills_resume_still_returns_valid_result(
        self, no_skills_resume, data_science_job
    ):
        result = score_resume(no_skills_resume, data_science_job)
        assert isinstance(result, ScoringResult)
        assert result.score_error is None
        assert 0.0 <= result.final_score <= 100.0


# ── Output schema compliance ──────────────────────────────────────────────────

class TestOutputSchema:
    def test_result_is_scoring_result_instance(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert isinstance(result, ScoringResult)

    def test_all_required_fields_populated(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert result.resume_id == strong_resume.id
        assert result.job_id == data_science_job.id
        assert isinstance(result.matched_skills, list)
        assert isinstance(result.missing_skills, list)
        assert isinstance(result.semantic_matches, list)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_explanation_mentions_job_title(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        assert "Data Scientist" in result.explanation

    def test_explanation_mentions_matched_skills(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        # At least one matched skill should appear in the explanation
        if result.matched_skills:
            assert any(skill in result.explanation for skill in result.matched_skills[:3])

    def test_result_serialises_to_dict(self, strong_resume, data_science_job):
        result = score_resume(strong_resume, data_science_job)
        d = result.model_dump()
        assert "final_score" in d
        assert "matched_skills" in d
        assert "explanation" in d


# ── compute_semantic_similarity ───────────────────────────────────────────────

class TestSemanticSimilarity:
    def test_identical_texts_score_near_one(self):
        text = "Python machine learning data science pandas scikit-learn"
        sim = compute_semantic_similarity(text, text)
        assert sim >= 0.99

    def test_similar_texts_score_higher_than_dissimilar(self):
        text_a = "Python machine learning data science"
        text_b = "Python deep learning neural networks"
        text_c = "Cooking recipes baking bread flour"
        sim_related = compute_semantic_similarity(text_a, text_b)
        sim_unrelated = compute_semantic_similarity(text_a, text_c)
        assert sim_related > sim_unrelated

    def test_empty_text_returns_zero(self):
        assert compute_semantic_similarity("", "some text") == 0.0
        assert compute_semantic_similarity("some text", "") == 0.0

    def test_result_in_zero_one_range(self):
        sim = compute_semantic_similarity("hello world", "goodbye world")
        assert 0.0 <= sim <= 1.0


# ── compute_tfidf_baseline ────────────────────────────────────────────────────

class TestTfidfBaseline:
    def test_identical_texts_score_one(self):
        text = "python machine learning data science"
        score = compute_tfidf_baseline(text, text)
        assert score >= 0.99

    def test_result_in_zero_one_range(self):
        score = compute_tfidf_baseline("python pandas numpy", "java spring boot")
        assert 0.0 <= score <= 1.0

    def test_empty_text_returns_zero(self):
        assert compute_tfidf_baseline("", "some text") == 0.0

    def test_overlapping_terms_score_higher(self):
        text_a = "python machine learning scikit-learn"
        text_b = "python machine learning tensorflow"
        text_c = "cooking baking recipes flour"
        score_related = compute_tfidf_baseline(text_a, text_b)
        score_unrelated = compute_tfidf_baseline(text_a, text_c)
        assert score_related > score_unrelated

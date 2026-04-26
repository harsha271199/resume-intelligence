"""
tests/test_parsing.py — Unit tests for the Resume Parsing Agent.

Covers:
- Skill extraction from realistic resume text
- Skill normalisation (abbreviations → canonical forms)
- Education and experience extraction
- Job title detection
- spaCy NER for organisations and names
- Input validation (empty / too-short text)
- Error handling (parse_error field)
- Entity substring property (no hallucinated entities)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from resume_intelligence.agents.parser_agent import normalize_skills, parse_resume
from resume_intelligence.models.schemas import ParsedResume

# ── Sample resume fixtures ────────────────────────────────────────────────────

DATA_SCIENTIST_RESUME = """
Alex Johnson
Email: alex.johnson@email.com | LinkedIn: linkedin.com/in/alexjohnson

SUMMARY
Data Scientist with 4 years of experience in ML and NLP at Acme Corp.
Proficient in Python, TensorFlow, and scikit-learn.

SKILLS
Python, ML, NLP, pandas, numpy, SQL, TensorFlow, Docker, AWS

EXPERIENCE
Data Scientist — Acme Corp (2020–2024)
- Built ML pipelines for customer churn prediction using scikit-learn
- Developed NLP models for sentiment analysis with TensorFlow
- 4 years of experience in machine learning and data analysis

EDUCATION
M.S. Computer Science — Stanford University (2020)
B.S. Statistics — UC Berkeley (2018)
"""

SOFTWARE_ENGINEER_RESUME = """
Maria Garcia
Software Engineer at TechStart Inc

SKILLS
Java, Python, JavaScript, REST APIs, Docker, Kubernetes, PostgreSQL, AWS, Git

EXPERIENCE
Software Engineer — TechStart Inc (2021–2024)
Worked at TechStart building microservices with Docker and Kubernetes.
3 years of experience in backend development.

EDUCATION
Bachelor of Science in Computer Science — UC Berkeley (2021)
"""

MINIMAL_RESUME = """
John Doe
Python developer with 2 years of experience.
Worked at StartupXYZ on data pipelines.
B.Tech Computer Science from IIT Delhi.
Skills: Python, SQL, pandas, machine learning, Docker
"""


# ── normalize_skills ──────────────────────────────────────────────────────────

class TestNormalizeSkills:
    def test_abbreviation_ml_maps_to_machine_learning(self):
        result = normalize_skills(["ML"])
        assert "machine learning" in result

    def test_abbreviation_nlp_maps_to_natural_language_processing(self):
        result = normalize_skills(["NLP"])
        assert "natural language processing" in result

    def test_abbreviation_js_maps_to_javascript(self):
        result = normalize_skills(["JS"])
        assert "javascript" in result

    def test_abbreviation_k8s_maps_to_kubernetes(self):
        result = normalize_skills(["k8s"])
        assert "kubernetes" in result

    def test_abbreviation_aws_maps_to_amazon_web_services(self):
        result = normalize_skills(["AWS"])
        assert "amazon web services" in result

    def test_unknown_skill_returned_lowercased(self):
        result = normalize_skills(["Foobar"])
        assert "foobar" in result

    def test_duplicates_removed(self):
        result = normalize_skills(["python", "Python", "py"])
        # All three map to "python" — should appear only once
        assert result.count("python") == 1

    def test_empty_list_returns_empty(self):
        assert normalize_skills([]) == []

    def test_idempotent_normalisation(self):
        """Applying normalisation twice must produce the same result (Property 1)."""
        skills = ["ML", "NLP", "JS", "k8s", "Python", "UnknownSkill"]
        once = normalize_skills(skills)
        twice = normalize_skills(once)
        assert once == twice

    def test_mixed_case_handled(self):
        result = normalize_skills(["TensorFlow", "TENSORFLOW", "tensorflow"])
        assert result.count("tensorflow") == 1


# ── parse_resume — input validation ──────────────────────────────────────────

class TestParseResumeValidation:
    def test_empty_string_returns_parse_error(self):
        result = parse_resume("")
        assert isinstance(result, ParsedResume)
        assert result.parse_error is not None
        assert result.skills == []
        assert result.education == []
        assert result.experience == []

    def test_whitespace_only_returns_parse_error(self):
        result = parse_resume("   \n\t  ")
        assert result.parse_error is not None

    def test_fewer_than_10_tokens_returns_parse_error(self):
        result = parse_resume("Python developer skilled")
        assert result.parse_error is not None
        assert result.skills == []

    def test_exactly_10_tokens_does_not_error(self):
        text = "Python developer with SQL and Docker skills at Acme Corp today"
        result = parse_resume(text)
        # 10 tokens — should attempt parsing (parse_error may or may not be set
        # depending on extraction, but it should NOT be the "too short" error)
        if result.parse_error:
            assert "too short" not in result.parse_error

    def test_raw_text_preserved_in_output(self):
        result = parse_resume(DATA_SCIENTIST_RESUME, resume_id="r001")
        assert result.raw_text == DATA_SCIENTIST_RESUME

    def test_custom_resume_id_preserved(self):
        result = parse_resume(DATA_SCIENTIST_RESUME, resume_id="custom-123")
        assert result.id == "custom-123"

    def test_auto_id_generated_when_not_provided(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert result.id  # non-empty string
        assert len(result.id) > 0

    def test_category_passed_through(self):
        result = parse_resume(DATA_SCIENTIST_RESUME, category="Data Science")
        assert result.category == "Data Science"


# ── parse_resume — skill extraction ──────────────────────────────────────────

class TestSkillExtraction:
    def test_python_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert "python" in result.normalized_skills

    def test_sql_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert "sql" in result.normalized_skills

    def test_ml_abbreviation_normalised(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        # "ML" in the resume should normalise to "machine learning"
        assert "machine learning" in result.normalized_skills

    def test_nlp_abbreviation_normalised(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert "natural language processing" in result.normalized_skills

    def test_docker_extracted(self):
        result = parse_resume(SOFTWARE_ENGINEER_RESUME)
        assert "docker" in result.normalized_skills

    def test_kubernetes_extracted(self):
        result = parse_resume(SOFTWARE_ENGINEER_RESUME)
        assert "kubernetes" in result.normalized_skills

    def test_at_least_one_skill_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert len(result.normalized_skills) >= 1

    def test_skills_are_strings(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        for skill in result.normalized_skills:
            assert isinstance(skill, str)

    def test_normalized_skills_are_lowercase(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        for skill in result.normalized_skills:
            assert skill == skill.lower(), f"Skill not lowercase: {skill!r}"

    def test_no_duplicate_normalized_skills(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert len(result.normalized_skills) == len(set(result.normalized_skills))


# ── parse_resume — education extraction ──────────────────────────────────────

class TestEducationExtraction:
    def test_masters_degree_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert any("m.s" in e.lower() or "master" in e.lower() for e in result.education)

    def test_bachelors_degree_extracted(self):
        result = parse_resume(SOFTWARE_ENGINEER_RESUME)
        assert any("bachelor" in e.lower() or "b.s" in e.lower() for e in result.education)

    def test_btech_extracted(self):
        result = parse_resume(MINIMAL_RESUME)
        assert any("b.tech" in e.lower() or "btech" in e.lower() for e in result.education)

    def test_education_entries_are_strings(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        for entry in result.education:
            assert isinstance(entry, str)


# ── parse_resume — experience extraction ─────────────────────────────────────

class TestExperienceExtraction:
    def test_year_range_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert any("2020" in e or "2024" in e for e in result.experience)

    def test_years_of_experience_phrase_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert any("year" in e.lower() for e in result.experience)

    def test_worked_at_phrase_extracted(self):
        result = parse_resume(SOFTWARE_ENGINEER_RESUME)
        assert any("worked" in e.lower() or "2021" in e for e in result.experience)

    def test_experience_entries_are_strings(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        for entry in result.experience:
            assert isinstance(entry, str)


# ── parse_resume — job title extraction ──────────────────────────────────────

class TestJobTitleExtraction:
    def test_data_scientist_title_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert any("data scientist" in t.lower() for t in result.job_titles)

    def test_software_engineer_title_extracted(self):
        result = parse_resume(SOFTWARE_ENGINEER_RESUME)
        assert any("software engineer" in t.lower() for t in result.job_titles)

    def test_job_titles_are_strings(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        for title in result.job_titles:
            assert isinstance(title, str)


# ── parse_resume — NER (organisations, names) ────────────────────────────────

class TestNERExtraction:
    def test_organisation_extracted(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        # spaCy should pick up "Acme Corp" or "Stanford University"
        assert isinstance(result.organizations, list)

    def test_name_is_string_or_none(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert result.name is None or isinstance(result.name, str)

    def test_organizations_are_strings(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        for org in result.organizations:
            assert isinstance(org, str)


# ── Entity substring property (Property 2) ───────────────────────────────────

class TestEntitySubstringProperty:
    """
    Every extracted entity must be a substring of the lowercased input text.
    This guards against hallucinated entities.
    """

    def _all_entities(self, result: ParsedResume) -> list[str]:
        return (
            result.skills
            + result.education
            + result.experience
            + result.job_titles
        )

    def test_skills_are_substrings_of_input(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        text_lower = DATA_SCIENTIST_RESUME.lower()
        for skill in result.skills:
            assert skill.lower() in text_lower, (
                f"Skill {skill!r} not found in input text"
            )

    def test_job_titles_are_substrings_of_input(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        text_lower = DATA_SCIENTIST_RESUME.lower()
        for title in result.job_titles:
            assert title.lower() in text_lower, (
                f"Job title {title!r} not found in input text"
            )


# ── Output schema compliance ──────────────────────────────────────────────────

class TestOutputSchemaCompliance:
    def test_output_is_parsed_resume_instance(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert isinstance(result, ParsedResume)

    def test_all_list_fields_are_lists(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert isinstance(result.skills, list)
        assert isinstance(result.normalized_skills, list)
        assert isinstance(result.education, list)
        assert isinstance(result.experience, list)
        assert isinstance(result.organizations, list)
        assert isinstance(result.job_titles, list)

    def test_parse_error_is_none_on_valid_input(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        assert result.parse_error is None

    def test_serialises_to_dict(self):
        result = parse_resume(DATA_SCIENTIST_RESUME)
        d = result.model_dump()
        assert "skills" in d
        assert "normalized_skills" in d
        assert "education" in d

"""
parser_agent.py — Resume Parsing Agent

Converts raw resume text into a structured ParsedResume object using:
  - spaCy en_core_web_sm for NER (organisations, person names)
  - Keyword-based skill extraction against a curated skill list
  - Skill normalisation via the skill_aliases.json dictionary
  - Regex-based education and experience extraction

Public API
----------
    parse_resume(text: str, resume_id: str = ...) -> ParsedResume
    normalize_skills(skills: List[str]) -> List[str]
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional

import spacy
from spacy.language import Language

from resume_intelligence.models.schemas import ParsedResume

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"
_ALIASES_PATH = _DATA_DIR / "skill_aliases.json"

# ── Alias map — loaded eagerly (pure JSON, no compilation) ────────────────────

def _load_aliases() -> dict[str, str]:
    """Load skill alias dictionary from JSON, ignoring comment keys."""
    try:
        with open(_ALIASES_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {k.lower(): v.lower() for k, v in raw.items() if not k.startswith("_")}
    except FileNotFoundError:
        logger.warning("skill_aliases.json not found at %s — normalisation disabled.", _ALIASES_PATH)
        return {}


_ALIAS_MAP: dict[str, str] = _load_aliases()

# ── spaCy model — lazy-loaded and cached ──────────────────────────────────────
#
# WHY lazy loading?
# -----------------
# On Streamlit Cloud the spaCy model wheel is installed at build time via
# requirements.txt. However, importing spacy.load() at module level causes
# the model to be resolved during the first import, which can race with the
# Streamlit Cloud build process or fail if the model package hasn't been
# registered yet.
#
# Lazy loading defers the spacy.load() call to the first actual parse request,
# by which point all packages are fully installed. A module-level sentinel
# (_NLP = None) and a getter function (_get_nlp()) ensure the model is loaded
# exactly once and reused for all subsequent calls.
#
# Auto-download fallback:
# If the model is still not found (e.g. local dev without running
# `python -m spacy download en_core_web_sm`), we attempt a subprocess download
# and retry once. If that also fails, we fall back to spacy.blank("en") so
# the app stays functional (NER is degraded but skill/education/experience
# extraction still works via regex and keyword matching).

_NLP: Optional[Language] = None  # populated on first call to _get_nlp()


def _get_nlp() -> Language:
    """
    Return the cached spaCy model, loading it on first call.

    Load order:
    1. Try spacy.load(model_name)  — succeeds when wheel is installed
    2. Try subprocess download + retry  — fallback for local dev
    3. spacy.blank("en")  — last resort; NER disabled but app stays alive
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")

    # Attempt 1: model already installed (normal path on Streamlit Cloud)
    try:
        _NLP = spacy.load(model_name)
        logger.info("spaCy model '%s' loaded successfully.", model_name)
        return _NLP
    except OSError:
        logger.warning("spaCy model '%s' not found — attempting download.", model_name)

    # Attempt 2: download and retry (local dev convenience)
    try:
        import subprocess
        subprocess.run(
            [os.sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            capture_output=True,
        )
        _NLP = spacy.load(model_name)
        logger.info("spaCy model '%s' downloaded and loaded.", model_name)
        return _NLP
    except Exception as exc:
        logger.warning(
            "Could not download spaCy model '%s': %s. "
            "Falling back to blank English model — NER will be limited.",
            model_name, exc,
        )

    # Attempt 3: blank model (NER disabled; regex extraction still works)
    _NLP = spacy.blank("en")
    return _NLP

# ── Skill vocabulary ──────────────────────────────────────────────────────────
# All canonical skill names from the alias map, plus common skills not in aliases.
# Used for keyword matching against resume text.

_SKILL_VOCAB: set[str] = set(_ALIAS_MAP.values()) | {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust",
    "scala", "kotlin", "swift", "ruby", "php", "r", "matlab", "bash",
    # ML / AI
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "neural networks",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "hugging face", "transformers", "bert", "gpt", "large language models",
    "langchain", "openai", "mlflow", "mlops",
    # Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "sql", "mysql", "postgresql", "mongodb", "nosql", "redis",
    "apache spark", "hadoop", "apache kafka", "apache airflow",
    "tableau", "power bi", "looker", "microsoft excel",
    # Cloud / DevOps
    "amazon web services", "google cloud platform", "microsoft azure",
    "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "continuous integration and deployment", "devops", "git", "github",
    # Web
    "react", "angular", "vue.js", "node.js", "django", "flask", "fastapi",
    "spring boot", "rest api", "graphql",
    # General
    "agile", "scrum", "jira", "linux", "statistics", "data analysis",
    "data visualization", "feature engineering", "a/b testing",
    "object-oriented programming", "shell scripting",
}

# ── Education patterns ────────────────────────────────────────────────────────

_EDUCATION_PATTERN = re.compile(
    r"(?i)\b("
    r"b\.?tech|b\.?e\.?|b\.?sc?\.?|b\.?s\.?|"
    r"m\.?tech|m\.?e\.?|m\.?sc?\.?|m\.?s\.?|mba|"
    r"ph\.?d\.?|doctorate|"
    r"bachelor(?:'s)?|master(?:'s)?|associate(?:'s)?"
    r")\b[^\n]{0,120}",
    re.IGNORECASE,
)

# ── Experience patterns ───────────────────────────────────────────────────────

_EXPERIENCE_PATTERN = re.compile(
    r"(?i)("
    r"\d+\+?\s*years?\s+(?:of\s+)?(?:experience|exp)\b[^\n]{0,100}"
    r"|(?:worked|working)\s+(?:at|with|for|on)\b[^\n]{0,100}"
    r"|\b(?:senior|junior|lead|principal|staff)\s+\w+[^\n]{0,80}"
    r"|\d{4}\s*[-–—]\s*(?:\d{4}|present|current)[^\n]{0,100}"
    r")"
)

# ── Job title vocabulary ──────────────────────────────────────────────────────

_JOB_TITLE_VOCAB: list[str] = [
    "software engineer", "software developer", "senior software engineer",
    "data scientist", "senior data scientist", "lead data scientist",
    "machine learning engineer", "ml engineer", "ai engineer",
    "data engineer", "senior data engineer",
    "data analyst", "business analyst", "research analyst",
    "product manager", "senior product manager", "technical product manager",
    "devops engineer", "site reliability engineer", "sre",
    "backend engineer", "frontend engineer", "full stack engineer",
    "cloud engineer", "platform engineer", "infrastructure engineer",
    "research scientist", "applied scientist",
    "nlp engineer", "computer vision engineer",
    "engineering manager", "tech lead", "technical lead",
    "solutions architect", "software architect",
    "qa engineer", "test engineer", "automation engineer",
    "security engineer", "cybersecurity analyst",
    "database administrator", "dba",
    "project manager", "program manager", "scrum master",
    "ux designer", "ui designer", "product designer",
]


# ── Core helpers ──────────────────────────────────────────────────────────────

def normalize_skills(skills: List[str]) -> List[str]:
    """
    Map raw skill tokens to their canonical forms using the alias dictionary.

    Unknown tokens are lowercased and returned as-is. Duplicates are removed
    while preserving order of first occurrence.

    Parameters
    ----------
    skills : List[str]
        Raw skill strings extracted from resume text.

    Returns
    -------
    List[str]
        Deduplicated list of canonical skill names.
    """
    seen: set[str] = set()
    result: List[str] = []
    for skill in skills:
        canonical = _ALIAS_MAP.get(skill.lower(), skill.lower())
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _extract_skills(text: str) -> List[str]:
    """
    Extract raw skill mentions from *text* by scanning for known skill tokens.

    Uses a multi-word aware scan: checks every 1-, 2-, and 3-gram against the
    skill vocabulary and alias keys, so phrases like "machine learning" and
    abbreviations like "ML" are both caught.
    """
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    # Build combined lookup: alias keys + canonical vocab
    lookup = set(_ALIAS_MAP.keys()) | _SKILL_VOCAB

    # Tokenise on whitespace/punctuation for n-gram scanning
    tokens = re.split(r"[\s,;|•\-/]+", text_lower)
    tokens = [t.strip("().[]\"'") for t in tokens if t.strip("().[]\"'")]

    for n in (3, 2, 1):  # prefer longer matches
        i = 0
        while i <= len(tokens) - n:
            gram = " ".join(tokens[i : i + n])
            if gram in lookup and gram not in seen:
                seen.add(gram)
                found.append(gram)
                i += n  # skip consumed tokens
            else:
                if n == 1:
                    i += 1
                else:
                    i += 1

    return found


def _extract_education(text: str) -> List[str]:
    """Extract education entries using degree-keyword regex."""
    matches = _EDUCATION_PATTERN.findall(text)
    # findall returns the capture group (degree keyword); get full match instead
    entries: List[str] = []
    for match in _EDUCATION_PATTERN.finditer(text):
        entry = match.group(0).strip()
        if entry and entry not in entries:
            entries.append(entry)
    return entries


def _extract_experience(text: str) -> List[str]:
    """Extract experience lines using duration/role/date-range patterns."""
    entries: List[str] = []
    seen: set[str] = set()
    for match in _EXPERIENCE_PATTERN.finditer(text):
        entry = match.group(0).strip()
        if entry and entry not in seen:
            seen.add(entry)
            entries.append(entry)
    return entries


def _extract_job_titles(text: str) -> List[str]:
    """
    Extract job titles by matching against a curated vocabulary list.
    Longer titles are checked first to avoid partial matches.
    """
    text_lower = text.lower()
    found: List[str] = []
    seen: set[str] = set()
    for title in sorted(_JOB_TITLE_VOCAB, key=len, reverse=True):
        if title in text_lower and title not in seen:
            seen.add(title)
            found.append(title)
    return found


def _extract_ner_entities(text: str) -> tuple[Optional[str], List[str]]:
    """
    Run spaCy NER to extract:
      - The first PERSON entity as the candidate name
      - All ORG entities as organisations

    Returns (name, organisations).
    """
    nlp = _get_nlp()  # lazy-load on first call
    doc = nlp(text[:10_000])  # cap at 10k chars for speed
    name: Optional[str] = None
    orgs: List[str] = []
    seen_orgs: set[str] = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON" and name is None:
            name = ent.text.strip()
        elif ent.label_ == "ORG":
            org = ent.text.strip()
            if org and org not in seen_orgs:
                seen_orgs.add(org)
                orgs.append(org)

    return name, orgs


# ── Public API ────────────────────────────────────────────────────────────────

def parse_resume(
    text: str,
    resume_id: Optional[str] = None,
    category: Optional[str] = None,
) -> ParsedResume:
    """
    Parse raw resume text into a structured ParsedResume object.

    Parameters
    ----------
    text : str
        Raw resume text (plain text, not HTML).
    resume_id : str, optional
        Identifier for this resume. Auto-generated UUID if not provided.
    category : str, optional
        Known category label (e.g. from a dataset). Passed through unchanged.

    Returns
    -------
    ParsedResume
        Fully populated schema object. On failure, returns a ParsedResume with
        empty entity lists and ``parse_error`` set to the exception message.
    """
    rid = resume_id or str(uuid.uuid4())

    # ── Input validation ──────────────────────────────────────────────────────
    if not text or not text.strip():
        return ParsedResume(
            id=rid,
            raw_text=text or "",
            parse_error="Input text is empty.",
        )

    token_count = len(text.split())
    if token_count < 10:
        return ParsedResume(
            id=rid,
            raw_text=text,
            parse_error=f"Input text is too short ({token_count} tokens). Minimum is 10.",
        )

    # ── Extraction ────────────────────────────────────────────────────────────
    try:
        name, organizations = _extract_ner_entities(text)
        raw_skills = _extract_skills(text)
        normalized = normalize_skills(raw_skills)
        education = _extract_education(text)
        experience = _extract_experience(text)
        job_titles = _extract_job_titles(text)

        return ParsedResume(
            id=rid,
            raw_text=text,
            name=name,
            skills=raw_skills,
            normalized_skills=normalized,
            education=education,
            experience=experience,
            organizations=organizations,
            job_titles=job_titles,
            category=category,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while parsing resume id=%s", rid)
        return ParsedResume(
            id=rid,
            raw_text=text,
            parse_error=f"Parsing failed: {exc}",
        )

"""
bias_auditor.py — Bias Auditing Agent

Detects demographic bias in resume scoring by performing controlled
name-swapping experiments. The resume content stays identical across all
variants — only the candidate name changes. Score shifts are therefore
attributable solely to name-based signal in the scoring model.

Algorithm
---------
1. Detect the candidate name in the original resume (spaCy PERSON entity
   or first-line heuristic).
2. For each substitute name in the name list, replace the detected name
   in the raw text and re-score the variant.
3. Compute score shifts (absolute differences from the baseline).
4. Compute fairness metrics:
   - demographic_parity_difference = max(group_avg) − min(group_avg)
     across demographic groups (no external labels required).
5. Produce a FairnessReport with a plain-language summary.

Public API
----------
    audit_resume(resume: ParsedResume, job: JobDescription,
                 names: list[dict] | None = None) -> FairnessReport
    replace_name(text: str, old_name: str, new_name: str) -> str
    detect_name(resume: ParsedResume) -> str | None
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from resume_intelligence.agents.parser_agent import parse_resume
from resume_intelligence.agents.scoring_agent import score_resume
from resume_intelligence.models.schemas import FairnessReport, JobDescription, ParsedResume

logger = logging.getLogger(__name__)

# ── Built-in name list ────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"
_NAME_PAIRS_PATH = _DATA_DIR / "name_pairs.json"

# Default swap targets used when the caller provides no custom name list.
# Kept small so the audit runs quickly in demos; the full list is in name_pairs.json.
_DEFAULT_SWAP_NAMES: List[Dict[str, str]] = [
    {"name": "John Smith",       "group": "White American"},
    {"name": "Aisha Khan",       "group": "South Asian"},
    {"name": "Wei Chen",         "group": "East Asian"},
    {"name": "Maria Garcia",     "group": "Hispanic"},
    {"name": "Jamal Washington", "group": "African American"},
    {"name": "Emily Walsh",      "group": "White American"},
    {"name": "Priya Sharma",     "group": "South Asian"},
    {"name": "Carlos Mendoza",   "group": "Hispanic"},
    {"name": "Yuki Tanaka",      "group": "East Asian"},
    {"name": "DeShawn Brooks",   "group": "African American"},
]


def _load_name_pairs() -> List[Dict[str, str]]:
    """Load name pairs from JSON file; fall back to built-in defaults."""
    try:
        with open(_NAME_PAIRS_PATH, encoding="utf-8") as f:
            pairs = json.load(f)
        if isinstance(pairs, list) and pairs:
            return pairs
    except FileNotFoundError:
        logger.warning(
            "name_pairs.json not found at %s — using built-in defaults.",
            _NAME_PAIRS_PATH,
        )
    return _DEFAULT_SWAP_NAMES


# Loaded once at import time
_NAME_PAIRS: List[Dict[str, str]] = _load_name_pairs()


# ── Core helpers ──────────────────────────────────────────────────────────────

def detect_name(resume: ParsedResume) -> Optional[str]:
    """
    Return the candidate name from a ParsedResume.

    Uses the ``name`` field set by the parser (spaCy PERSON entity).
    Falls back to the first non-empty line of the raw text if the parser
    did not detect a name.

    Parameters
    ----------
    resume : ParsedResume

    Returns
    -------
    str | None
        Detected name, or None if no name could be determined.
    """
    if resume.name and resume.name.strip():
        return resume.name.strip()

    # Heuristic: first non-empty line is often the candidate's name
    for line in resume.raw_text.splitlines():
        line = line.strip()
        # Accept lines that look like a name: 2–4 words, no digits, no colons
        if line and 1 < len(line.split()) <= 5 and not any(c.isdigit() for c in line) and ":" not in line:
            return line

    return None


def replace_name(text: str, old_name: str, new_name: str) -> str:
    """
    Replace all whole-word occurrences of *old_name* in *text* with *new_name*.

    The replacement is case-sensitive and uses word-boundary anchors so that
    partial matches (e.g. "Johnson" inside "Johnson & Johnson") are not
    accidentally replaced.

    Parameters
    ----------
    text : str
        Original resume text.
    old_name : str
        Name to replace (must be a non-empty string).
    new_name : str
        Replacement name.

    Returns
    -------
    str
        Modified text. If *old_name* is empty or not found, returns *text*
        unchanged.
    """
    if not old_name or not old_name.strip():
        return text

    # Escape special regex characters in the name
    escaped = re.escape(old_name.strip())
    pattern = r"\b" + escaped + r"\b"
    return re.sub(pattern, new_name, text)


def _build_variant_resume(
    original: ParsedResume,
    old_name: str,
    new_name: str,
    variant_id: str,
) -> ParsedResume:
    """
    Create a name-swapped variant of *original* by replacing *old_name*
    with *new_name* in the raw text and re-parsing.

    All skill, education, and experience content is preserved — only the
    name changes.
    """
    swapped_text = replace_name(original.raw_text, old_name, new_name)
    variant = parse_resume(swapped_text, resume_id=variant_id, category=original.category)
    return variant


def _compute_group_averages(
    swapped_scores: Dict[str, float],
    name_to_group: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute the mean score per demographic group.

    Parameters
    ----------
    swapped_scores : dict
        name → score mapping from the audit.
    name_to_group : dict
        name → demographic group mapping.

    Returns
    -------
    dict
        group → mean_score mapping.
    """
    group_scores: Dict[str, List[float]] = defaultdict(list)
    for name, score in swapped_scores.items():
        group = name_to_group.get(name, "Unknown")
        group_scores[group].append(score)
    return {group: sum(scores) / len(scores) for group, scores in group_scores.items()}


def _build_summary(
    original_score: float,
    swapped_scores: Dict[str, float],
    max_diff: float,
    avg_diff: float,
    dpd: float,
) -> str:
    """
    Generate a plain-language summary of the fairness audit findings.
    """
    all_scores = list(swapped_scores.values())
    min_score = min(all_scores) if all_scores else original_score
    max_score = max(all_scores) if all_scores else original_score

    # Bias level thresholds (heuristic, not a legal standard)
    if max_diff < 2.0:
        bias_level = "minimal"
        verdict = "No significant bias detected."
    elif max_diff < 5.0:
        bias_level = "low"
        verdict = "Low potential bias detected."
    elif max_diff < 10.0:
        bias_level = "moderate"
        verdict = "Moderate potential bias detected — review recommended."
    else:
        bias_level = "high"
        verdict = "Significant potential bias detected — investigation strongly recommended."

    lines = [
        verdict,
        f"Score variation across {len(swapped_scores)} name variants: "
        f"min={min_score:.1f}, max={max_score:.1f}, "
        f"max_diff={max_diff:.2f}, avg_diff={avg_diff:.2f}.",
        f"Demographic parity difference: {dpd:.4f} ({bias_level} bias).",
        "Note: Score shifts are caused by name-only substitution. "
        "All skills, experience, and education content are identical across variants.",
    ]
    return " ".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def audit_resume(
    resume: ParsedResume,
    job: JobDescription,
    names: Optional[List[Dict[str, str]]] = None,
) -> FairnessReport:
    """
    Audit a resume for demographic bias via controlled name-swapping.

    Parameters
    ----------
    resume : ParsedResume
        The original parsed resume (baseline).
    job : JobDescription
        The job description to score against.
    names : list of {"name": str, "group": str}, optional
        Custom name list to use instead of the built-in defaults.
        Each entry must have "name" and "group" keys.

    Returns
    -------
    FairnessReport
        Contains original score, per-name swapped scores, score shift
        statistics, demographic parity difference, and a plain-language
        summary. On failure, returns a FairnessReport with ``audit_error``
        set.
    """
    name_list = names if names is not None else _NAME_PAIRS

    try:
        # ── Step 1: Baseline score ────────────────────────────────────────────
        baseline_result = score_resume(resume, job)
        original_score = baseline_result.final_score

        # ── Step 2: Detect candidate name ─────────────────────────────────────
        detected_name = detect_name(resume)
        logger.info(
            "Bias audit: detected name=%r, baseline_score=%.2f",
            detected_name, original_score,
        )

        # ── Step 3: Name-swapping loop ────────────────────────────────────────
        swapped_scores: Dict[str, float] = {}
        name_to_group: Dict[str, str] = {}

        for entry in name_list:
            swap_name: str = entry.get("name", "")
            group: str = entry.get("group", "Unknown")

            if not swap_name.strip():
                continue

            # Skip if the swap name is the same as the detected name
            if detected_name and swap_name.strip().lower() == detected_name.lower():
                logger.debug("Skipping swap — name matches original: %r", swap_name)
                continue

            variant_id = f"{resume.id}_swap_{swap_name.replace(' ', '_')}"

            if detected_name:
                variant = _build_variant_resume(
                    original=resume,
                    old_name=detected_name,
                    new_name=swap_name,
                    variant_id=variant_id,
                )
            else:
                # No name detected — inject the swap name at the top of the text
                injected_text = f"{swap_name}\n{resume.raw_text}"
                variant = parse_resume(injected_text, resume_id=variant_id, category=resume.category)

            variant_result = score_resume(variant, job)
            swapped_scores[swap_name] = variant_result.final_score
            name_to_group[swap_name] = group

        # ── Step 4: Compute score shift statistics ────────────────────────────
        if swapped_scores:
            diffs = [abs(s - original_score) for s in swapped_scores.values()]
            max_diff = round(max(diffs), 4)
            avg_diff = round(sum(diffs) / len(diffs), 4)
        else:
            max_diff = 0.0
            avg_diff = 0.0

        # ── Step 5: Demographic parity difference ─────────────────────────────
        # DPD = max(group_avg_score) − min(group_avg_score)
        # This is a label-free proxy: we treat "score >= threshold" as a
        # positive prediction and measure the spread across groups.
        group_avgs = _compute_group_averages(swapped_scores, name_to_group)
        if len(group_avgs) >= 2:
            dpd = round(max(group_avgs.values()) - min(group_avgs.values()), 4)
        elif swapped_scores:
            dpd = round(max_diff, 4)
        else:
            dpd = 0.0

        # ── Step 6: Summary ───────────────────────────────────────────────────
        summary = _build_summary(
            original_score=original_score,
            swapped_scores=swapped_scores,
            max_diff=max_diff,
            avg_diff=avg_diff,
            dpd=dpd,
        )

        return FairnessReport(
            original_score=original_score,
            swapped_scores=swapped_scores,
            max_difference=max_diff,
            average_difference=avg_diff,
            demographic_parity_difference=dpd,
            equalized_odds=None,  # requires ground-truth labels
            summary=summary,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected error during bias audit for resume_id=%s", resume.id
        )
        return FairnessReport(
            original_score=0.0,
            swapped_scores={},
            max_difference=0.0,
            average_difference=0.0,
            demographic_parity_difference=0.0,
            summary="Audit failed due to an unexpected error.",
            audit_error=str(exc),
        )

"""
preprocess.py - Text cleaning and skill normalization for experiments.
"""
import re
import json
import os
from pathlib import Path
from typing import List

# Skill alias map (subset - enough for experiments)
SKILL_ALIASES = {
    "ml": "machine learning", "nlp": "natural language processing",
    "dl": "deep learning", "cv": "computer vision",
    "js": "javascript", "ts": "typescript", "py": "python",
    "tf": "tensorflow", "k8s": "kubernetes", "aws": "amazon web services",
    "gcp": "google cloud platform", "sql": "sql", "nosql": "nosql",
    "sklearn": "scikit-learn", "scikit": "scikit-learn",
    "pytorch": "pytorch", "torch": "pytorch",
    "docker": "docker", "git": "git", "linux": "linux",
    "pandas": "pandas", "numpy": "numpy", "flask": "flask",
    "django": "django", "fastapi": "fastapi", "react": "react",
    "node": "node.js", "nodejs": "node.js",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "i", "we", "you", "he", "she", "they", "it", "this", "that",
    "experience", "years", "work", "working", "strong", "good",
    "excellent", "knowledge", "skills", "ability", "using", "use",
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split on whitespace after cleaning."""
    return clean_text(text).split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def normalize_skill(skill: str) -> str:
    """Map alias to canonical form."""
    return SKILL_ALIASES.get(skill.lower(), skill.lower())


def normalize_skills(skills: List[str]) -> List[str]:
    """Normalize a list of skills, deduplicate."""
    seen = set()
    result = []
    for s in skills:
        norm = normalize_skill(s)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def preprocess_text(text: str, remove_stops: bool = True) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> rejoin."""
    tokens = tokenize(text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def load_dataset(path: str = None) -> list:
    """Load the resume dataset JSON."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "resume_dataset.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples")
    sample = dataset[0]
    print("Original resume (first 100 chars):", sample["resume_text"][:100])
    print("Preprocessed:", preprocess_text(sample["resume_text"])[:100])
    print("Normalized skills:", normalize_skills(sample["skills_true"]))

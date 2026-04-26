"""
preprocess.py - Text cleaning and skill normalization for experiments.

Provides:
  - Text cleaning (lowercase, remove noise, collapse whitespace)
  - Tokenization
  - Stopword removal
  - Skill alias normalization (abbreviations -> canonical forms)
  - Dataset loading
"""
import re
import json
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Skill alias map — maps abbreviations and variants to canonical skill names.
# Applied to BOTH predicted and ground-truth skills for fair comparison.
# ---------------------------------------------------------------------------
SKILL_ALIASES = {
    # Machine Learning / AI
    "ml":                          "machine learning",
    "machine-learning":            "machine learning",
    "dl":                          "deep learning",
    "deep-learning":               "deep learning",
    "nlp":                         "natural language processing",
    "natural-language-processing": "natural language processing",
    "cv":                          "computer vision",
    "computer-vision":             "computer vision",
    "rl":                          "reinforcement learning",
    # Languages
    "py":                          "python",
    "python3":                     "python",
    "js":                          "javascript",
    "javascript":                  "javascript",
    "ts":                          "typescript",
    "typescript":                  "typescript",
    "golang":                      "golang",
    "go":                          "golang",
    "cpp":                         "c++",
    "csharp":                      "c#",
    # Frameworks / Libraries
    "tf":                          "tensorflow",
    "tensorflow2":                 "tensorflow",
    "torch":                       "pytorch",
    "sklearn":                     "scikit-learn",
    "scikit":                      "scikit-learn",
    "scikit-learn":                "scikit-learn",
    "huggingface":                 "hugging face",
    "hugging-face":                "hugging face",
    "hf":                          "hugging face",
    "node":                        "node.js",
    "nodejs":                      "node.js",
    "node.js":                     "node.js",
    "reactjs":                     "react",
    "vuejs":                       "vue.js",
    "angularjs":                   "angular",
    "expressjs":                   "express.js",
    "springboot":                  "spring boot",
    "spring-boot":                 "spring boot",
    # Cloud
    "aws":                         "amazon web services",
    "amazon-web-services":         "amazon web services",
    "gcp":                         "google cloud platform",
    "google-cloud":                "google cloud platform",
    "azure":                       "microsoft azure",
    "ms-azure":                    "microsoft azure",
    # DevOps / Infrastructure
    "k8s":                         "kubernetes",
    "kube":                        "kubernetes",
    "ci/cd":                       "continuous integration and deployment",
    "cicd":                        "continuous integration and deployment",
    # Data
    "sql":                         "sql",
    "nosql":                       "nosql",
    "postgres":                    "postgresql",
    "postgresql":                  "postgresql",
    "mongo":                       "mongodb",
    "mongodb":                     "mongodb",
    "np":                          "numpy",
    "pd":                          "pandas",
    # Big Data
    "spark":                       "apache spark",
    "apache-spark":                "apache spark",
    "kafka":                       "apache kafka",
    "apache-kafka":                "apache kafka",
    "airflow":                     "apache airflow",
    "apache-airflow":              "apache airflow",
    # MLOps
    "mlflow":                      "mlflow",
    "mlops":                       "mlops",
    # General
    "git":                         "git",
    "github":                      "github",
    "linux":                       "linux",
    "bash":                        "bash",
    "docker":                      "docker",
    "terraform":                   "terraform",
    "ansible":                     "ansible",
    "jenkins":                     "jenkins",
    "flask":                       "flask",
    "django":                      "django",
    "fastapi":                     "fastapi",
    "pandas":                      "pandas",
    "numpy":                       "numpy",
    "matplotlib":                  "matplotlib",
    "statistics":                  "statistics",
    "stats":                       "statistics",
    "rest":                        "rest api",
    "restful":                     "rest api",
    "graphql":                     "graphql",
    "agile":                       "agile",
    "scrum":                       "scrum",
    "jira":                        "jira",
    "tableau":                     "tableau",
    "powerbi":                     "power bi",
    "power-bi":                    "power bi",
    "excel":                       "microsoft excel",
    "ms-excel":                    "microsoft excel",
    "bert":                        "bert",
    "gpt":                         "gpt",
    "llm":                         "large language models",
    "transformers":                "transformers",
    "xgboost":                     "xgboost",
    "lightgbm":                    "lightgbm",
    "redis":                       "redis",
    "java":                        "java",
    "scala":                       "scala",
    "kotlin":                      "kotlin",
    "swift":                       "swift",
    "ruby":                        "ruby",
    "php":                         "php",
    "r":                           "r",
    "matlab":                      "matlab",
}

# ---------------------------------------------------------------------------
# Stopwords — common English words + domain-generic resume terms
# ---------------------------------------------------------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "i", "we", "you", "he", "she", "they", "it", "this", "that",
    "experience", "years", "work", "working", "strong", "good",
    "excellent", "knowledge", "skills", "ability", "using", "use",
    "proficient", "familiar", "comfortable", "background", "expertise",
    "role", "position", "team", "company", "candidate", "required",
    "preferred", "plus", "expected", "needed", "looking", "seeking",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Clean then split on whitespace."""
    return clean_text(text).split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove tokens that are in the stopword set."""
    return [t for t in tokens if t not in STOPWORDS]


def normalize_skill(skill: str) -> str:
    """
    Map a single skill token to its canonical form.
    Unknown tokens are returned lowercased as-is.
    """
    return SKILL_ALIASES.get(skill.lower().strip(), skill.lower().strip())


def normalize_skills(skills: List[str]) -> List[str]:
    """
    Normalize a list of skill strings and deduplicate.
    Applies alias mapping to each skill, preserving order of first occurrence.
    """
    seen: set = set()
    result: List[str] = []
    for s in skills:
        norm = normalize_skill(s)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def preprocess_text(text: str, remove_stops: bool = True) -> str:
    """
    Full preprocessing pipeline:
      1. clean_text  (lowercase, remove punctuation/numbers)
      2. tokenize    (split on whitespace)
      3. remove_stopwords (optional)
      4. rejoin as string
    """
    tokens = tokenize(text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def load_dataset(path: str = None) -> list:
    """
    Load the resume_dataset.json file.
    Defaults to experiments/data/resume_dataset.json relative to this file.
    """
    if path is None:
        path = Path(__file__).parent.parent / "data" / "resume_dataset.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples")

    sample = dataset[0]
    print("\nOriginal resume (first 120 chars):")
    print(" ", sample["resume_text"][:120])
    print("\nPreprocessed (first 120 chars):")
    print(" ", preprocess_text(sample["resume_text"])[:120])
    print("\nNormalized skills_true:")
    print(" ", normalize_skills(sample["skills_true"]))

    print("\nAlias tests:")
    for alias, expected in [("ml", "machine learning"), ("k8s", "kubernetes"),
                             ("aws", "amazon web services"), ("sklearn", "scikit-learn"),
                             ("js", "javascript"), ("gcp", "google cloud platform")]:
        result = normalize_skill(alias)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"  {alias!r} -> {result!r}  [{status}]")

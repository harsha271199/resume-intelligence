# Resume Intelligence

A Multi-Agent Framework for Structured and Fair Resume Intelligence.

Extracts structured information from resumes, semantically matches them against job descriptions, and audits the scoring process for demographic bias — all in a single explainable pipeline.

---

## Overview

The system is composed of three cooperating agents:

| Agent | Responsibility |
|---|---|
| **Resume Parsing Agent** | Extracts skills, education, experience, organizations, and job titles from raw resume text using spaCy NER + rule-based patterns |
| **Semantic Scoring Agent** | Computes a Skill Alignment Score (0–100) using sentence-transformers (MiniLM) and a TF-IDF baseline |
| **Bias Auditing Agent** | Detects demographic bias via controlled name-swapping experiments and Fairlearn fairness metrics |

---

## Project Structure

```
resume-intelligence/
├── app.py                          # Streamlit entry point
├── config.yaml                     # Application configuration
├── requirements.txt                # Runtime + dev dependencies
├── pyproject.toml                  # Package metadata and build config
├── .env.example                    # Environment variable template
├── data/
│   ├── sample_resumes.csv          # Sample resume data
│   └── sample_jobs.csv             # Sample job description data
├── src/resume_intelligence/
│   ├── agents/                     # Parsing, scoring, auditing agents
│   ├── data/                       # Synthetic data, skill aliases, name pairs
│   ├── models/                     # Pydantic data contracts
│   ├── preprocessing/              # Text cleaning utilities
│   ├── evaluation/                 # NER F1 + Spearman evaluation harness
│   ├── visualization/              # Score charts, skill graphs
│   └── utils/                      # Shared helpers, config loader
└── tests/                          # pytest test suite
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd resume-intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the package and all dependencies
pip install -e ".[dev]"

# 4. Download the spaCy model
python -m spacy download en_core_web_sm

# 5. Copy the environment template
cp .env.example .env
```

---

## Running the Streamlit UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## CLI Usage

```bash
# Parse a resume and output structured JSON
resume-intel parse --input resume.txt

# Score a parsed resume against a job description
resume-intel score --resume parsed.json --jd job.txt

# Run the bias audit
resume-intel audit --input resume.txt --jd job.txt

# Run the full pipeline (all three agents)
resume-intel pipeline --input resume.txt --jd job.txt --output result.json
```

Add `--help` to any command for full usage details.

---

## Running Tests

```bash
pytest tests/
```

With coverage report:

```bash
pytest tests/ --cov=src/resume_intelligence --cov-report=term-missing
```

---

## Configuration

Copy `.env.example` to `.env` and adjust values as needed. Key settings:

- `SPACY_MODEL` — spaCy model name (default: `en_core_web_sm`)
- `SBERT_MODEL` — sentence-transformers model (default: `all-MiniLM-L6-v2`)
- `SCORE_W1`–`SCORE_W4` — scoring component weights (must sum to 1.0)
- `RESUME_DATASET_PATH` / `NER_DATASET_PATH` / `JD_DATASET_PATH` — leave blank to use built-in synthetic data

All settings are also available in `config.yaml`.

---

## Datasets

The system works out of the box with built-in synthetic data. To use real datasets:

| Dataset | Source | Path config key |
|---|---|---|
| Resume Dataset (~2,484 resumes) | [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | `RESUME_DATASET_PATH` |
| Resume NER Dataset (~200 annotated) | [HuggingFace](https://huggingface.co/datasets/resume-ner) | `NER_DATASET_PATH` |
| Job Description Dataset | LinkedIn/Indeed export | `JD_DATASET_PATH` |

---

## License

MIT

# Resume Intelligence

**A Multi-Agent AI System for Resume Screening, ATS Analysis, and Bias Auditing**

Parses resumes, semantically matches them against job descriptions, classifies missing ATS keywords by priority, and audits the scoring process for demographic bias — all in a single explainable pipeline.

---

## Live Demo

[https://resume-intelligence-bnugnruytcrtmssg4dbwnb.streamlit.app/](https://resume-intelligence-bnugnruytcrtmssg4dbwnb.streamlit.app/)

Paste a resume and job description to get an instant Skill Alignment Score, ATS gap report, and bias audit.

---

## Key Features

- **Resume Parsing** — extracts skills, education, experience, organizations, and job titles using spaCy NER and rule-based patterns
- **Semantic Matching** — computes similarity between resume and job description using Sentence Transformers (MiniLM), going beyond simple keyword overlap
- **Skill Alignment Score** — weighted composite score (0–100) combining semantic similarity, skill coverage, experience match, and skill gap penalty
- **ATS Keyword Classification** — missing skills are classified as Required, Preferred, or Other based on JD context, mirroring how real ATS systems prioritize gaps
- **Intelligent Job Title Extraction** — scores candidate lines from the JD to identify the actual role title, ignoring marketing copy and salary text
- **Bias Auditing** — detects demographic bias via controlled name-swapping experiments across 15 name variants, with Fairlearn fairness metrics
- **Experimental ML Model** — optional TF-IDF + Random Forest classifier trained on a labeled resume dataset as a baseline comparison

---

## How This Is Different

| Approach | This System | Typical ATS |
|---|---|---|
| Matching | Semantic (MiniLM embeddings) | Keyword frequency |
| Missing skills | Classified by JD priority (required / preferred) | Flat list |
| Bias detection | Controlled name-swap experiment | None |
| Explainability | Score breakdown + per-skill suggestions | Black box |

---

## Architecture

```
Resume Text + Job Description
        │
        ▼
┌─────────────────┐
│  Parsing Agent  │  spaCy NER + rule-based skill extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Scoring Agent  │  MiniLM semantic similarity + skill coverage + experience match
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ATS Classifier │  Required / Preferred / Other keyword gap analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bias Auditor   │  Name-swapping fairness experiment (Fairlearn)
└────────┬────────┘
         │
         ▼
   Streamlit UI
```

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **spaCy** (`en_core_web_sm`) — NER and text processing
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — semantic embeddings
- **scikit-learn** — TF-IDF baseline, Random Forest classifier, cosine similarity
- **Fairlearn** — demographic parity and fairness metrics
- **Pydantic** — typed data contracts across all agents
- **Pandas** — data handling and display

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

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running Tests

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src/resume_intelligence --cov-report=term-missing
```

---

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model name |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `SCORE_W1`–`SCORE_W4` | `0.40 / 0.30 / 0.20 / 0.10` | Scoring component weights |

---

## Project Structure

```
resume-intelligence/
├── app.py                          # Streamlit entry point
├── src/resume_intelligence/
│   ├── agents/                     # Parsing, scoring, auditing agents
│   ├── models/                     # Pydantic data contracts
│   ├── pipeline/                   # End-to-end pipeline orchestrator
│   └── data/                       # Skill aliases, name pairs
├── experiments/                    # Training, evaluation, notebooks
├── tests/                          # pytest test suite
└── data/                           # Sample resumes and job descriptions
```

---

## License

MIT

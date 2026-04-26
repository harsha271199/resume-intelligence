# Resume Intelligence — Data Mining Experiments

This directory contains the complete data mining experiments pipeline for the Resume Intelligence project. All code here is **isolated from the production application** and is intended for research, model comparison, and evaluation purposes only.

---

## Project Overview

The goal of these experiments is to evaluate different machine learning approaches for the **resume–job matching** task: given a resume and a job description, predict whether the candidate is a good match (label = 1) or not (label = 0).

Three modelling approaches are compared:
1. TF-IDF features + Logistic Regression (baseline)
2. TF-IDF features + Random Forest
3. Sentence Embeddings (MiniLM) + Logistic Regression

---

## Directory Structure

```
experiments/
├── data/
│   ├── resume_dataset.json      # 25-sample labelled dataset
│   └── ground_truth.json        # Ground truth for parser evaluation
├── preprocessing/
│   └── preprocess.py            # Text cleaning and skill normalisation
├── training/
│   ├── train_model.py           # Feature engineering + model training
│   ├── training_results.json    # Generated after running train_model.py
│   └── saved_models/            # Pickled model artefacts
├── evaluation/
│   ├── evaluate_model.py        # Full dataset evaluation + confusion matrices
│   ├── evaluate_parser.py       # Parser skill extraction evaluation
│   ├── evaluation_results.json  # Generated after running evaluate_model.py
│   └── parser_eval_results.json # Generated after running evaluate_parser.py
├── notebooks/
│   └── analysis.ipynb           # End-to-end analysis notebook
└── README.md                    # This file
```

---

## Dataset Description

**File:** `data/resume_dataset.json`

The dataset contains **25 manually crafted samples** covering a range of technical roles and skill sets.

| Property | Value |
|---|---|
| Total samples | 25 |
| Positive samples (label = 1, good match) | 13 |
| Negative samples (label = 0, weak/no match) | 12 |

Each sample has the following schema:

```json
{
  "id": "sample_01",
  "resume_text": "...",
  "job_text": "...",
  "skills_true": ["python", "sql", "docker"],
  "label": 1
}
```

**Positive samples** pair resumes and job descriptions from the same technical domain (data science, ML engineering, DevOps, full-stack development, etc.).

**Negative samples** deliberately mismatch domains — for example, a graphic designer resume paired with a data scientist job description, or a financial analyst resume paired with a DevOps role.

Skill synonyms and abbreviations are used throughout to test normalisation robustness (e.g. `ML → machine learning`, `JS → javascript`, `k8s → kubernetes`, `PyTorch → pytorch`).

---

## Preprocessing

**File:** `preprocessing/preprocess.py`

The preprocessing pipeline applies the following steps in order:

1. **Lowercasing** — all text is converted to lowercase.
2. **Punctuation and number removal** — non-alphabetic characters are replaced with spaces.
3. **Whitespace normalisation** — consecutive spaces are collapsed.
4. **Tokenisation** — text is split on whitespace.
5. **Stopword removal** — a curated set of common English words and domain-generic terms (e.g. "experience", "skills") are removed.
6. **Skill normalisation** — a hand-crafted alias map maps abbreviations and variants to canonical forms (e.g. `"ml" → "machine learning"`, `"sklearn" → "scikit-learn"`).

Key functions:

| Function | Description |
|---|---|
| `clean_text(text)` | Lowercase + remove punctuation/numbers |
| `tokenize(text)` | Clean then split on whitespace |
| `remove_stopwords(tokens)` | Filter stopword list |
| `normalize_skill(skill)` | Map single skill alias to canonical form |
| `normalize_skills(skills)` | Normalise and deduplicate a skill list |
| `preprocess_text(text)` | Full pipeline: clean → tokenize → remove stops → rejoin |
| `load_dataset(path)` | Load `resume_dataset.json` |

---

## Feature Engineering

Two feature representations are compared:

### TF-IDF (Term Frequency–Inverse Document Frequency)
- Combined resume + job description text is preprocessed and vectorised.
- `TfidfVectorizer` with `max_features=500` and `ngram_range=(1, 2)` (unigrams and bigrams).
- Sparse matrix representation; fast to compute, no external model required.

### Sentence Embeddings (MiniLM)
- Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model.
- Produces dense 384-dimensional vectors capturing semantic meaning.
- Better at handling synonyms and paraphrasing than TF-IDF.

---

## Models Trained

| Model | Features | Notes |
|---|---|---|
| Logistic Regression | TF-IDF | Baseline; fast, interpretable |
| Random Forest | TF-IDF | Ensemble; handles non-linear boundaries |
| Logistic Regression | MiniLM Embeddings | Semantic; best generalisation expected |

All models are trained with a **75/25 stratified train/test split** (`random_state=42`).

Trained models are saved as pickle files in `training/saved_models/`.

---

## Evaluation Metrics

Models are evaluated on accuracy, precision, recall, and F1 score. The table below shows placeholder values — run the scripts to populate with actual results.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | — | — | — | — |
| TF-IDF + Random Forest | — | — | — | — |
| Embeddings + Logistic Regression | — | — | — | — |

Confusion matrices are printed to stdout by `evaluate_model.py`.

---

## Parser Evaluation

**File:** `evaluation/evaluate_parser.py`

Evaluates the production `parse_resume()` agent's skill extraction against the ground truth in `data/ground_truth.json`.

Metrics computed per sample and macro-averaged:
- **Precision** — fraction of predicted skills that are correct.
- **Recall** — fraction of true skills that were predicted.
- **F1** — harmonic mean of precision and recall.

Skill normalisation is applied to both predicted and ground-truth skill lists before comparison, so alias variants are handled correctly.

---

## Limitations

- **Small dataset (25 samples):** Results have high variance. Metrics should be interpreted as directional rather than definitive.
- **No cross-validation:** A single train/test split is used. K-fold CV would give more reliable estimates.
- **Synthetic data:** Samples were crafted manually, which may not fully reflect the diversity of real-world resumes and job postings.
- **Skill alias coverage:** The alias map covers common abbreviations but will miss domain-specific or emerging terminology.
- **Embedding model size:** `all-MiniLM-L6-v2` is a lightweight model. Larger models (e.g. `all-mpnet-base-v2`) may yield better semantic representations.

---

## How to Run

All commands should be run from the `resume-intelligence/` directory.

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers scikit-learn
```

### 2. Verify preprocessing

```bash
python experiments/preprocessing/preprocess.py
```

### 3. Train models

```bash
python experiments/training/train_model.py
```

### 4. Evaluate models on full dataset

```bash
python experiments/evaluation/evaluate_model.py
```

### 5. Evaluate parser skill extraction

```bash
python experiments/evaluation/evaluate_parser.py
```

### 6. Open the analysis notebook

```bash
jupyter notebook experiments/notebooks/analysis.ipynb
```

### Quick import check

```bash
python -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
from experiments.preprocessing.preprocess import load_dataset, preprocess_text
d = load_dataset('experiments/data/resume_dataset.json')
print(len(d), 'samples loaded')
"
```

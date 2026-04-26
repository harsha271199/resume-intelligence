"""
train_model.py - Feature engineering and model training for resume matching.

Models trained:
  1. TF-IDF + Logistic Regression  (baseline)
  2. TF-IDF + Random Forest
  3. Sentence Embeddings (MiniLM) + Logistic Regression

Strategy:
  - 75/25 stratified train/test split (random_state=42 for reproducibility)
  - Cross-validation (5-fold) reported alongside held-out test metrics
  - TF-IDF: max_features=300, unigrams+bigrams, sublinear_tf=True
  - Embeddings: all-MiniLM-L6-v2 (384-dim dense vectors)

Outputs:
  - Pickled models in training/saved_models/
  - training_results.json with per-model metrics

Usage:
    python experiments/training/train_model.py
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.preprocessing.preprocess import load_dataset, preprocess_text

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.25
CV_FOLDS     = 5


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(dataset: list) -> tuple:
    """
    Combine resume + job description text, preprocess, extract labels.
    Returns (texts, labels).
    """
    texts, labels = [], []
    for s in dataset:
        combined = preprocess_text(s["resume_text"] + " " + s["job_text"])
        texts.append(combined)
        labels.append(int(s["label"]))
    return texts, labels


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_tfidf_features(texts_train: list, texts_test: list) -> tuple:
    """
    Fit TF-IDF on training texts, transform both splits.
    Uses sublinear_tf=True to dampen high-frequency term dominance.
    """
    vec = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X_train = vec.fit_transform(texts_train)
    X_test  = vec.transform(texts_test)
    return X_train, X_test, vec


def build_embedding_features(texts_train: list, texts_test: list) -> tuple:
    """
    Encode texts using MiniLM sentence-transformers (384-dim).
    Returns numpy arrays.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = model.encode(texts_train, convert_to_numpy=True, show_progress_bar=False)
    X_test  = model.encode(texts_test,  convert_to_numpy=True, show_progress_bar=False)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    name: str,
    X_train, X_test,
    y_train: list, y_test: list,
    clf,
    X_all=None, y_all=None,
) -> tuple:
    """
    Fit classifier, evaluate on held-out test set, optionally run CV.
    Returns (results_dict, fitted_clf).
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    results = {
        "model":     name,
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),                    4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)),  4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),      4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)),          4),
        "cv_f1_mean": None,
        "cv_f1_std":  None,
    }

    # Cross-validation on full dataset (if provided)
    if X_all is not None and y_all is not None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(clf, X_all, y_all, cv=cv, scoring="f1")
        results["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
        results["cv_f1_std"]  = round(float(cv_scores.std()),  4)

    return results, clf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    dataset = load_dataset()
    texts, labels = load_and_prepare(dataset)

    print(f"Dataset: {len(texts)} samples  "
          f"(pos={sum(labels)}, neg={len(labels)-sum(labels)})")

    # Stratified split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train: {len(X_train_txt)}  Test: {len(X_test_txt)}\n")

    all_results = []

    # ------------------------------------------------------------------
    # Model 1: TF-IDF + Logistic Regression
    # ------------------------------------------------------------------
    print("Training: TF-IDF + Logistic Regression...")
    X_tr, X_te, vec1 = build_tfidf_features(X_train_txt, X_test_txt)
    X_all_tfidf = vec1.transform(texts)

    res1, clf1 = train_and_evaluate(
        "TF-IDF + Logistic Regression",
        X_tr, X_te, y_train, y_test,
        LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE,
                           solver="liblinear"),
        X_all=X_all_tfidf, y_all=labels,
    )
    all_results.append(res1)
    pickle.dump(
        {"vectorizer": vec1, "classifier": clf1},
        open(SAVED_MODELS_DIR / "tfidf_logreg.pkl", "wb"),
    )

    # ------------------------------------------------------------------
    # Model 2: TF-IDF + Random Forest
    # ------------------------------------------------------------------
    print("Training: TF-IDF + Random Forest...")
    res2, clf2 = train_and_evaluate(
        "TF-IDF + Random Forest",
        X_tr, X_te, y_train, y_test,
        RandomForestClassifier(n_estimators=200, max_depth=None,
                               random_state=RANDOM_STATE, n_jobs=-1),
        X_all=X_all_tfidf, y_all=labels,
    )
    all_results.append(res2)
    pickle.dump(
        {"vectorizer": vec1, "classifier": clf2},
        open(SAVED_MODELS_DIR / "tfidf_rf.pkl", "wb"),
    )

    # ------------------------------------------------------------------
    # Model 3: Sentence Embeddings + Logistic Regression
    # ------------------------------------------------------------------
    print("Training: Sentence Embeddings + Logistic Regression...")
    X_tr_emb, X_te_emb = build_embedding_features(X_train_txt, X_test_txt)

    # For CV we need all embeddings
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    X_all_emb = _sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    res3, clf3 = train_and_evaluate(
        "Embeddings + Logistic Regression",
        X_tr_emb, X_te_emb, y_train, y_test,
        LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        X_all=X_all_emb, y_all=labels,
    )
    all_results.append(res3)
    pickle.dump(clf3, open(SAVED_MODELS_DIR / "emb_logreg.pkl", "wb"))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("MODEL TRAINING RESULTS")
    print("=" * 65)
    print(f"{'Model':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'CV-F1':>10}")
    print("-" * 65)
    for r in all_results:
        cv_str = (f"{r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f}"
                  if r["cv_f1_mean"] is not None else "  N/A  ")
        print(f"{r['model']:<40} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f}  {cv_str:>10}")

    best = max(all_results, key=lambda x: x["f1"])
    print(f"\nBest model by F1: {best['model']}  (F1={best['f1']:.4f})")

    # Save results
    results_path = Path(__file__).parent / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Models saved to  {SAVED_MODELS_DIR}")


if __name__ == "__main__":
    main()

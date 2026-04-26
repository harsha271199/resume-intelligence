"""
train_model.py - Feature engineering and model training for resume matching.

Models trained:
  1. TF-IDF + Logistic Regression (baseline)
  2. TF-IDF + Random Forest
  3. Sentence Embeddings (MiniLM) + Logistic Regression

Saves trained models to experiments/training/saved_models/
"""
import sys
import os
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path setup
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.preprocessing.preprocess import load_dataset, preprocess_text

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def load_and_prepare(dataset):
    """Combine resume+job text, preprocess, extract labels."""
    texts, labels = [], []
    for s in dataset:
        combined = preprocess_text(s["resume_text"] + " " + s["job_text"])
        texts.append(combined)
        labels.append(s["label"])
    return texts, labels


def build_tfidf_features(texts_train, texts_test):
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    return X_train, X_test, vec


def build_embedding_features(texts_train, texts_test):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = model.encode(texts_train, convert_to_numpy=True, show_progress_bar=False)
    X_test = model.encode(texts_test, convert_to_numpy=True, show_progress_bar=False)
    return X_train, X_test


def train_and_evaluate(name, X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results = {
        "model": name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    return results, clf


def main():
    dataset = load_dataset()
    texts, labels = load_and_prepare(dataset)

    # Stratified split: 75% train, 25% test
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    all_results = []

    # --- Model 1: TF-IDF + Logistic Regression ---
    X_tr, X_te, vec1 = build_tfidf_features(X_train_txt, X_test_txt)
    res1, clf1 = train_and_evaluate(
        "TF-IDF + Logistic Regression",
        X_tr, X_te, y_train, y_test,
        LogisticRegression(max_iter=1000, random_state=42)
    )
    all_results.append(res1)
    pickle.dump({"vectorizer": vec1, "classifier": clf1},
                open(SAVED_MODELS_DIR / "tfidf_logreg.pkl", "wb"))

    # --- Model 2: TF-IDF + Random Forest ---
    res2, clf2 = train_and_evaluate(
        "TF-IDF + Random Forest",
        X_tr, X_te, y_train, y_test,
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    all_results.append(res2)
    pickle.dump({"vectorizer": vec1, "classifier": clf2},
                open(SAVED_MODELS_DIR / "tfidf_rf.pkl", "wb"))

    # --- Model 3: Embeddings + Logistic Regression ---
    X_tr_emb, X_te_emb = build_embedding_features(X_train_txt, X_test_txt)
    res3, clf3 = train_and_evaluate(
        "Embeddings + Logistic Regression",
        X_tr_emb, X_te_emb, y_train, y_test,
        LogisticRegression(max_iter=1000, random_state=42)
    )
    all_results.append(res3)
    pickle.dump(clf3, open(SAVED_MODELS_DIR / "emb_logreg.pkl", "wb"))

    # Print results
    print("\n" + "=" * 60)
    print("MODEL TRAINING RESULTS")
    print("=" * 60)
    for r in all_results:
        print(f"\n{r['model']}")
        print(f"  Accuracy:  {r['accuracy']}")
        print(f"  Precision: {r['precision']}")
        print(f"  Recall:    {r['recall']}")
        print(f"  F1 Score:  {r['f1']}")

    # Save results JSON
    results_path = Path(__file__).parent / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("Models saved to", SAVED_MODELS_DIR)


if __name__ == "__main__":
    main()

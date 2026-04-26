"""
evaluate_model.py - Full model evaluation with confusion matrix and comparison.

Loads saved models and evaluates on the full dataset.
Compares TF-IDF vs Embedding approaches.
"""
import sys
import os
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.preprocessing.preprocess import load_dataset, preprocess_text

SAVED_MODELS_DIR = Path(__file__).parent.parent / "training" / "saved_models"


def print_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix — {model_name}")
    print(f"  {'':12} Pred 0   Pred 1")
    print(f"  {'True 0':12} {cm[0][0]:6}   {cm[0][1]:6}")
    print(f"  {'True 1':12} {cm[1][0]:6}   {cm[1][1]:6}")


def evaluate_saved_model(model_path, texts, labels, use_vectorizer=True):
    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    if use_vectorizer:
        vec = saved["vectorizer"]
        clf = saved["classifier"]
        X = vec.transform(texts)
    else:
        clf = saved
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        X = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    y_pred = clf.predict(X)
    return y_pred


def main():
    dataset = load_dataset()
    texts = [preprocess_text(s["resume_text"] + " " + s["job_text"]) for s in dataset]
    labels = [s["label"] for s in dataset]

    models = [
        ("TF-IDF + Logistic Regression", SAVED_MODELS_DIR / "tfidf_logreg.pkl", True),
        ("TF-IDF + Random Forest",       SAVED_MODELS_DIR / "tfidf_rf.pkl",     True),
        ("Embeddings + Logistic Regression", SAVED_MODELS_DIR / "emb_logreg.pkl", False),
    ]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    comparison = []

    for name, path, use_vec in models:
        if not path.exists():
            print(f"\n[SKIP] {name} — model not found. Run train_model.py first.")
            continue

        y_pred = evaluate_saved_model(path, texts, labels, use_vec)

        acc  = accuracy_score(labels, y_pred)
        prec = precision_score(labels, y_pred, zero_division=0)
        rec  = recall_score(labels, y_pred, zero_division=0)
        f1   = f1_score(labels, y_pred, zero_division=0)

        print(f"\n{name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print_confusion_matrix(labels, y_pred, name)

        comparison.append({
            "model": name, "accuracy": round(acc, 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)
        })

    if comparison:
        best = max(comparison, key=lambda x: x["f1"])
        print(f"\n{'=' * 60}")
        print(f"Best model by F1: {best['model']} (F1={best['f1']})")

        # Save comparison
        out = Path(__file__).parent / "evaluation_results.json"
        with open(out, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

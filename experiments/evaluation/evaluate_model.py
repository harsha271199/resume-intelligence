"""
evaluate_model.py - Full model evaluation with confusion matrix and comparison.

Loads saved models from training/saved_models/ and evaluates on the full dataset.
Compares TF-IDF vs Embedding approaches.

Usage:
    python experiments/evaluation/evaluate_model.py

Prerequisites:
    Run experiments/training/train_model.py first.
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.preprocessing.preprocess import load_dataset, preprocess_text

SAVED_MODELS_DIR = Path(__file__).parent.parent / "training" / "saved_models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true: list, y_pred, model_name: str) -> None:
    """Print a formatted 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix — {model_name}")
    print(f"  {'':16} Predicted 0   Predicted 1")
    print(f"  {'Actual 0':16}     {tn:5d}         {fp:5d}")
    print(f"  {'Actual 1':16}     {fn:5d}         {tp:5d}")
    print(f"  (TN={tn}  FP={fp}  FN={fn}  TP={tp})")


def evaluate_saved_model(model_path: Path, texts: list, use_vectorizer: bool):
    """Load a pickled model and return predictions on texts."""
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

    return clf.predict(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load and preprocess full dataset
    dataset = load_dataset()
    texts  = [preprocess_text(s["resume_text"] + " " + s["job_text"]) for s in dataset]
    labels = [int(s["label"]) for s in dataset]

    print(f"\nEvaluating on full dataset: {len(dataset)} samples "
          f"(pos={sum(labels)}, neg={len(labels)-sum(labels)})")

    models = [
        ("TF-IDF + Logistic Regression",     SAVED_MODELS_DIR / "tfidf_logreg.pkl", True),
        ("TF-IDF + Random Forest",            SAVED_MODELS_DIR / "tfidf_rf.pkl",     True),
        ("Embeddings + Logistic Regression",  SAVED_MODELS_DIR / "emb_logreg.pkl",   False),
    ]

    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS")
    print("=" * 65)

    comparison = []

    for name, path, use_vec in models:
        if not path.exists():
            print(f"\n[SKIP] {name}")
            print(f"       Model file not found: {path}")
            print("       Run experiments/training/train_model.py first.")
            continue

        y_pred = evaluate_saved_model(path, texts, use_vec)

        acc  = float(accuracy_score(labels, y_pred))
        prec = float(precision_score(labels, y_pred, zero_division=0))
        rec  = float(recall_score(labels, y_pred, zero_division=0))
        f1   = float(f1_score(labels, y_pred, zero_division=0))

        print(f"\n{name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print_confusion_matrix(labels, y_pred, name)

        comparison.append({
            "model":     name,
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        })

    if not comparison:
        print("\nNo models evaluated. Run train_model.py first.")
        return

    # Best model
    best = max(comparison, key=lambda x: x["f1"])
    print(f"\n{'=' * 65}")
    print(f"Best model by F1: {best['model']}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1 Score:  {best['f1']:.4f}")

    # Save results
    out = Path(__file__).parent / "evaluation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"models": comparison, "best_model": best}, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

"""
evaluate_parser.py - Evaluate the Resume Parsing Agent's skill extraction.

Compares predicted normalized skills against ground-truth skill lists.
Normalization is applied to BOTH sides before comparison so alias variants
(e.g. "ML" vs "machine learning") are handled correctly.

Metrics computed:
  - Per-sample: Precision, Recall, F1
  - Macro-averaged: Precision, Recall, F1

Usage:
    python experiments/evaluation/evaluate_parser.py
"""
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works from any working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent          # resume-intelligence/
sys.path.insert(0, str(ROOT / "src"))               # src/resume_intelligence/...
sys.path.insert(0, str(ROOT))                       # experiments/...

from resume_intelligence.agents.parser_agent import parse_resume
from experiments.preprocessing.preprocess import normalize_skills


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(path=None) -> dict:
    """Load ground_truth.json. Keys are resume IDs, values have 'text' and 'skills'."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: list, true: list) -> dict:
    """
    Compute Precision, Recall, F1 between predicted and true skill sets.

    Both lists are normalized before comparison so alias variants match.
    Returns a dict with keys: precision, recall, f1, tp, fp, fn.
    """
    pred_set = set(normalize_skills(pred))
    true_set = set(normalize_skills(true))

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_ground_truth()

    print("\n" + "=" * 65)
    print("PARSER EVALUATION — Skill Extraction (Normalized)")
    print("=" * 65)
    print(f"Evaluating {len(data)} resumes from ground_truth.json\n")

    all_p, all_r, all_f1 = [], [], []
    results = []

    for key, item in data.items():
        # Run the production parser
        parsed = parse_resume(item["text"])
        pred_skills = parsed.normalized_skills

        # Ground truth (already canonical in our updated ground_truth.json,
        # but normalize anyway for robustness)
        true_skills = item["skills"]

        m = compute_metrics(pred_skills, true_skills)
        all_p.append(m["precision"])
        all_r.append(m["recall"])
        all_f1.append(m["f1"])

        print(f"{key}")
        print(f"  True  ({len(true_skills):2d}): {sorted(normalize_skills(true_skills))}")
        print(f"  Pred  ({len(pred_skills):2d}): {sorted(normalize_skills(pred_skills))}")
        matched = sorted(set(normalize_skills(pred_skills)) & set(normalize_skills(true_skills)))
        missing = sorted(set(normalize_skills(true_skills)) - set(normalize_skills(pred_skills)))
        extra   = sorted(set(normalize_skills(pred_skills)) - set(normalize_skills(true_skills)))
        print(f"  Match: {matched}")
        if missing:
            print(f"  Miss:  {missing}")
        if extra:
            print(f"  Extra: {extra}")
        print(f"  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}"
              f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        print()

        results.append({
            "id": key,
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "predicted": sorted(normalize_skills(pred_skills)),
            "true":      sorted(normalize_skills(true_skills)),
        })

    # Macro averages
    macro_p  = round(sum(all_p)  / len(all_p),  4)
    macro_r  = round(sum(all_r)  / len(all_r),  4)
    macro_f1 = round(sum(all_f1) / len(all_f1), 4)

    print("=" * 65)
    print("MACRO-AVERAGED RESULTS")
    print("=" * 65)
    print(f"  Precision : {macro_p:.4f}")
    print(f"  Recall    : {macro_r:.4f}")
    print(f"  F1 Score  : {macro_f1:.4f}")

    if macro_f1 >= 0.5:
        print(f"\n  [PASS] Macro F1 = {macro_f1:.4f} >= 0.50 threshold")
    else:
        print(f"\n  [WARN] Macro F1 = {macro_f1:.4f} < 0.50 — check ground truth or parser")

    # Save results
    out = Path(__file__).parent / "parser_eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "per_sample": results,
            "macro": {
                "precision": macro_p,
                "recall":    macro_r,
                "f1":        macro_f1,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

"""
evaluate_parser.py - Evaluate the Resume Parsing Agent's skill extraction.

Compares predicted normalized skills against ground truth labels.
Computes Precision, Recall, F1 per sample and macro-averaged.
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from resume_intelligence.agents.parser_agent import parse_resume
from experiments.preprocessing.preprocess import normalize_skills

# ── Load ground truth ─────────────────────────────────────────────────────────


def load_ground_truth(path=None):
    if path is None:
        path = Path(__file__).parent.parent / "data" / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(pred: list, true: list) -> dict:
    pred_set = set(normalize_skills(pred))
    true_set = set(normalize_skills(true))
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    data = load_ground_truth()

    print("\n" + "=" * 60)
    print("PARSER EVALUATION — Skill Extraction")
    print("=" * 60)

    all_p, all_r, all_f1 = [], [], []
    results = []

    for key, item in data.items():
        parsed = parse_resume(item["text"])
        pred_skills = parsed.normalized_skills
        true_skills = item["skills"]

        m = compute_metrics(pred_skills, true_skills)
        all_p.append(m["precision"])
        all_r.append(m["recall"])
        all_f1.append(m["f1"])

        print(f"\n{key}")
        print(f"  True skills:      {true_skills}")
        print(f"  Predicted skills: {pred_skills}")
        print(f"  Precision={m['precision']:.2f}  Recall={m['recall']:.2f}  F1={m['f1']:.2f}")

        results.append({"id": key, **m, "predicted": pred_skills, "true": true_skills})

    macro_p  = sum(all_p)  / len(all_p)
    macro_r  = sum(all_r)  / len(all_r)
    macro_f1 = sum(all_f1) / len(all_f1)

    print(f"\n{'=' * 60}")
    print("MACRO-AVERAGED RESULTS")
    print(f"  Precision: {macro_p:.4f}")
    print(f"  Recall:    {macro_r:.4f}")
    print(f"  F1 Score:  {macro_f1:.4f}")

    # Save results
    out = Path(__file__).parent / "parser_eval_results.json"
    with open(out, "w") as f:
        json.dump({
            "per_sample": results,
            "macro": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4)
            }
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

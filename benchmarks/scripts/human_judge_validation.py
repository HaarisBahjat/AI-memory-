"""
human_judge_validation.py — Phase 9.4

Validates the LLM-as-judge reliability by comparing LLM scores against
human-authored ground-truth ratings on a subset of generated answers.

Usage:
  1. Run answer_benchmark.py first to generate system answers
  2. Run this script to load 30 answers and compute LLM judge scores
  3. The script prints a human annotation template (JSON) you can fill in
  4. After filling in human scores, re-run with --human-scores path to compute Kappa

Cohen's Kappa scale:
  < 0.00  Poor
  0.01-0.20  Slight
  0.21-0.40  Fair
  0.41-0.60  Moderate
  0.61-0.80  Substantial
  0.81-1.00  Almost perfect

Rubric (same for human and LLM judge):
  0 = Completely incorrect
  1 = Mostly incorrect
  2 = Partially correct
  3 = Mostly correct
  4 = Fully correct
  "CORRECT_ABSTENTION" = Correct abstention on NO_ANSWER query
  "INCORRECT_ABSTENTION" = Wrong abstention when answer should exist
  "HALLUCINATED_ANSWER" = Confident but wrong answer
"""
import os
import sys
import json
import argparse
import math
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(BASE_DIR)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def cohen_kappa(ratings_a: list, ratings_b: list) -> float:
    """
    Compute Cohen's Kappa for two raters.
    Expects parallel lists of ratings (same length).
    """
    assert len(ratings_a) == len(ratings_b), "Rater lists must be same length"
    categories = sorted(set(ratings_a) | set(ratings_b))
    n = len(ratings_a)

    # Observed agreement
    p_o = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    # Expected agreement
    p_e = sum(
        (ratings_a.count(c) / n) * (ratings_b.count(c) / n)
        for c in categories
    )

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(ratings_a: list[int], ratings_b: list[int],
                   max_rating: int = 4) -> float:
    """
    Linear weighted Cohen's Kappa for ordinal scales.
    Weight = 1 - |i - j| / max_rating
    """
    n = len(ratings_a)
    categories = list(range(max_rating + 1))
    k = len(categories)

    # Weight matrix (linear weights)
    W = [[1 - abs(i - j) / max_rating for j in categories] for i in categories]

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        if isinstance(a, int) and isinstance(b, int):
            if 0 <= a <= max_rating and 0 <= b <= max_rating:
                conf[a][b] += 1

    # Marginals
    row_sum = [sum(conf[i]) for i in range(k)]
    col_sum = [sum(conf[i][j] for i in range(k)) for j in range(k)]

    # Observed and expected weighted agreement
    p_o = sum(W[i][j] * conf[i][j] for i in range(k) for j in range(k)) / n
    p_e = sum(
        W[i][j] * (row_sum[i] / n) * (col_sum[j] / n)
        for i in range(k) for j in range(k)
    )

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def load_answer_samples(n: int = 30) -> list[dict]:
    """Load the most recent answer benchmark results and sample n items."""
    results_dir = os.path.join(BASE_DIR, "results")
    if not os.path.exists(results_dir):
        return []
    runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")], reverse=True)
    for run in runs:
        candidate = os.path.join(results_dir, run, "answer_benchmark_per_query.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                data = json.load(f)
            import random
            random.seed(42)
            sample = random.sample(data, min(n, len(data)))
            print(f"Loaded {len(sample)} answers from {run}/{os.path.basename(candidate)}")
            return sample
    print("No answer_benchmark_per_query.json found. Run answer_benchmark.py first.")
    return []


def generate_annotation_template(samples: list[dict]) -> dict:
    """Generate a JSON template for human annotators to fill in."""
    template = {
        "_instructions": (
            "Rate each answer on a 0-4 scale (0=completely incorrect, 4=fully correct). "
            "For NO_ANSWER queries, use: 'CORRECT_ABSTENTION', 'INCORRECT_ABSTENTION', or 'HALLUCINATED_ANSWER'. "
            "Fill in 'human_score' for each item."
        ),
        "_rubric": {
            "0": "Completely incorrect",
            "1": "Mostly incorrect",
            "2": "Partially correct",
            "3": "Mostly correct",
            "4": "Fully correct",
            "CORRECT_ABSTENTION": "Correctly said it doesn't know",
            "INCORRECT_ABSTENTION": "Said it doesn't know but should have answered",
            "HALLUCINATED_ANSWER": "Gave a confident wrong answer"
        },
        "annotations": []
    }
    for s in samples:
        template["annotations"].append({
            "query_id": s.get("query_id", ""),
            "category": s.get("category", ""),
            "query": s.get("query", ""),
            "system_answer": s.get("system_answer", ""),
            "ground_truth_answer": s.get("ground_truth_answer", ""),
            "llm_judge_score": s.get("llm_score", None),
            "human_score": None,   # <- fill this in
        })
    return template


def compute_kappa_from_human_scores(annotation_path: str) -> dict:
    """Load filled-in annotation JSON and compute Cohen's Kappa."""
    with open(annotation_path) as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    llm_scores, human_scores = [], []
    skipped = 0

    for a in annotations:
        llm  = a.get("llm_judge_score")
        human = a.get("human_score")
        if llm is None or human is None:
            skipped += 1
            continue
        # Normalize categorical labels to integers for kappa calc
        label_map = {
            "CORRECT_ABSTENTION": 4, "INCORRECT_ABSTENTION": 1, "HALLUCINATED_ANSWER": 0,
        }
        if isinstance(llm, str):
            llm = label_map.get(llm, None)
        if isinstance(human, str):
            human = label_map.get(human, None)
        if llm is None or human is None:
            skipped += 1
            continue
        llm_scores.append(int(llm))
        human_scores.append(int(human))

    n = len(llm_scores)
    if n < 5:
        return {"error": f"Too few valid pairs ({n}) for Kappa computation", "skipped": skipped}

    kappa = cohen_kappa(llm_scores, human_scores)
    wkappa = weighted_kappa(llm_scores, human_scores)

    exact_agreement = sum(1 for a, b in zip(llm_scores, human_scores) if a == b) / n

    interpretation = (
        "almost perfect" if kappa > 0.80 else
        "substantial"   if kappa > 0.60 else
        "moderate"      if kappa > 0.40 else
        "fair"          if kappa > 0.20 else
        "slight"        if kappa > 0.00 else
        "poor"
    )

    result = {
        "n_pairs": n,
        "skipped": skipped,
        "cohen_kappa": round(kappa, 4),
        "weighted_kappa_linear": round(wkappa, 4),
        "exact_agreement_rate": round(exact_agreement, 4),
        "interpretation": interpretation,
        "mean_llm_score": round(sum(llm_scores) / n, 4),
        "mean_human_score": round(sum(human_scores) / n, 4),
    }

    print("\n=== LLM Judge Validation Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


def run(args):
    if args.generate_template:
        samples = load_answer_samples(args.n)
        if not samples:
            return
        template = generate_annotation_template(samples)
        out_path = os.path.join(BASE_DIR, "results", "human_annotation_template.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(template, f, indent=2)
        print(f"\nTemplate saved: {out_path}")
        print("Fill in 'human_score' for each item, then run:")
        print(f"  python human_judge_validation.py --human-scores {out_path}")

    elif args.human_scores:
        result = compute_kappa_from_human_scores(args.human_scores)
        out_path = os.path.join(BASE_DIR, "results", "llm_judge_validation.json")
        with open(out_path, "w") as f:
            json.dump({
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "annotation_file": args.human_scores,
                **result
            }, f, indent=2)
        print(f"\nSaved: {out_path}")
    else:
        print("Usage:")
        print("  Generate template: python human_judge_validation.py --generate-template")
        print("  Compute kappa:     python human_judge_validation.py --human-scores path/to/annotations.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-template", action="store_true")
    parser.add_argument("--human-scores", type=str, default="")
    parser.add_argument("--n", type=int, default=30, help="Number of answers to sample")
    args = parser.parse_args()
    run(args)

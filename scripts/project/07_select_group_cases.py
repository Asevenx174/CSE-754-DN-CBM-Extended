"""EXP-07.1: deterministically select improved, tied, and worsened groups."""

import csv
import json
from collections import defaultdict
from pathlib import Path


EXP06 = Path("experiment-results/exp06")
OUTPUT = Path("experiment-results/exp07")
CASES_PER_CELL = 3


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def classify(delta, lower, upper):
    if delta >= 0.05 and lower > 0:
        return "improved"
    if abs(delta) <= 0.005 and lower <= 0 <= upper:
        return "tied"
    if delta <= -0.01:
        return "worsened_point_estimate"
    return None


def rank(row):
    delta = float(row["delta_auroc_sparse_minus_raw_max"])
    if row["case_category"] == "improved":
        return -delta
    if row["case_category"] == "tied":
        return abs(delta)
    return delta


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores = {
        (row["protocol"], row["concept"]): row
        for row in read(EXP06 / "sparse_test_scores.csv")
    }
    bootstrap = {
        (row["protocol"], row["concept"]): row
        for row in read(EXP06 / "sparse_paired_bootstrap.csv")
    }

    candidates = defaultdict(list)
    for key, score in scores.items():
        uncertainty = bootstrap[key]
        delta = float(score["delta_auroc_sparse_minus_raw_max"])
        lower = float(uncertainty["auroc_sparse_minus_raw_max_ci_lower"])
        upper = float(uncertainty["auroc_sparse_minus_raw_max_ci_upper"])
        category = classify(delta, lower, upper)
        if category:
            candidates[(score["protocol"], category)].append(
                {
                    "protocol": score["protocol"],
                    "case_category": category,
                    "concept": score["concept"],
                    "candidate_count": score["candidate_count"],
                    "nonzero_coefficients": score["nonzero_coefficients"],
                    "positive_coefficients": score["positive_coefficients"],
                    "negative_coefficients": score["negative_coefficients"],
                    "sparse_test_auroc": score["sparse_test_auroc"],
                    "delta_auroc_sparse_minus_raw_max": delta,
                    "delta_ci_lower": lower,
                    "delta_ci_upper": upper,
                }
            )

    selected, counts = [], {}
    for protocol in ("A", "B"):
        counts[protocol] = {}
        for category in ("improved", "tied", "worsened_point_estimate"):
            rows = sorted(candidates[(protocol, category)], key=rank)
            chosen = rows[:CASES_PER_CELL]
            selected.extend(chosen)
            counts[protocol][category] = {
                "eligible": len(rows),
                "selected": len(chosen),
            }

    path = OUTPUT / "selected_group_cases.csv"
    if selected:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=selected[0].keys())
            writer.writeheader()
            writer.writerows(selected)

    summary = {
        "primary_comparison": "sparse aggregation minus raw maximum AUROC",
        "rules": {
            "improved": "delta >= 0.05 and 95% CI lower > 0",
            "tied": "absolute delta <= 0.005 and 95% CI includes 0",
            "worsened_point_estimate": "delta <= -0.01; CI need not exclude 0",
        },
        "cases_per_protocol_category": CASES_PER_CELL,
        "counts": counts,
        "selected_cases": len(selected),
    }
    (OUTPUT / "group_case_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

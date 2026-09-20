"""EXP-09.1: descriptive comparison of frozen Protocol A and B populations."""

import csv
import json
from pathlib import Path

import numpy as np


EXP01 = Path("experiment-results/exp01")
EXP02 = Path("experiment-results/exp02")
EXP03 = Path("experiment-results/exp03")
OUT = Path("experiment-results/exp09")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(protocol):
    rows = read_csv(EXP02 / f"protocol_{protocol.lower()}_atom_scores.csv")
    values = lambda field: np.array([float(row[field]) for row in rows])
    auroc = values("auroc")
    geometry = values("geometry_cosine")
    return {
        "evaluated_atoms": len(rows),
        "evaluated_fraction_of_8192": len(rows) / 8192,
        "evaluated_attributes": len({row["mapped_sun_attribute"] for row in rows}),
        "unique_assigned_names_in_evaluation": len(
            {row["assigned_name"] for row in rows}
        ),
        "mean_geometry_cosine": float(geometry.mean()),
        "median_geometry_cosine": float(np.median(geometry)),
        "negative_geometry_cosines": int((geometry < 0).sum()),
        "mean_test_auroc": float(auroc.mean()),
        "median_test_auroc": float(np.median(auroc)),
        "atoms_auroc_at_least_0.70": int((auroc >= 0.70).sum()),
        "fraction_auroc_at_least_0.70": float((auroc >= 0.70).mean()),
        "inverse_atoms_auroc_below_0.50": int((auroc < 0.50).sum()),
        "mean_test_ap": float(values("average_precision").mean()),
        "mean_activation_frequency": float(values("activation_frequency").mean()),
        "constant_atoms": sum(row["constant_activation"] == "True" for row in rows),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = read_json(EXP03 / "geometry_semantics_summary.json")
    a_mapping = read_json(EXP01 / "protocol_a_mapping_summary.json")
    b_naming = read_json(EXP01 / "sun_naming_summary.json")
    summary = {
        "comparison_type": "descriptive; frozen populations remain separate",
        "A": describe("A"),
        "B": describe("B"),
        "geometry_semantics": geometry,
        "naming_coverage": {
            "A_original_atoms_named": a_mapping["number_of_atoms"],
            "A_unique_original_names": a_mapping["unique_original_names"],
            "A_evaluation_eligible_atoms": a_mapping["evaluation_eligible_atoms"],
            "B_sun_atoms_named": b_naming["number_of_atoms"],
            "B_unique_sun_names": b_naming["concepts_receiving_atoms"],
        },
        "warnings": [
            "Protocol A searches 20k words; Protocol B searches 102 SUN attributes.",
            "Do not interpret different cosine distributions as a fair naming-quality comparison.",
            "Protocol A exclusions mean unevaluable on SUN, not semantically incorrect.",
            "No assignments or populations were redefined after observing test outcomes.",
        ],
    }
    rows = []
    for metric in summary["A"]:
        rows.append(
            {
                "metric": metric,
                "protocol_A": summary["A"][metric],
                "protocol_B": summary["B"][metric],
            }
        )
    with (OUT / "protocol_descriptive_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "protocol_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

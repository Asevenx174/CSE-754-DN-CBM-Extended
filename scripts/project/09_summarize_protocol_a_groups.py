"""EXP-09.2: audit and summarize Protocol-A same-original-name groups."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("experiment-results")
OUT = ROOT / "exp09"


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    pairs = read(ROOT / "exp01/protocol_a_atom_concept_pairs.csv")
    frozen = defaultdict(list)
    for row in pairs:
        if row["evaluation_eligible"] == "True":
            frozen[row["mapped_sun_attribute"]].append(row)

    selections = {
        row["concept"]: row
        for row in read(ROOT / "exp05/validation_selected_atoms.csv")
        if row["protocol"] == "A"
    }
    baselines = {
        row["concept"]: row
        for row in read(ROOT / "exp05/fixed_baseline_test_scores.csv")
        if row["protocol"] == "A"
    }
    sparse = {
        row["concept"]: row
        for row in read(ROOT / "exp06/sparse_test_scores.csv")
        if row["protocol"] == "A"
    }

    rows = []
    for concept, selection in sorted(selections.items()):
        group = frozen[concept]
        names = {row["original_name"] for row in group}
        atoms = sorted(int(row["atom_id"]) for row in group)
        selected_atoms = sorted(map(int, selection["candidate_atom_ids"].split(";")))
        if names != {concept} or atoms != selected_atoms:
            raise RuntimeError(f"Protocol-A group audit failed for {concept!r}.")
        base, model = baselines[concept], sparse[concept]
        rows.append(
            {
                "concept": concept,
                "same_original_name_verified": True,
                "candidate_count": len(atoms),
                "candidate_atom_ids": ";".join(map(str, atoms)),
                "validation_selected_atom": int(selection["selected_atom_id"]),
                "single_test_auroc": float(base["single_test_auroc"]),
                "raw_max_test_auroc": float(base["raw_max_test_auroc"]),
                "sparse_test_auroc": float(model["sparse_test_auroc"]),
                "sparse_minus_single_auroc": float(
                    model["delta_auroc_sparse_minus_single"]
                ),
                "sparse_minus_raw_max_auroc": float(
                    model["delta_auroc_sparse_minus_raw_max"]
                ),
                "single_test_ap": float(base["single_test_ap"]),
                "raw_max_test_ap": float(base["raw_max_test_ap"]),
                "sparse_test_ap": float(model["sparse_test_ap"]),
                "nonzero_sparse_coefficients": int(model["nonzero_coefficients"]),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "protocol_a_group_recovery.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    values = lambda field: np.array([row[field] for row in rows], dtype=float)
    delta_single = values("sparse_minus_single_auroc")
    delta_max = values("sparse_minus_raw_max_auroc")
    summary = {
        "population": "Protocol A eligible same-original-name groups only",
        "groups": len(rows),
        "same_original_name_groups_verified": sum(
            row["same_original_name_verified"] for row in rows
        ),
        "mean_single_test_auroc": float(values("single_test_auroc").mean()),
        "mean_raw_max_test_auroc": float(values("raw_max_test_auroc").mean()),
        "mean_sparse_test_auroc": float(values("sparse_test_auroc").mean()),
        "mean_sparse_minus_single_auroc": float(delta_single.mean()),
        "groups_sparse_above_single": int((delta_single > 0).sum()),
        "mean_sparse_minus_raw_max_auroc": float(delta_max.mean()),
        "groups_sparse_above_raw_max": int((delta_max > 0).sum()),
        "fixed_baseline_bootstrap": read_json(
            ROOT / "exp05/fixed_baseline_bootstrap_summary.json"
        )["A"],
        "sparse_bootstrap": read_json(ROOT / "exp06/sparse_bootstrap_summary.json")["A"],
        "interpretation": (
            "Predictive group recovery for a small exact-match subset; not evidence "
            "that all original names are semantically correct or atoms complementary."
        ),
    }
    (OUT / "protocol_a_group_recovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""EXP-07.2: audit coefficients, atom quality, and activation correlations."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


EXP01 = Path("experiment-results/exp01")
EXP02 = Path("experiment-results/exp02")
EXP06 = Path("experiment-results/exp06")
EXP07 = Path("experiment-results/exp07")
CACHE = Path("data/exp02/sun_frozen_cache.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def correlation_summary(matrix):
    active = matrix.std(axis=0) > 0
    matrix = matrix[:, active]
    if matrix.shape[1] < 2:
        return 0.0, 0.0, int(matrix.shape[1])
    correlations = np.corrcoef(matrix, rowvar=False)
    values = np.abs(correlations[np.triu_indices_from(correlations, k=1)])
    return float(values.mean()), float(values.max()), int(matrix.shape[1])


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    test_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "test"],
        dtype=torch.long,
    )
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    activations = cache["sae_activations"][test_ids].numpy()
    labels = cache["binary_labels"][test_ids].numpy()

    models = {
        (row["protocol"], row["concept"]): row
        for row in read(EXP06 / "selected_sparse_models.csv")
    }
    atom_scores = {}
    for protocol in ("A", "B"):
        for row in read(EXP02 / f"protocol_{protocol.lower()}_atom_scores.csv"):
            atom_scores[(protocol, row["mapped_sun_attribute"], int(row["atom_id"]))] = row

    group_rows, atom_rows = [], []
    for case in read(EXP07 / "selected_group_cases.csv"):
        protocol, concept = case["protocol"], case["concept"]
        model = models[(protocol, concept)]
        atoms = np.asarray(json.loads(model["candidate_atom_ids"]), dtype=int)
        mean = np.asarray(json.loads(model["training_mean"]), dtype=float)
        scale = np.asarray(json.loads(model["training_scale"]), dtype=float)
        coefficients = np.asarray(json.loads(model["coefficients"]), dtype=float)
        nonzero = coefficients != 0
        standardized = (activations[:, atoms] - mean) / scale
        selected = standardized[:, nonzero]
        selected_coefficients = coefficients[nonzero]
        y = labels[:, concept_index[concept]].astype(bool)
        mean_corr, max_corr, variable_atoms = correlation_summary(selected)

        group_rows.append(
            {
                "protocol": protocol,
                "case_category": case["case_category"],
                "concept": concept,
                "nonzero_coefficients": int(nonzero.sum()),
                "positive_coefficients": int((selected_coefficients > 0).sum()),
                "negative_coefficients": int((selected_coefficients < 0).sum()),
                "variable_selected_atoms": variable_atoms,
                "mean_absolute_pairwise_correlation": mean_corr,
                "maximum_absolute_pairwise_correlation": max_corr,
            }
        )

        for atom, coefficient, values in zip(
            atoms[nonzero], selected_coefficients, selected.T
        ):
            score = atom_scores[(protocol, concept, int(atom))]
            contribution = coefficient * values
            atom_rows.append(
                {
                    "protocol": protocol,
                    "case_category": case["case_category"],
                    "concept": concept,
                    "atom_id": int(atom),
                    "coefficient": coefficient,
                    "coefficient_sign": "positive" if coefficient > 0 else "negative",
                    "absolute_coefficient": abs(coefficient),
                    "geometry_cosine": score["geometry_cosine"],
                    "individual_test_auroc": score["auroc"],
                    "individual_test_ap": score["average_precision"],
                    "positive_minus_negative_mean_contribution": float(
                        contribution[y].mean() - contribution[~y].mean()
                    ),
                }
            )

    for filename, rows in (
        ("group_structure_summary.csv", group_rows),
        ("selected_group_atom_details.csv", atom_rows),
    ):
        with (EXP07 / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Audited {len(group_rows)} groups and {len(atom_rows)} nonzero coefficients.")


if __name__ == "__main__":
    main()

"""EXP-04.4: audit worker disagreement and correlated attributes."""

import csv
from pathlib import Path

import numpy as np
import torch


EXP01 = Path("experiment-results/exp01")
EXP04 = Path("experiment-results/exp04")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
TOP_K = 5


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compact(names, scores):
    order = np.argsort(scores)[::-1][:TOP_K]
    return "; ".join(f"{names[i]}:{scores[i]:.3f}" for i in order)


def main():
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    splits = read(EXP01 / "sample_splits.csv")
    train_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "train"]
    )
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    labels = cache["binary_labels"].float()
    agreements = cache["agreement"].float()

    examples = read(EXP04 / "inspection_manifest.csv")
    by_atom = {}
    for row in examples:
        by_atom.setdefault((row["protocol"], row["atom_id"]), []).append(row)

    output = []
    for atom in read(EXP04 / "selected_case_atoms.csv"):
        key = atom["protocol"], atom["atom_id"]
        rows = by_atom[key]
        target = concept_index[atom["mapped_sun_attribute"]]
        high_negatives = [r for r in rows if r["example_type"] == "high_scoring_negative"]
        low_positives = [r for r in rows if r["example_type"] == "low_scoring_positive"]
        negative_ids = torch.tensor([int(r["sample_id"]) for r in high_negatives])

        target_train = labels[train_ids, target].numpy()
        train_matrix = labels[train_ids].numpy()
        correlations = np.array(
            [np.corrcoef(target_train, train_matrix[:, i])[0, 1] for i in range(len(concepts))]
        )
        correlations[target] = -np.inf
        correlations = np.nan_to_num(correlations, nan=-np.inf)

        negative_agreement = agreements[negative_ids].mean(dim=0).numpy()
        negative_agreement[target] = -np.inf
        high_negative_target_agreement = np.array(
            [float(r["worker_agreement"]) for r in high_negatives]
        )
        low_positive_target_agreement = np.array(
            [float(r["worker_agreement"]) for r in low_positives]
        )

        output.append(
            {
                "protocol": atom["protocol"],
                "category": atom["category"],
                "atom_id": atom["atom_id"],
                "assigned_name": atom["assigned_name"],
                "mapped_sun_attribute": atom["mapped_sun_attribute"],
                "high_negative_partial_agreement_fraction": float(
                    np.mean(high_negative_target_agreement > 0)
                ),
                "high_negative_target_mean_agreement": float(
                    high_negative_target_agreement.mean()
                ),
                "low_positive_target_mean_agreement": float(
                    low_positive_target_agreement.mean()
                ),
                "top_training_correlates": compact(concepts, correlations),
                "top_attributes_in_high_negatives": compact(concepts, negative_agreement),
            }
        )

    path = EXP04 / "case_context_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"Audited {len(output)} selected atoms; saved {path}.")


if __name__ == "__main__":
    main()

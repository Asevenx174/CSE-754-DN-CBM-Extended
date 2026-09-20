"""EXP-02.2: score eligible atoms on the frozen SUN test split."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("experiment-results/exp01")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
OUT = Path("experiment-results/exp02")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_lines(path):
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def test_ids():
    rows = read_csv(ROOT / "sample_splits.csv")
    ids = [int(row["sample_id"]) for row in rows if row["split"] == "test"]
    if len(ids) != 2868 or len(ids) != len(set(ids)):
        raise ValueError("Expected 2,868 unique test sample IDs.")
    return torch.tensor(ids, dtype=torch.long)


def protocol_rows(protocol, valid):
    if protocol == "A":
        rows = read_csv(ROOT / "protocol_a_atom_concept_pairs.csv")
        return [
            {
                "atom_id": row["atom_id"],
                "assigned_name": row["original_name"],
                "concept": row["mapped_sun_attribute"],
                "cosine": row["original_cosine"],
            }
            for row in rows
            if row["evaluation_eligible"] == "True"
        ]

    rows = read_csv(ROOT / "sun_atom_assignments.csv")
    return [
        {
            "atom_id": row["atom_id"],
            "assigned_name": row["sun_concept"],
            "concept": row["sun_concept"],
            "cosine": row["cosine"],
        }
        for row in rows
        if row["sun_concept"] in valid
    ]


def score(protocol, candidates, activations, labels, concept_index):
    results = []
    for row in candidates:
        atom = int(row["atom_id"])
        concept = row["concept"]
        y = labels[:, concept_index[concept]]
        z = activations[:, atom]
        positive = y == 1
        negative = ~positive
        constant = bool(np.ptp(z) == 0)

        results.append(
            {
                "atom_id": atom,
                "protocol": protocol,
                "assigned_name": row["assigned_name"],
                "mapped_sun_attribute": concept,
                "geometry_cosine": float(row["cosine"]),
                "test_positives": int(positive.sum()),
                "test_negatives": int(negative.sum()),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, z)),
                "average_precision": float(average_precision_score(y, z)),
                "activation_frequency": float((z > 0).mean()),
                "positive_mean_activation": float(z[positive].mean()),
                "negative_mean_activation": float(z[negative].mean()),
                "positive_minus_negative_mean": float(
                    z[positive].mean() - z[negative].mean()
                ),
                "constant_activation": constant,
            }
        )
    return results


def save(protocol, rows):
    path = OUT / f"protocol_{protocol.lower()}_atom_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    ids = test_ids()
    if not torch.equal(cache["sample_ids"], torch.arange(14340)):
        raise ValueError("Cache sample order is invalid.")

    activations = cache["sae_activations"][ids].numpy()
    labels = cache["binary_labels"][ids].numpy()
    concepts = read_lines(ROOT / "sun102_vocabulary.txt")
    valid = set(read_lines(ROOT / "valid_concepts.txt"))
    concept_index = {name: index for index, name in enumerate(concepts)}

    summary = {}
    for protocol in ("A", "B"):
        rows = score(
            protocol,
            protocol_rows(protocol, valid),
            activations,
            labels,
            concept_index,
        )
        path = save(protocol, rows)
        auroc = np.array([row["auroc"] for row in rows])
        summary[protocol] = {
            "atoms": len(rows),
            "concepts": len({row["mapped_sun_attribute"] for row in rows}),
            "mean_auroc": float(auroc.mean()),
            "median_auroc": float(np.median(auroc)),
            "auroc_at_least_0.70": int((auroc >= 0.70).sum()),
            "inverse_auroc": int((auroc < 0.50).sum()),
            "constant_atoms": sum(row["constant_activation"] for row in rows),
            "output": str(path),
        }

    (OUT / "atom_scoring_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

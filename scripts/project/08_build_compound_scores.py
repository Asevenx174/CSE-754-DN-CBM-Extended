"""EXP-08.2: build raw-maximum SUN compound-node scores."""

import csv
import json
from pathlib import Path

import torch


EXP08 = Path("experiment-results/exp08")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
OUTPUT = Path("data/exp08/compound_scores.pth")


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    activations = cache["sae_activations"]
    with (EXP08 / "compound_groups.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        groups = list(csv.DictReader(file))

    scores = torch.full(
        (activations.shape[0], len(groups)), float("nan"), dtype=torch.float32
    )
    available = torch.zeros(len(groups), dtype=torch.bool)
    counts = []
    for index, row in enumerate(groups):
        atoms = torch.tensor(json.loads(row["retained_atom_ids"]), dtype=torch.long)
        counts.append(len(atoms))
        if len(atoms):
            scores[:, index] = activations[:, atoms].max(dim=1).values.float()
            available[index] = True

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "compound_scores": scores,
            "available_attributes": available,
            "attribute_names": [row["sun_concept"] for row in groups],
            "aggregation": "raw maximum over retained same-name atoms",
        },
        OUTPUT,
    )
    valid_counts = [count for count in counts if count]
    summary = {
        "status": "documented implementation, not exact reproduction",
        "samples": scores.shape[0],
        "sun_attributes": scores.shape[1],
        "available_attributes": int(available.sum()),
        "unavailable_attributes": [
            groups[i]["sun_concept"] for i in range(len(groups)) if not available[i]
        ],
        "retained_atoms": sum(counts),
        "minimum_atoms_per_available_group": min(valid_counts),
        "mean_atoms_per_available_group": sum(valid_counts) / len(valid_counts),
        "maximum_atoms_per_available_group": max(valid_counts),
        "aggregation": "raw maximum",
        "score_cache": str(OUTPUT),
    }
    (EXP08 / "compound_score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

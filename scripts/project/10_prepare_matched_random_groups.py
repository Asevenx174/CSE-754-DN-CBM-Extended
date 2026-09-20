"""EXP-10.3a: freeze same-size, activity-matched random atom groups."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


EXP01 = Path("experiment-results/exp01")
EXP05 = Path("experiment-results/exp05")
OUT = Path("experiment-results/exp10")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
REPLICATES = 20
NEAREST_POOL = 50
SEED = 49


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def matched_group(real_atoms, frequency, rng):
    excluded = set(real_atoms)
    chosen = []
    for atom in real_atoms:
        distance = np.abs(frequency - frequency[atom])
        order = np.argsort(distance, kind="stable")
        candidates = [
            int(candidate)
            for candidate in order
            if candidate not in excluded and candidate not in chosen
        ][:NEAREST_POOL]
        chosen.append(int(rng.choice(candidates)))
    return sorted(chosen)


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    train_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "train"],
        dtype=torch.long,
    )
    frequency = (
        (cache["sae_activations"][train_ids] > 0).float().mean(dim=0).numpy()
    )
    groups = read(EXP05 / "validation_selected_atoms.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for group in groups:
        real = sorted(map(int, group["candidate_atom_ids"].split(";")))
        for replicate in range(REPLICATES):
            control = matched_group(real, frequency, rng)
            rows.append(
                {
                    "protocol": group["protocol"],
                    "concept": group["concept"],
                    "replicate": replicate,
                    "group_size": len(real),
                    "real_atom_ids": json.dumps(real),
                    "random_atom_ids": json.dumps(control),
                    "real_mean_activation_frequency": float(frequency[real].mean()),
                    "random_mean_activation_frequency": float(
                        frequency[control].mean()
                    ),
                    "absolute_mean_frequency_difference": float(
                        abs(frequency[real].mean() - frequency[control].mean())
                    ),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "matched_random_groups.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    difference = np.array(
        [row["absolute_mean_frequency_difference"] for row in rows]
    )
    summary = {
        "real_groups": len(groups),
        "random_replicates_per_group": REPLICATES,
        "matched_random_groups": len(rows),
        "seed": SEED,
        "matching": (
            "same size; each constituent sampled from its 50 nearest unused atoms "
            "by training activation frequency"
        ),
        "training_data_only": True,
        "mean_absolute_group_frequency_difference": float(difference.mean()),
        "maximum_absolute_group_frequency_difference": float(difference.max()),
        "output": str(path),
    }
    (OUT / "matched_random_group_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

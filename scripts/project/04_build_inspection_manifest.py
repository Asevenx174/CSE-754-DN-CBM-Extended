"""EXP-04.2: select case-study atoms and their diagnostic SUN images."""

import csv
from collections import defaultdict
from pathlib import Path

import torch


EXP01 = Path("experiment-results/exp01")
EXP04 = Path("experiment-results/exp04")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
ATOMS_PER_CELL, IMAGES_PER_TYPE = 2, 5


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rank(row):
    g, s = float(row["geometry_cosine"]), float(row["auroc"])
    category = row["category"]
    if category == "high_geometry_high_detection":
        return s, g
    if category == "high_geometry_low_detection":
        return g, -s
    if category == "low_geometry_high_detection":
        return s, -g
    return -s, -g


def main():
    groups = defaultdict(list)
    for row in read(EXP04 / "case_candidates.csv"):
        groups[(row["protocol"], row["category"])].append(row)

    selected = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=rank, reverse=True)[:ATOMS_PER_CELL])
    write(EXP04 / "selected_case_atoms.csv", selected)

    split_rows = read(EXP01 / "sample_splits.csv")
    paths = {int(row["sample_id"]): row["relative_path"] for row in split_rows}
    test_ids = torch.tensor(
        [int(row["sample_id"]) for row in split_rows if row["split"] == "test"]
    )
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)

    manifest = []
    for row in selected:
        atom = int(row["atom_id"])
        concept = row["mapped_sun_attribute"]
        z = cache["sae_activations"][test_ids, atom]
        y = cache["binary_labels"][test_ids, concept_index[concept]].bool()
        agreement = cache["agreement"][test_ids, concept_index[concept]]
        choices = {
            "top_activation": torch.argsort(z, descending=True)[:IMAGES_PER_TYPE],
            "high_scoring_negative": torch.where(~y)[0][
                torch.argsort(z[~y], descending=True)[:IMAGES_PER_TYPE]
            ],
            "low_scoring_positive": torch.where(y)[0][
                torch.argsort(z[y])[:IMAGES_PER_TYPE]
            ],
        }
        for kind, indices in choices.items():
            for order, index in enumerate(indices.tolist(), start=1):
                sample_id = int(test_ids[index])
                manifest.append(
                    {
                        "protocol": row["protocol"],
                        "category": row["category"],
                        "atom_id": atom,
                        "assigned_name": row["assigned_name"],
                        "mapped_sun_attribute": concept,
                        "example_type": kind,
                        "rank": order,
                        "sample_id": sample_id,
                        "relative_path": paths[sample_id],
                        "activation": float(z[index]),
                        "binary_label": int(y[index]),
                        "worker_agreement": float(agreement[index]),
                    }
                )
    write(EXP04 / "inspection_manifest.csv", manifest)
    print(f"Selected {len(selected)} atoms and wrote {len(manifest)} image records.")


if __name__ == "__main__":
    main()

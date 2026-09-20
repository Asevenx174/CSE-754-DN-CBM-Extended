"""EXP-08.1: documented filtering for published-style SUN compound nodes."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


EXP01 = Path("experiment-results/exp01")
OUTPUT = Path("experiment-results/exp08")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
MIN_COSINE = 0.10


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    train_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "train"],
        dtype=torch.long,
    )
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    activations = cache["sae_activations"][train_ids]
    active = (activations > 0).any(dim=0)

    retained = defaultdict(list)
    audit = []
    for row in read(EXP01 / "sun_atom_assignments.csv"):
        atom = int(row["atom_id"])
        concept = row["sun_concept"]
        cosine = float(row["cosine"])
        reasons = []
        if not bool(active[atom]):
            reasons.append("inactive_on_training")
        if cosine < MIN_COSINE:
            reasons.append("cosine_below_0.10")
        keep = not reasons
        if keep:
            retained[concept].append(atom)
        audit.append(
            {
                "atom_id": atom,
                "sun_concept": concept,
                "cosine": cosine,
                "active_on_training": bool(active[atom]),
                "retained": keep,
                "exclusion_reason": ";".join(reasons),
            }
        )

    groups = [
        {
            "sun_concept": concept,
            "retained_atom_count": len(retained[concept]),
            "retained_atom_ids": json.dumps(sorted(retained[concept])),
        }
        for concept in concepts
    ]
    for filename, rows in (
        ("compound_node_filter_audit.csv", audit),
        ("compound_groups.csv", groups),
    ):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "status": "documented implementation, not exact reproduction",
        "sun_attributes": len(concepts),
        "atoms_before_filtering": len(audit),
        "atoms_retained": sum(row["retained"] for row in audit),
        "inactive_atoms": sum(not row["active_on_training"] for row in audit),
        "cosine_below_0.10": sum(row["cosine"] < MIN_COSINE for row in audit),
        "attributes_with_retained_atoms": sum(
            row["retained_atom_count"] > 0 for row in groups
        ),
        "alignment_cutoff": MIN_COSINE,
        "activity_filter": "at least one positive activation on training images",
    }
    (OUTPUT / "compound_filter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

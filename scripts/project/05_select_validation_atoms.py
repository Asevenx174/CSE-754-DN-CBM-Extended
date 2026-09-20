"""EXP-05.1: select one atom per frozen group using validation AUROC only."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score


EXP01 = Path("experiment-results/exp01")
OUTPUT = Path("experiment-results/exp05")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
MIN_GROUP_SIZE = 2


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def groups(protocol, valid):
    result = defaultdict(set)
    if protocol == "A":
        rows = read(EXP01 / "protocol_a_atom_concept_pairs.csv")
        for row in rows:
            if row["evaluation_eligible"] == "True":
                result[row["mapped_sun_attribute"]].add(int(row["atom_id"]))
    else:
        rows = read(EXP01 / "sun_atom_assignments.csv")
        for row in rows:
            if row["sun_concept"] in valid:
                result[row["sun_concept"]].add(int(row["atom_id"]))
    return {
        concept: sorted(atoms)
        for concept, atoms in result.items()
        if len(atoms) >= MIN_GROUP_SIZE
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    split = read(EXP01 / "sample_splits.csv")
    validation_rows = [
        row
        for row in split
        if row["split"].strip().lower() in {"val", "validation"}
    ]
    if not validation_rows:
        observed = sorted({row["split"] for row in split})
        raise ValueError(
            f"No validation rows found; observed split labels: {observed}"
        )
    val_ids = torch.tensor(
        [int(row["sample_id"]) for row in validation_rows],
        dtype=torch.long,
    )
    concepts = lines(EXP01 / "sun102_vocabulary.txt")
    valid = set(lines(EXP01 / "valid_concepts.txt"))
    concept_index = {name: index for index, name in enumerate(concepts)}
    activations = cache["sae_activations"][val_ids]
    labels = cache["binary_labels"][val_ids]

    selections, summary = [], {}
    for protocol in ("A", "B"):
        frozen_groups = groups(protocol, valid)
        for concept, atoms in sorted(frozen_groups.items()):
            y = labels[:, concept_index[concept]].numpy()
            scores = [
                (roc_auc_score(y, activations[:, atom].numpy()), atom)
                for atom in atoms
            ]
            best_score = max(score for score, _ in scores)
            selected = min(atom for score, atom in scores if score == best_score)
            selections.append(
                {
                    "protocol": protocol,
                    "concept": concept,
                    "candidate_count": len(atoms),
                    "candidate_atom_ids": ";".join(map(str, atoms)),
                    "selected_atom_id": selected,
                    "validation_auroc": best_score,
                    "tie_rule": "highest validation AUROC, then smallest atom ID",
                }
            )
        summary[protocol] = {
            "eligible_groups": len(frozen_groups),
            "minimum_group_size": MIN_GROUP_SIZE,
        }

    path = OUTPUT / "validation_selected_atoms.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=selections[0].keys())
        writer.writeheader()
        writer.writerows(selections)
    summary["selection_split"] = "validation"
    summary["test_metrics_computed"] = False
    (OUTPUT / "validation_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
